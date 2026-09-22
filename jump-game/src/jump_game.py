"""Live jump game: webcam in, skeleton painted on a fullscreen video stream.

Packaged as `pkgs.jump-game`; run.sh runs the same code from a checkout, for
edit-and-rerun iteration without a nixos-rebuild.

    jump-game                         # see run.sh to run it from a checkout

Detection is GPU-only, like the rest of this repo: no CPU fallback.

Two things here are not the obvious choice, both for speed:

* Skeletons are drawn by hand rather than with ultralytics' `r.plot()`. That
  helper copies the frame and redraws boxes and text labels on the CPU every
  frame; pose_bench.py's own numbers show it costing more than TensorRT
  inference does, which is exactly the cost we are trying not to pay.

* Frames go to **ffplay over a pipe**, not `cv2.imshow`. This opencv *does*
  have a GUI now (GTK3), so `--imshow` works -- it is just slower: measured
  head to head at the 1707x960 stage, imshow+waitKey is 29.4 ms a frame
  against 16.0 ms for tobytes+pipe, and it cannot be threaded because its
  event loop has to own the main thread. Xorg here runs with `Disable "dri"`,
  so a GTK blit is going through software. ffplay also gives us fullscreen,
  q-to-quit and f-to-toggle for free.

* **The game does not run at the pose rate.** A pose worker thread owns the
  camera, the model, the detector and the camera layer; the main thread steps
  and draws the game at --render-fps. Inference is 16 ms and the camera layer
  another 4, so coupling them meant obstacles moved in 60 ms jumps -- 68 px at
  the opening speed, a strobe rather than a stutter. See loop.py for the
  handover, whose one idea is that poses are levels and takeoffs are edges.
"""

import argparse
import fcntl
import glob
import os
import re
import struct
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch

# Captured here, deliberately mid-import, because the next line has a side
# effect: importing ultralytics calls cv2.setNumThreads(1), which leaves
# every OpenCV call in this process single-threaded for the rest of the run.
# That is a reasonable default for a training dataloader and a bad one for
# us -- inference is on the GPU, so the CPU cores are sitting idle, and the
# whole render path is OpenCV. Measured on the Jetson: the camera downscale
# goes 2.7 ms -> 14.0 ms and the dim pass 1.1 ms -> 3.5 ms. main() puts it
# back; see --cv-threads.
CV_THREADS = cv2.getNumThreads()

from ultralytics import YOLO  # noqa: E402

import coco

import loop as looplib
import record as recording
from game import JumpGame
from takeoff import TakeoffDetector

WEIGHTS_DIR = os.environ.get("POSE_WEIGHTS_DIR", ".")
ENGINE_DIR = os.environ.get("POSE_ENGINE_DIR", ".")

WINDOW = "jump-game"

# Left limbs cyan, right limbs orange, torso/head green -- so it is obvious at
# a glance when the model swaps a side.
#
# Bright again. These were cut to about a third of full strength back when
# the skeleton was painted life-size over the whole stage, where it competed
# with the avatar for attention; it now lives in a small sharp panel of its
# own (see draw_inset), against an undimmed picture, where a third-strength
# hue is simply invisible. The knocking-back moved from the colour to the
# layout, which is the better place for it.
LIMBS, LEFT, RIGHT = coco.LIMBS, coco.LEFT_SIDE, coco.RIGHT_SIDE
C_LEFT, C_RIGHT, C_MID, C_BOX = (255, 220, 30), (30, 150, 255), (30, 255, 120), (90, 200, 90)
C_KP = (255, 255, 255)
C_PANEL = (150, 150, 150)  # the camera panel's frame and caption

KP_MIN = 0.5  # hide keypoints the model is only guessing at

# Camera controls are poked at directly through V4L2 ioctls rather than by
# shelling out to v4l2-ctl (one less runtime dependency to carry into the
# package) or via cv2's CAP_PROP_* (which can set a value but cannot ask what
# the valid range is). Encoding is the asm-generic one:
# _IOWR(dir=3, 'V', nr, sizeof).
V4L2_CID_FOCUS_ABSOLUTE = 0x009A090A
V4L2_CID_FOCUS_AUTO = 0x009A090C
V4L2_CID_ZOOM_ABSOLUTE = 0x009A090D
_QUERYCTRL_FMT = "II32siiiiI8x"  # struct v4l2_queryctrl, 68 bytes
_CTRL_FMT = "Ii"  # struct v4l2_control, 8 bytes
VIDIOC_QUERYCTRL = (3 << 30) | (struct.calcsize(_QUERYCTRL_FMT) << 16) | (ord("V") << 8) | 36
VIDIOC_G_CTRL = (3 << 30) | (struct.calcsize(_CTRL_FMT) << 16) | (ord("V") << 8) | 27
VIDIOC_S_CTRL = (3 << 30) | (struct.calcsize(_CTRL_FMT) << 16) | (ord("V") << 8) | 28


class V4l2Controls:
    """Minimal read/write access to one video device's controls.

    Every method treats "this camera does not have that control" as a normal
    answer (None) rather than an error, so the app still runs against a
    webcam that is not a PTZ unit.
    """

    def __init__(self, index):
        self.path = f"/dev/video{index}"
        try:
            self.fd = os.open(self.path, os.O_RDWR)
        except OSError as e:
            print(f"camera controls: cannot open {self.path} ({e})", flush=True)
            self.fd = None

    def range(self, cid):
        """(min, max) for `cid`, or None if unsupported."""
        if self.fd is None:
            return None
        try:
            q = fcntl.ioctl(
                self.fd, VIDIOC_QUERYCTRL,
                struct.pack(_QUERYCTRL_FMT, cid, 0, b"", 0, 0, 0, 0, 0),
            )
        except OSError:
            return None
        _id, _type, _name, lo, hi, _step, _default, _flags = struct.unpack(_QUERYCTRL_FMT, q)
        return lo, hi

    def get(self, cid):
        if self.fd is None:
            return None
        try:
            raw = fcntl.ioctl(self.fd, VIDIOC_G_CTRL, struct.pack(_CTRL_FMT, cid, 0))
        except OSError:
            return None
        return struct.unpack(_CTRL_FMT, raw)[1]

    def set(self, cid, value):
        """Set `cid`, returning what the camera actually settled on."""
        if self.fd is None:
            return None
        try:
            fcntl.ioctl(self.fd, VIDIOC_S_CTRL, struct.pack(_CTRL_FMT, cid, int(value)))
        except OSError:
            return None
        return self.get(cid)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def set_zoom(ctl, want):
    """Set zoom; `want` is an int or "min".

    Wide angle is what this game needs -- the player has to fit in frame
    while jumping, from a few metres back -- and the OBSBOT Tiny 2 remembers
    whatever zoom it was last left on, across power cycles included. So
    rather than trusting it, pin the zoom on every startup. The minimum is
    queried rather than assumed to be 0.
    """
    rng = ctl.range(V4L2_CID_ZOOM_ABSOLUTE)
    if rng is None:
        print("zoom: camera has no zoom control; leaving it alone", flush=True)
        return
    lo, hi = rng
    before = ctl.get(V4L2_CID_ZOOM_ABSOLUTE)
    target = lo if want == "min" else max(lo, min(hi, int(want)))
    after = ctl.set(V4L2_CID_ZOOM_ABSOLUTE, target)
    print(
        f"zoom: {before} -> {after}{' (widest)' if after == lo else ''}  [range {lo}..{hi}]",
        flush=True,
    )
    if after != target:
        print(f"zoom: camera refused {target}, kept {after}", flush=True)


def lock_focus(ctl, want):
    """Turn continuous autofocus off; `want` is "lock" or an absolute value.

    Autofocus is actively harmful here. The player is the only large moving
    thing in frame, so AF hunts exactly when they jump -- the moment the
    keypoints most need to be sharp -- and each hunt both blurs the frame and
    shifts the apparent scale slightly, which is the signal jump_detector.py
    measures against.

    "lock" freezes the lens where AF last left it rather than picking a
    distance for it, which is why open_camera() lets AF settle on the real
    scene for a moment before calling this. Note that focus_absolute reads 0
    on the Tiny 2 whether AF is on or off, so it cannot be used to report
    where the lens actually ended up -- do not read anything into that 0.
    """
    if ctl.range(V4L2_CID_FOCUS_AUTO) is None:
        print("focus: camera has no autofocus control; leaving it alone", flush=True)
        return
    was = ctl.get(V4L2_CID_FOCUS_AUTO)
    now = ctl.set(V4L2_CID_FOCUS_AUTO, 0)
    if now != 0:
        print(f"focus: camera refused to disable autofocus (still {now})", flush=True)
        return
    msg = f"focus: autofocus {'on -> off' if was else 'already off'}, lens frozen"
    if want != "lock":
        rng = ctl.range(V4L2_CID_FOCUS_ABSOLUTE) or (0, 100)
        target = max(rng[0], min(rng[1], int(want)))
        got = ctl.set(V4L2_CID_FOCUS_ABSOLUTE, target)
        msg = f"focus: autofocus off, focus_absolute -> {got} [range {rng[0]}..{rng[1]}]"
    print(msg, flush=True)


def limb_color(a, b):
    if a in LEFT and b in LEFT:
        return C_LEFT
    if a in RIGHT and b in RIGHT:
        return C_RIGHT
    return C_MID


def load_model(name, imgsz):
    """Load the TensorRT engine, building it first if there is not one.

    The engine is not optional. Eager PyTorch costs 44 ms a frame against the
    engine's 16 on this board -- two thirds of the loop, for the same numbers
    -- so running on the .pt weights is a mistake rather than a fallback, and
    the only thing a silent fallback would buy is a game that feels broken for
    a reason nobody can see.

    Engines are locked to one exact GPU, TensorRT version, precision and
    imgsz, and are therefore gitignored and built per device. Building one
    takes a few minutes and happens once.
    """
    path = f"{ENGINE_DIR}/{name}.engine"
    if not os.path.exists(path):
        print(f"no {path} yet -- compiling the TensorRT engine, "
              f"this takes a few minutes and happens once", flush=True)
        _build_engine(name, imgsz)

    model = YOLO(path)
    # YOLO() does not deserialise an engine: for a non-.pt suffix ultralytics
    # defers AutoBackend to the first predict(), so a stale engine -- one
    # built before a TensorRT bump, a driver change or a different --imgsz --
    # constructs perfectly happily and then explodes on the first camera
    # frame. Probe it here, on a blank frame the model will letterbox like
    # any other, so the failure lands at startup where it can be fixed.
    probe = np.zeros((imgsz, imgsz, 3), np.uint8)
    kwargs = dict(imgsz=imgsz, device="cuda", verbose=False)
    try:
        model.predict(probe, **kwargs)
    except Exception as e:  # TensorRT surfaces these as anything at all
        print(f"{path} did not load ({type(e).__name__}: {e}).\n"
              f"An engine is locked to the exact GPU, TensorRT version, "
              f"precision and imgsz it was built with, so this usually means "
              f"one of those changed. Rebuilding it now.", flush=True)
        os.remove(path)
        _build_engine(name, imgsz)
        model = YOLO(path)
        try:
            model.predict(probe, **kwargs)
        except Exception as e2:
            sys.exit(f"{path} still will not load after a rebuild: "
                     f"{type(e2).__name__}: {e2}")

    print(f"model: {path}", flush=True)
    return model, dict(device="cuda", verbose=False)


def _build_engine(name, imgsz):
    """Compile `name` to a TensorRT engine in ENGINE_DIR.

    Imported lazily and reused rather than reimplemented: pose_bench's version
    already handles the part that is easy to get wrong, which is that
    ultralytics writes the .engine next to the .pt it is exporting and
    WEIGHTS_DIR is a read-only Nix store path.
    """
    from pose_bench import export_engine

    export_engine(name, imgsz)


def open_camera(index, width, height, fps, fourcc, zoom, focus, settle):
    """Open /dev/video<index> at the requested mode, with controls pinned."""
    # CAP_V4L2 explicitly: the default backend may pick GStreamer, which
    # ignores the FOURCC/size/FPS properties set below.
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {index}")
    # MJPG rather than raw YUYV: uncompressed 720p60 does not fit through USB
    # and the camera silently drops to a fraction of the requested rate.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Keep the queue short: a deeper buffer trades latency we care about for
    # smoothness we do not. The player has to see themselves move *now*.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # After the format is set, not before: some UVC cameras reset their
    # controls when the stream is (re)configured.
    ctl = V4l2Controls(index)
    try:
        if zoom != "keep":
            set_zoom(ctl, zoom)
        if focus != "auto":
            # Pull frames first so autofocus has the real, wide-angle scene to
            # settle on -- the zoom above just changed it -- and only then
            # freeze the lens. Locking immediately would freeze it mid-hunt.
            if settle > 0:
                deadline = time.perf_counter() + settle
                while time.perf_counter() < deadline:
                    cap.read()
            lock_focus(ctl, focus)
    finally:
        ctl.close()

    print(
        f"camera {index}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
        f"x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps"
        f" ({fourcc})",
        flush=True,
    )
    return cap


def screen_size():
    """(w, h) of the X screen, or None if it cannot be determined.

    Needed because -fs alone is not enough here. ffplay asks for fullscreen
    through SDL, which sets _NET_WM_STATE_FULLSCREEN -- a hint only a *window
    manager* acts on. The demo box deliberately runs none (see common.nix), so
    nothing answers it and the window stays at the stage size on a larger
    screen. Sizing the window to the screen ourselves does not involve the WM
    at all, and with no WM to place it a screen-sized window lands at 0,0.
    """
    try:
        r = subprocess.run(
            ["xrandr", "--current"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    # "DP-1 connected primary 3840x2160+0+0 (normal ...)" -- take the geometry
    # of the primary output, or the first connected one if none is primary.
    best = None
    for line in r.stdout.splitlines():
        if " connected" not in line:
            continue
        for tok in line.split():
            m = re.match(r"^(\d+)x(\d+)\+\d+\+\d+$", tok)
            if m:
                size = (int(m.group(1)), int(m.group(2)))
                if "primary" in line:
                    return size
                best = best or size
                break
    return best


class FfplayDisplay:
    """Fullscreen sink: raw BGR frames down a pipe into ffplay.

    ffplay owns the window, so it also owns the keyboard: q/ESC quits, f
    toggles fullscreen. Quitting closes the pipe, which surfaces here as a
    broken pipe and ends the run -- that is the normal way out, not an error.
    """

    def __init__(self, width, height, fps, fullscreen=True):
        cmd = [
            "ffplay", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            # Everything on this line is latency: show each frame as it
            # arrives instead of buffering ahead, and drop rather than lag if
            # we ever outrun the display.
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-probesize", "32", "-analyzeduration", "0", "-framedrop",
            "-autoexit", "-window_title", WINDOW,
        ]
        if fullscreen:
            # -fs still goes in, so this keeps working under a window manager
            # (and f still toggles). -x/-y is what actually fills the screen
            # when there is none; -noborder keeps a WM, if one appears later,
            # from adding decoration that would push the picture off-screen.
            cmd.append("-fs")
            screen = screen_size()
            if screen:
                cmd += ["-x", str(screen[0]), "-y", str(screen[1]), "-noborder"]
            else:
                print("could not read the screen size; -fs only", flush=True)
        cmd += ["-i", "-"]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def prepare(self, frame):
        """Snapshot `frame` for a later present() on another thread.

        Serialising here rather than in the display thread is what makes the
        threaded handoff safe: the caller reuses its canvas the moment show()
        returns, so anything held by reference would be rewritten underneath
        the writer. This moves the copy rather than adding one -- the same
        tobytes() used to happen on the far side.
        """
        return frame.tobytes()

    def present(self, buf):
        """Return False once the viewer has gone away."""
        try:
            self.proc.stdin.write(buf)
            return True
        except (BrokenPipeError, ValueError):
            return False

    def show(self, frame):
        return self.present(self.prepare(frame))

    def close(self):
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        self.proc.wait(timeout=5)


class ImshowDisplay:
    """cv2.imshow sink. Only usable against an opencv built with a GUI."""

    def __init__(self, width, height, fps, fullscreen=True):
        del width, height, fps
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        self.fullscreen = fullscreen
        self._apply()

    def _apply(self):
        cv2.setWindowProperty(
            WINDOW,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
        )

    def prepare(self, frame):
        return frame.copy()

    def present(self, frame):
        return self.show(frame)

    def show(self, frame):
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord("f"):
            self.fullscreen = not self.fullscreen
            self._apply()
        return True

    def close(self):
        cv2.destroyAllWindows()


class Pose:
    """One frame's detections, pulled off the GPU exactly once.

    Both the renderer and the jump detector need these numbers, and a
    `.cpu().numpy()` per consumer is a real cost at 60 fps, so do it here and
    hand the arrays around.
    """

    __slots__ = ("xy", "kp_conf", "box_xyxy", "box_conf")

    def __init__(self, result):
        kps = result.keypoints
        empty = kps is None or kps.xy is None or not len(kps.xy)
        self.xy = [] if empty else kps.xy.cpu().numpy()
        self.kp_conf = None if empty or kps.conf is None else kps.conf.cpu().numpy()
        has_boxes = result.boxes is not None and len(result.boxes)
        self.box_xyxy = result.boxes.xyxy.cpu().numpy() if has_boxes else []
        self.box_conf = result.boxes.conf.cpu().numpy() if has_boxes else []

    def __len__(self):
        return len(self.xy)


class ThreadedCapture:
    """Reads the camera in a background thread, keeping only the newest frame.

    Two separate wins, both latency:

    * Grabbing stops being a serial stage. It measured ~11 ms against ~15 ms
      of inference, and almost all of that was the main loop sitting idle
      waiting for the sensor -- time inference could have been using.

    * V4L2 hands frames over in order, so a consumer slower than the camera
      accumulates staleness: every frame we take is one the sensor produced
      several intervals ago. Draining the queue continuously and keeping only
      the most recent frame means we always infer on the freshest one, which
      cuts real end-to-end delay by more than the frame-rate gain alone.

    Dropped frames are the *point*, not a failure -- they are the stale ones
    we deliberately skipped -- so they are counted and reported rather than
    warned about. cv2's read() releases the GIL while it waits on the device
    and decodes MJPG, so this genuinely runs in parallel.
    """

    def __init__(self, cap):
        self.cap = cap
        self._cv = threading.Condition()
        self._frame = None
        self._seq = 0
        self._alive = True
        self.dropped = 0
        self._taken = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            ok, frame = self.cap.read()
            with self._cv:
                if not self._alive:
                    return
                if not ok:
                    self._alive = False
                    self._cv.notify_all()
                    return
                if self._frame is not None:
                    self.dropped += 1  # never consumed; it was stale anyway
                self._frame, self._seq = frame, self._seq + 1
                self._cv.notify_all()

    def read(self, timeout=5.0):
        """Block until a frame newer than the last one we returned."""
        with self._cv:
            if not self._cv.wait_for(
                lambda: self._frame is not None or not self._alive, timeout
            ):
                return False, None
            if self._frame is None:
                return False, None
            frame, self._frame = self._frame, None
            self._taken += 1
            return True, frame

    def release(self):
        with self._cv:
            self._alive = False
            self._cv.notify_all()
        self._thread.join(timeout=1.0)
        self.cap.release()


class PoseWorker(threading.Thread):
    """Camera, model, detector and the camera layer, on their own clock.

    Everything here is a pure function of one camera frame, and all of it is
    slow: inference alone is 16 ms with the engine and the backdrop another 4.
    Running it in the render loop meant the game advanced once per inference,
    so obstacles moved in 60 ms jumps -- 68 px at the opening speed on a 1707
    stage, which is a strobe rather than a stutter.

    It publishes whole stage frames, never copies: the ring in loop.Shared
    moves buffer ownership instead. It has no deadline of its own, which is
    the reason the deadline-bound half stays on the main thread.
    """

    def __init__(self, cap, model, predict_kwargs, detector, backdrop, shared,
                 first, mirror=True, boxes=True, skeleton=True, raw=False,
                 limit=None):
        super().__init__(daemon=True)
        self.cap, self.model, self.kw = cap, model, predict_kwargs
        self.detector, self.backdrop, self.shared = detector, backdrop, shared
        self.sf = first
        self.mirror, self.boxes, self.skeleton = mirror, boxes, skeleton
        self.raw, self.limit = raw, limit
        self.alive = True
        self.frames = 0
        self.jumps = 0
        self.ducks = 0
        self.grab_t, self.infer_t, self.det_t, self.layer_t, self.pose_t = (
            deque(maxlen=60) for _ in range(5))
        self._lock_seq = -1

    def stop(self):
        self.alive = False

    def run(self):
        was_ducking = False
        prev = None
        while self.alive and self.sf is not None:
            if self.limit is not None and self.frames >= self.limit:
                break
            t0 = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok:
                print("capture delivered no frame; stopping", flush=True)
                break
            raw_frame = frame.copy() if self.raw else None
            if self.mirror:
                # The player is looking at themselves, so their right hand has
                # to be the on-screen figure's right hand.
                frame = cv2.flip(frame, 1)
            t1 = time.perf_counter()

            r = self.model.predict(frame, **self.kw)[0]
            # CUDA is async; without this the inference number is fiction and
            # its real cost lands in whichever bucket next touches the GPU.
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            pose = Pose(r)

            # The run belongs to whoever started it, and the lock has to be
            # taken on the frame the jump was seen on -- `selected` is still
            # them at that instant. So the renderer *arms* and this thread
            # locks itself; publishing "playing" and locking a frame later
            # would let a bystander win select() in between, which is the
            # exact trap takeoff.select() refuses nearest-to-previous over.
            playing, arm_lock, ack_seq = self.shared.intent()
            if self.detector.locked and not playing and ack_seq >= self._lock_seq:
                self.detector.release()
            jump = self.detector.update(pose)
            if jump is not None:
                self.jumps += 1
                print(f"jump #{self.jumps} {jump.height:.2f} bh "
                      f"({self.detector.using})", flush=True)
                if arm_lock and not self.detector.locked:
                    self.detector.lock()
                    self._lock_seq = self.frames
                # An edge, so it goes in the mailbox rather than on the frame.
                self.shared.post_jump(jump)
            ducking = self.detector.ducking
            if ducking and not was_ducking:
                self.ducks += 1
                print(f"duck #{self.ducks} {self.detector.descent:.2f} bh", flush=True)
            was_ducking = ducking
            t3 = time.perf_counter()

            sf = self.sf
            # Backdrop first: its blur source is the same half-size copy the
            # panel is scaled from, so the full frame is read once. And the
            # skeleton goes on *after*, or a smeared one ends up in the room.
            self.backdrop.step(frame, sf.slot, sf.panel)
            draw_panel_skeleton(sf.panel, pose, frame.shape[1],
                                boxes=self.boxes, skeleton=self.skeleton)
            sf.pose, sf.selected = pose, self.detector.selected
            sf.ducking = ducking
            sf.present = self.detector.selected is not None
            sf.descent, sf.using = self.detector.descent, self.detector.using
            sf.raw = raw_frame
            sf.seq = self.frames
            self.frames += 1
            t4 = time.perf_counter()

            # Only from the second frame on, so every bucket has the same
            # sample set as the interval it is meant to tile: the first frame
            # has no interval, and it is also the slowest one there is.
            if prev is not None:
                self.grab_t.append(t1 - t0)
                self.infer_t.append(t2 - t1)
                self.det_t.append(t3 - t2)
                self.layer_t.append(t4 - t3)
                self.pose_t.append(t4 - prev)
            prev = t4
            self.sf = self.shared.publish(sf)
        self.shared.finish()


class ThreadedDisplay:
    """Hands frames to a sink from a background thread, newest-first.

    Writing a 1280x960 BGR frame down the ffplay pipe is ~3.7 MB and measured
    ~10 ms -- a third of the loop, spent blocking on a pipe while the GPU sat
    idle. The main loop drops the frame off here and moves straight on to the
    next inference.

    Keeps one slot, not a queue: a queue would trade the latency we are
    trying to remove for smoothness nobody asked for. If the sink falls
    behind, the pending frame is discarded in favour of the newer one, which
    is the right answer for a mirror the player is standing in front of.

    What is handed across is the sink's own snapshot of the frame, taken on
    the *caller's* thread, never the caller's buffer. The game composites
    into one reusable stage canvas and starts on the next frame the instant
    show() returns, so a reference handed over here would be rewritten while
    the sink was still reading it.

    `dropped` used to mean "a game state the player never saw", back when the
    game advanced once per displayed frame. It does not any more: the render
    loop redraws at its own rate, so a dropped frame is a redundant redraw of
    a state that will be drawn again a few milliseconds later. Worth watching
    as a sign the pipe cannot keep up with --render-fps, not as an alarm.
    """

    def __init__(self, sink):
        self.sink = sink
        self._cv = threading.Condition()
        self._pending = None
        self._alive = True
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._pending is not None or not self._alive)
                if not self._alive and self._pending is None:
                    return
                frame, self._pending = self._pending, None
            if not self.sink.present(frame):
                with self._cv:
                    self._alive = False
                return

    def show(self, frame):
        with self._cv:
            if not self._alive:
                return False
        snap = self.sink.prepare(frame)  # outside the lock; see the docstring
        with self._cv:
            if not self._alive:
                return False
            if self._pending is not None:
                self.dropped += 1
            self._pending = snap
            self._cv.notify()
            return True

    def close(self):
        with self._cv:
            self._alive = False
            self._cv.notify_all()
        self._thread.join(timeout=2.0)
        self.sink.close()


def stage_size(spec, cam_w, cam_h):
    """(width, height) of the rendered stage for `spec`.

    Three forms, and the difference between the last two matters:

    * "off"      -- render at the camera's own size.
    * "16:9"     -- keep the camera pixel-for-pixel and only widen the
                    canvas around it. The extra width is runway: obstacles
                    enter at the right edge of the *stage*, not of the
                    webcam, so they are visible for longer without slowing
                    the game down or shrinking the picture.
    * "1280x720" -- render at exactly that, scaling the camera to fit inside
                    it. This is the knob for when the display is the
                    bottleneck: every pixel is three bytes down the ffplay
                    pipe every single frame, so 1280x720 is 2.8 MB a frame
                    against 4.9 MB for a 1707x960 stage.

    Either way the camera image is shown whole -- letterboxed, never cropped.
    The player has to be able to see themselves.
    """
    if spec.lower() in ("off", "none"):
        return cam_w, cam_h
    explicit = "x" in spec.lower()
    sep = "x" if explicit else ":"
    try:
        a, b = (float(v) for v in spec.lower().split(sep))
        if a <= 0 or b <= 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--stage wants 16:9, 1280x720 or off, not {spec!r}"
        ) from None
    if explicit:
        return int(a), int(b)
    # Aspect only: keep the camera unscaled, widen (never narrow) around it.
    return max(cam_w, int(round(cam_h * a / b))), cam_h


def fit_camera(cam_w, cam_h, stage_w, stage_h):
    """Where the camera image sits in the stage: (w, h, x, y, scale)."""
    s = min(stage_w / cam_w, stage_h / cam_h)
    cw, ch = max(1, round(cam_w * s)), max(1, round(cam_h * s))
    return cw, ch, (stage_w - cw) // 2, (stage_h - ch) // 2, s


def draw_skeletons(frame, pose, boxes=True, dx=0, dy=0, scale=1.0, thick=2):
    """Paint every detected person's skeleton onto `frame`, in place.

    Keypoints arrive in camera pixels, so `scale` then (`dx`, `dy`) map them
    onto wherever the camera image ended up inside the stage.
    """
    def at(pt):
        return int(pt[0] * scale) + dx, int(pt[1] * scale) + dy
    xy, conf, box_xyxy = pose.xy, pose.kp_conf, pose.box_xyxy

    for i in range(len(xy)):
        pts = xy[i]
        c = conf[i] if conf is not None else None
        # A keypoint the model could not place comes back as (0, 0) rather
        # than as missing, so filter on that as well as on confidence.
        good = [
            (c is None or c[j] >= KP_MIN) and (pts[j][0] > 0 or pts[j][1] > 0)
            for j in range(len(pts))
        ]
        for a, b in LIMBS:
            if good[a] and good[b]:
                cv2.line(
                    frame,
                    at(pts[a]),
                    at(pts[b]),
                    limb_color(a, b),
                    thick,
                    cv2.LINE_AA,
                )
        for j in range(len(pts)):
            if good[j]:
                cv2.circle(frame, at(pts[j]), thick + 1, C_KP, -1, cv2.LINE_AA)
        if boxes and i < len(box_xyxy):
            bx = box_xyxy[i]
            cv2.rectangle(frame, at(bx[:2]), at(bx[2:]), C_BOX, max(1, thick - 1))
    return len(xy)


class Backdrop:
    """The camera layer: the panel, and the softened room behind the game.

    Both are pure functions of one camera frame and together they were 12.3 ms
    of a 13 ms draw -- the game itself is 1.2 ms. They live in one object
    because they share their expensive intermediate: the panel and the blur
    are both derived from the same half-resolution copy, so the full frame is
    read exactly once.

    Every buffer is allocated once. `cv2.resize` and `pyrDown` allocate on
    every call otherwise, which at 60 rendered frames a second is a lot of
    4 MB garbage.
    """

    def __init__(self, cam_w, cam_h, slot_w, slot_h, stage_w, frac, lut):
        self.lut = lut
        # Halve until another halving would undershoot the panel. Exact
        # powers of two only -- see step().
        self.halves = []
        w, h = cam_w, cam_h
        pw = max(80, int(stage_w * frac)) if frac > 0 else 0
        while w % 2 == 0 and h % 2 == 0 and w // 2 >= max(pw, 8):
            w, h = w // 2, h // 2
            self.halves.append(np.empty((h, w, 3), np.uint8))
        self.half = self.halves[-1] if self.halves else None

        # The panel's *shape*, not a buffer: with the pose worker publishing
        # into a ring, each stage frame owns its own panel, or the worker
        # would overwrite the one the renderer is holding.
        self.panel_shape = None
        if frac > 0:
            self.panel_shape = (max(60, round(pw * cam_h / cam_w)), pw, 3)

        # The blur source, and the greyscale ladder the backdrop is built on.
        src_w, src_h = (self.half.shape[1], self.half.shape[0]) if self.half is not None else (cam_w, cam_h)
        self.small = np.empty((max(8, src_h // 5), max(8, src_w // 5)), np.uint8)
        self.grey_slot = np.empty((slot_h, slot_w), np.uint8)

    def step(self, cam, slot, panel=None):
        """Fill `slot` with the softened room, and `panel` with the inset.

        Halving with an exact-2x INTER_AREA rather than pyrDown, which is the
        correction that made this cheap: an integer-ratio INTER_AREA is a 2x2
        box filter and takes resizeAreaFast_'s SIMD path at 0.36 ms, where
        pyrDown's 5x5 Gaussian is 5.67 ms for the same 1280x960 -> 640x480.
        Both filter properly; only one of them is fast. (An INTER_AREA at a
        *non*-integer ratio is the slow general path, which is what made
        pyrDown look good when this was first written.)
        """
        src = cam
        for buf in self.halves:
            cv2.resize(src, (buf.shape[1], buf.shape[0]), dst=buf,
                       interpolation=cv2.INTER_AREA)
            src = buf

        if panel is not None:
            cv2.resize(src, (panel.shape[1], panel.shape[0]), dst=panel,
                       interpolation=cv2.INTER_LINEAR)

        # Grey first, at a fifth of the half-size, so the dim LUT and the
        # colour drain both run over a hundredth of the pixels.
        grey_small = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        cv2.resize(grey_small, (self.small.shape[1], self.small.shape[0]),
                   dst=self.small, interpolation=cv2.INTER_AREA)
        if self.lut is not None:
            cv2.LUT(self.small, self.lut, dst=self.small)

        # Upscale ONE channel and fan out to three at the end, rather than
        # upscaling three. Linear interpolation is per-channel and all three
        # are equal here, so the result is identical -- and it is 0.97 ms
        # against 3.73 ms, because two thirds of the work was interpolating
        # copies of a number we already had.
        cv2.resize(self.small, (slot.shape[1], slot.shape[0]), dst=self.grey_slot,
                   interpolation=cv2.INTER_LINEAR)
        cv2.cvtColor(self.grey_slot, cv2.COLOR_GRAY2BGR, dst=slot)


def draw_panel_skeleton(panel, pose, cam_w, boxes=True, skeleton=True):
    """Paint the detection onto the camera panel. Runs on the pose worker.

    The skeleton lives here and nowhere else, and that is the whole point.
    Painted life-size over the stage it made a second figure -- bigger, more
    central and more human-shaped than the avatar -- and every player seeing
    the game for the first time watched that one instead. Shrunk into a
    labelled panel it keeps all of its diagnostic value; a sharp small
    skeleton is easier to read than a dim large one.
    """
    if not skeleton or panel is None:
        return
    iw = panel.shape[1]
    draw_skeletons(panel, pose, boxes=boxes, scale=iw / cam_w,
                   thick=max(1, round(iw / 220)))


def draw_inset(view, panel):
    """Blit the finished panel onto the stage, with its frame and caption.

    Drawn after the game, so it is a panel bolted over the scene rather than
    scenery: an obstacle on its way out of frame passes behind it.
    """
    if panel is None:
        return
    h, w = view.shape[:2]
    ih, iw = panel.shape[:2]
    x, y = int(w * 0.012), int(h * 0.022)
    view[y:y + ih, x:x + iw] = panel
    cv2.rectangle(view, (x - 2, y - 2), (x + iw + 2, y + ih + 2), C_PANEL, 2)
    f, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.55 * h / 700
    cv2.putText(view, "CAMERA", (x + 10, y + ih - 12), f, fs, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(view, "CAMERA", (x + 10, y + ih - 12), f, fs, C_PANEL, 1, cv2.LINE_AA)


def draw_hud(frame, lines):
    """Bottom-left stats, outlined so they stay readable over anything."""
    y = frame.shape[0] - 14 - 26 * (len(lines) - 1)
    for line in lines:
        cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


def mean_ms(samples):
    return 1000 * sum(samples) / len(samples) if samples else 0.0


def fps_of(ms):
    return 1000 / ms if ms > 0 else 0.0


# Zones are resolved by name, not by number: the thermal_zone*/ numbering is
# not stable across kernels, and this board leaves its three cv*-thermal zones
# unwired -- they exist and read back an error. Resolving to nothing is what
# keeps the HUD line unchanged on a dev machine that is not a Jetson.
_ZONES = None
_THERM = (0.0, "")


def thermals(period=1.0):
    """`cpu 62C  gpu 61C  tj 62C`, re-read at most once a second.

    tj is the one to watch: junction temperature, the hottest point on the die
    and what the throttle keys off (95 C active, 104.5 C critical). Under
    MAXN_SUPER there is no power cap, so heat is the only thing holding the
    clocks down. The sensors move far slower than the render loop, hence the
    cache -- a read per frame would be 60x the syscalls for the same digits.
    """
    global _ZONES, _THERM
    if _ZONES is None:
        want = {"cpu-thermal": "cpu", "gpu-thermal": "gpu", "tj-thermal": "tj"}
        _ZONES = []
        for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            try:
                label = want.get(open(f"{z}/type").read().strip())
            except OSError:
                continue
            if label:
                _ZONES.append((label, f"{z}/temp"))

    now = time.monotonic()
    if now - _THERM[0] < period:
        return _THERM[1]
    out = []
    for label, path in _ZONES:
        try:
            out.append(f"{label} {int(open(path).read()) / 1000:.0f}C")
        except (OSError, ValueError):
            pass  # sensor unwired or read back junk -- just drop it this tick
    _THERM = (now, "  ".join(out))
    return _THERM[1]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="yolo26n-pose")
    p.add_argument("--camera", type=int, default=0, help="/dev/video<N>")
    p.add_argument("--source", help="play this video file instead of the camera")
    # 4:3, not 16:9. On the OBSBOT Tiny 2 the 16:9 modes are a vertical crop
    # of the same sensor width -- 1280x720 and 1280x960 see exactly as much
    # left-to-right, but 720 throws away the top and bottom of the frame.
    # Headroom is the one thing a jump game cannot afford to lose, and the
    # model letterboxes to a square 640 either way, so the taller mode is
    # free at inference time.
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--fps", type=int, default=60, help="requested camera fps")
    p.add_argument(
        "--render-fps",
        type=int,
        default=60,
        help="how often the game is stepped and drawn, independently of the "
        "pose model. The obstacles and the jump arc move at this rate; the "
        "camera layer and the avatar's limbs update whenever a pose lands. "
        "0 uncaps it, for benchmarking",
    )
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument(
        "--zoom",
        default="min",
        help="camera zoom at startup: 'min' (widest, the default), 'keep', or a number",
    )
    p.add_argument(
        "--focus",
        default="lock",
        help="'lock' (autofocus off, lens frozen where it settled -- the default), "
        "'auto' (leave autofocus on), or a number to set focus_absolute",
    )
    p.add_argument(
        "--focus-settle",
        type=float,
        default=1.5,
        metavar="S",
        help="seconds of streaming to let autofocus settle before locking it",
    )
    p.add_argument("--imgsz", type=int, default=640, help="must match the engine's export imgsz")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--no-mirror", action="store_true", help="do not flip horizontally")
    p.add_argument("--windowed", action="store_true", help="do not go fullscreen")
    p.add_argument("--no-boxes", action="store_true")
    p.add_argument("--no-skeleton", action="store_true",
                   help="hide the pose overlay inside the camera panel")
    p.add_argument(
        "--inset",
        type=float,
        default=0.26,
        help="width of the camera panel, as a fraction of the stage; 0 hides it "
        "entirely, which also hides the skeleton with it",
    )
    p.add_argument(
        "--signal",
        default="ankle",
        choices=("ankle", "hip"),
        help="which keypoints drive takeoff detection; ankle has ~4x the "
        "signal-to-noise and falls back to hip when the feet are not visible",
    )
    p.add_argument(
        "--duck-signal",
        default="shoulder",
        choices=("shoulder", "boxtop", "off"),
        help="which signal drives duck detection; shoulders are the only pair "
        "that never drops below the confidence floor and the only one the "
        "arms do not move. 'off' disables duck detection entirely",
    )
    p.add_argument(
        "--no-duck",
        action="store_true",
        help="no ceiling bars: the game is exactly what it was before ducking "
        "existed. Also the switch to bisect with when the detector misbehaves",
    )
    p.add_argument(
        "--ground",
        type=float,
        default=0.96,
        help="the ground line's height, as a fraction from the top of the frame. "
        "Low on the frame by design: the arc has to clear obstacles that are "
        "nearly as tall as the avatar, and the room for it comes from here",
    )
    p.add_argument("--imshow", action="store_true", help="use cv2.imshow instead of ffplay")
    p.add_argument(
        "--cv-threads",
        type=int,
        default=0,
        help="OpenCV worker threads for the render path. 0 restores whatever "
        "OpenCV picked before ultralytics forced it to 1",
    )
    p.add_argument(
        "--stage",
        default="16:9",
        help="the rendered stage: an aspect (16:9) widens the canvas around "
        "the camera at full resolution; an explicit size (1280x720) renders "
        "at exactly that, scaling the camera down to fit, which cuts what "
        "goes down the display pipe; 'off' uses the camera's own size. The "
        "camera image is always shown whole and centred, never cropped",
    )
    p.add_argument(
        "--dim",
        type=float,
        default=0.26,
        help="scale the room down to this brightness before drawing the game on "
        "top of it; 1.0 leaves it alone. Lower than it used to be (0.45), "
        "because the room is now blurred and greyed as well -- it is a "
        "backdrop rather than a picture you are meant to read, and what you "
        "are meant to read is in the camera panel",
    )
    p.add_argument("--limit", type=int, help="stop after N frames (for timing runs)")
    p.add_argument(
        "--record",
        metavar="NAME",
        help="record the session to NAME.mp4 (the rendered game) and NAME.jsonl "
        "(a per-frame trace that replays without a GPU)",
    )
    p.add_argument(
        "--record-raw",
        action="store_true",
        help="also write NAME-raw.mp4: the camera frames before mirroring and "
        "drawing",
    )
    p.add_argument(
        "--no-thread",
        action="store_true",
        help="read the camera in the main loop instead of a background thread",
    )
    a = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device -- this is GPU-only by design")

    model, predict_kwargs = load_model(a.model, a.imgsz)
    predict_kwargs.update(imgsz=a.imgsz, conf=a.conf)

    if a.source:
        cap = cv2.VideoCapture(a.source)
        if not cap.isOpened():
            sys.exit(f"cannot open {a.source}")
    else:
        cap = open_camera(
            a.camera, a.width, a.height, a.fps, a.fourcc, a.zoom, a.focus, a.focus_settle
        )

    ok, frame = cap.read()
    if not ok:
        sys.exit("capture opened but delivered no frames")
    h, w = frame.shape[:2]
    # Only for the live camera: on a file there is no staleness to shed, and
    # dropping frames would just skip parts of the clip.
    if not a.source and not a.no_thread:
        cap = ThreadedCapture(cap)

    # The first inference pays CUDA context init and model upload; spend it
    # before the window exists, so the stream does not open with a stall.
    print("warming up ...", flush=True)
    model.predict(frame, **predict_kwargs)
    torch.cuda.synchronize()

    # The stage is as tall as the camera frame and usually wider; the camera
    # image is centred in it and everything is drawn onto the stage, so this
    # is the size that the recorder, the display and the game all work in.
    # Undo ultralytics' cv2.setNumThreads(1); see the note by the import.
    cv2.setNumThreads(a.cv_threads if a.cv_threads > 0 else CV_THREADS)

    stage_w, stage_h = stage_size(a.stage, w, h)
    cam_w, cam_h, ox, oy, _scale = fit_camera(w, h, stage_w, stage_h)
    canvas = (np.zeros((stage_h, stage_w, 3), np.uint8)
              if (stage_w, stage_h) != (w, h) else None)

    rec = None
    if a.record:
        rec = recording.Recorder(
            a.record, stage_w, stage_h,
            meta=dict(model=a.model, engine=f"{ENGINE_DIR}/{a.model}.engine",
                      imgsz=a.imgsz, signal=a.signal,
                      duck_signal=a.duck_signal, bars=not a.no_duck,
                      ground=a.ground, conf=a.conf, source=a.source or f"camera{a.camera}"),
            raw=a.record_raw, raw_size=(w, h),
        )
        print(f"recording to {rec.video_path} + {rec.trace_path}", flush=True)

    # The render rate, not the camera rate. ffplay's -framerate sets the clock
    # it paces its pipe against, so feeding it the camera's requested 60 while
    # actually delivering 16 made it the one number in the pipeline that was
    # furthest from true.
    sink = (ImshowDisplay if a.imshow else FfplayDisplay)(
        stage_w, stage_h, a.render_fps or 120, fullscreen=not a.windowed
    )
    # imshow has to stay on the main thread (its event loop is not
    # thread-safe); the ffplay pipe has no such constraint.
    if not a.imshow and not a.no_thread:
        sink = ThreadedDisplay(sink)
    print(
        f"running {stage_w}x{stage_h}"
        + (f" (camera {w}x{h} drawn {cam_w}x{cam_h} at {ox},{oy})"
           if canvas is not None else "")
        + " -- q or ESC quits the window, f toggles fullscreen",
        flush=True,
    )

    # fps here only scales the jump's reported time_s; the state machine
    # itself counts frames. It is the *pose* rate, which is neither the camera
    # request nor the render rate -- and since nothing reads time_s, an
    # honest-ish estimate beats the 60 that used to be passed while the real
    # rate was 16.
    detector = TakeoffDetector(fps=a.render_fps or 30, signal=a.signal,
                               conf_min=a.conf, duck_signal=a.duck_signal)
    game = JumpGame(ground=a.ground, aspect=stage_w / stage_h, bars=not a.no_duck)
    # Dimming through a lookup table rather than convertScaleAbs: same result,
    # measured 2.1 ms against 5.3 ms per 1707x960 stage on the Jetson, and it
    # runs on every single frame so that difference is worth having.
    dim_lut = np.clip(np.arange(256) * a.dim, 0, 255).astype(np.uint8) if a.dim < 1.0 else None
    backdrop = Backdrop(w, h, cam_w, cam_h, stage_w, a.inset, dim_lut)

    # Three stage buffers: one the renderer is holding, one published and
    # waiting, one the worker is filling. Each carries its own panel, because
    # the worker would otherwise overwrite the panel being drawn.
    ring = []
    for _ in range(3):
        bgr = np.zeros((stage_h, stage_w, 3), np.uint8)
        ring.append(looplib.StageFrame(
            bgr=bgr,
            slot=bgr[oy:oy + cam_h, ox:ox + cam_w],
            panel=(np.empty(backdrop.panel_shape, np.uint8)
                   if backdrop.panel_shape else None)))
    shared = looplib.Shared(ring[1:])
    worker = PoseWorker(cap, model, predict_kwargs, detector, backdrop, shared,
                        ring[0], mirror=not a.no_mirror, boxes=not a.no_boxes,
                        skeleton=not a.no_skeleton,
                        raw=bool(rec and a.record_raw), limit=a.limit)

    view = canvas if canvas is not None else np.zeros((stage_h, stage_w, 3), np.uint8)

    # Rolling windows, so the HUD reacts within a second rather than
    # averaging the whole session into one flat number. The render loop's own
    # buckets; the worker keeps its own for grab and inference.
    wait_t, step_t, draw_t, show_t, loop_t = (deque(maxlen=120) for _ in range(5))
    frames = 0
    started = time.perf_counter()
    prev_end = None
    prev = started
    held = None
    pending = []
    period = 1.0 / a.render_fps if a.render_fps > 0 else 0.0
    deadline = started
    # The default 5 ms switch interval can park this thread behind one of the
    # worker's Python slices for a third of a 60 fps budget, which shows up as
    # jitter. 1 ms costs nothing measurable.
    sys.setswitchinterval(0.001)
    worker.start()

    try:
        while True:
            now = time.perf_counter()
            if period:
                # Wake on the deadline OR on a new pose, whichever comes
                # first, so decoupling never costs input latency: a takeoff
                # that lands mid-interval is drawn at once rather than up to a
                # frame later.
                shared.wait(deadline - now)
            sf = shared.take(held)
            if sf is not None:
                held = sf
                game.set_pose(held.pose, held.selected)
            if held is None:
                if shared.done:
                    break
                continue

            now = time.perf_counter()
            if prev_end is not None:
                wait_t.append(now - prev_end)
            dt, prev = now - prev, now
            # Edges are drained once and consumed one per tick; levels ride on
            # the frame and are simply sampled.
            pending += shared.drain_jumps()
            lift = pending.pop(0).height if pending else None
            game.update(dt, lift, held.present, ducking=held.ducking)
            # The renderer arms the lock; the worker takes it on the frame it
            # sees the jump. ack_seq tells the worker we have seen the frame
            # the lock was taken on, so a release cannot race the start.
            shared.set_intent(
                playing=game.state == JumpGame.PLAYING,
                arm_lock=(game.state == JumpGame.WAITING
                          or (game.state == JumpGame.OVER
                              and game.since_over >= JumpGame.OVER_LOCKOUT)),
                ack_seq=held.seq,
            )
            t1 = time.perf_counter()

            np.copyto(view, held.bgr)
            game.draw(view)
            draw_inset(view, held.panel)
            hud = f"{fps_of(mean_ms(worker.pose_t)):.0f} fps yolo"
            if loop_t:
                hud += f"   {fps_of(mean_ms(loop_t)):.0f} fps loop"
            if getattr(sink, "dropped", 0):
                hud += f"   {sink.dropped} dropped"
            if temps := thermals():
                hud += f"   {temps}"
            draw_hud(view, [hud])
            t2 = time.perf_counter()

            if rec is not None:
                rec.add(dt, view,
                        recording.frame_entry(frames, now - started, held.pose,
                                              held.selected, None, game,
                                              duck_in=held.ducking,
                                              descent=held.descent)
                        if sf is not None else None,
                        held.raw)
            alive = sink.show(view)
            t3 = time.perf_counter()

            step_t.append(t1 - now)
            draw_t.append(t2 - t1)
            show_t.append(t3 - t2)
            if prev_end is not None:
                loop_t.append(t3 - prev_end)
            prev_end = t3

            if not alive:
                break
            frames += 1
            if period:
                # max(), so a stall does not bank frames and then sprint.
                deadline = max(deadline + period, now)
            if shared.done and shared.take(held) is None:
                break
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        shared.finish()
        worker.join(timeout=2.0)
        cap.release()
        sink.close()
        if rec is not None:
            rec.close(summary=dict(jumps=worker.jumps, ducks=worker.ducks,
                                   best=max(game.best, game.score)))

    wall = time.perf_counter() - started
    if frames:
        # Two clocks now, because they are genuinely two loops. The worker's
        # buckets tile its camera frame; the renderer's tile its own tick.
        pose = [("grab", worker.grab_t), ("infer", worker.infer_t),
                ("detect", worker.det_t), ("layer", worker.layer_t)]
        # "wait" is the deliberate idle between deadlines, not unmeasured
        # work -- but it has to be in the tiling or the self-check below
        # reports the loop's own pacing as a hole.
        rend = [("wait", wait_t), ("step", step_t), ("draw", draw_t),
                ("show", show_t)]
        print(
            f"\n{worker.frames} poses / {frames} frames in {wall:.1f}s\n"
            f"pose   " + "  ".join(f"{n} {mean_ms(d):.1f} ms" for n, d in pose)
            + f"  ->  {fps_of(mean_ms(worker.pose_t)):.1f} fps\n"
            f"render " + "  ".join(f"{n} {mean_ms(d):.1f} ms" for n, d in rend)
            + f"  ->  {fps_of(mean_ms(loop_t)):.1f} fps\n"
            f"cv2 threads {cv2.getNumThreads()}"
            f" | {shared.overruns} poses superseded before the renderer saw them"
            + (f" | dropped {cap.dropped} stale frames"
               if isinstance(cap, ThreadedCapture) else ""),
            flush=True,
        )
        # The self-check that stops these numbers drifting back into fiction:
        # each set is meant to tile its own loop with no gap, so a meaningful
        # shortfall means something is being done that nothing is measuring --
        # which is exactly the state this replaced.
        for name, parts, total in (("pose", pose, worker.pose_t),
                                   ("render", rend, loop_t)):
            got, acc = mean_ms(total), sum(mean_ms(d) for _, d in parts)
            if got > 0 and abs(got - acc) / got > 0.05:
                print(f"  WARNING: {name} loop has {got - acc:+.1f} ms "
                      f"unaccounted ({100 * (got - acc) / got:+.0f}%)", flush=True)
    if rec is not None:
        print(rec.report(), flush=True)


if __name__ == "__main__":
    main()
