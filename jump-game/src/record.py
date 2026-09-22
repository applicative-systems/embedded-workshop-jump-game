"""Record a session, for demos and for tests.

`--record NAME` writes two files, because the two uses want different things:

* **NAME.mp4** -- the rendered game, exactly as it appeared on screen. For
  showing people.
* **NAME.jsonl** -- one line per frame: every detection, the takeoff events,
  and the game state. For tests.

The trace is the interesting half. It holds raw keypoints rather than derived
numbers, so a test can replay a real session through `TakeoffDetector` and
`JumpGame` with **no GPU and no camera** -- the same trick
`tests/fixtures/*.csv` plays, extended to carry whole poses and game events
instead of one hip coordinate. `load()` and `RecordedPose` below make that a
few lines.

Optionally **NAME-raw.mp4**: the camera frames before mirroring or drawing, so
a good session can be re-run through a newer model.
"""

import json
import os
import queue
import threading

# cv2 is imported inside _open_writers() rather than here on purpose. The
# bottom half of this module -- load(), RecordedPose -- is the replay path,
# and its whole point is that a test can drive the real detector and the real
# game with no model, no CUDA and no OpenCV. `nix flake check` builds a python
# with pytest and nothing else, so a module-level `import cv2` would make
# every fixture-replay test unimportable there.
FORMAT_VERSION = 2

# Frames spent measuring the real loop rate before the video writer is
# opened. A VideoWriter needs its frame rate up front and cannot be told
# later, and the loop rate here depends on the model, the resolution and what
# else the Jetson is doing -- so measure it rather than guess. These frames
# are dropped rather than buffered: half a second off the front of a
# recording costs nothing, where buffering 1280x960 frames costs ~4 MB each.
WARMUP = 20


class Recorder:
    """Writes the video and the trace. One per session."""

    # 48, not 240. At a 1707x960 stage a queued frame is 4.92 MB, so 240 of
    # them is a 1.18 GB latent allocation on an 8 GB board that is also
    # holding a CUDA context and a TensorRT engine. 48 is ~236 MB and still
    # four seconds of slack at 12 fps; the overflow path already counts and
    # reports what it drops.
    def __init__(self, base, width, height, meta, raw=False, raw_size=None,
                 queue_size=48):
        base = os.path.splitext(base)[0]
        self.video_path = f"{base}.mp4"
        self.raw_path = f"{base}-raw.mp4" if raw else None
        self.trace_path = f"{base}.jsonl"
        self.width, self.height = width, height
        # The raw video is the *camera* frame, which is not the stage: with
        # --stage 16:9 a 1280x960 camera sits in a 1707x960 canvas. Opening
        # the raw writer at the stage size made VideoWriter silently refuse
        # every frame, so --record-raw produced an empty file.
        self.raw_size = tuple(raw_size) if raw_size else (width, height)

        self._writer = None
        self._raw_writer = None
        self._dts = []
        self._warmup_left = WARMUP
        self.fps = None
        self.frames = 0
        self.entries = 0
        self.video_frames = 0
        self.dropped = 0

        self._trace = open(self.trace_path, "w")
        self._write({"kind": "header", "version": FORMAT_VERSION,
                     "width": width, "height": height, **meta})

        # Encoding a frame costs more than the game loop can spare, so it
        # happens off the loop -- one thread per video. A single thread
        # encoding both kept up with the stage video alone but dropped 24% of
        # frames once --record-raw doubled its work; two encoders run on two
        # cores. Unlike the display, a recording must not silently lose
        # frames: the queues are deep, and anything dropped is counted and
        # written to the trace as a "drop" line, so video and trace can still
        # be lined up afterwards.
        streams = ["_writer"] + (["_raw_writer"] if raw else [])
        self._qs = {s: queue.Queue(maxsize=queue_size) for s in streams}
        self._threads = [threading.Thread(target=self._run, args=(s,), daemon=True)
                         for s in streams]
        for t in self._threads:
            t.start()

    def _write(self, obj):
        self._trace.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def _run(self, stream):
        # Frames are only queued after the warm-up, by which time
        # _open_writers() has run, so the writer is never None here.
        q = self._qs[stream]
        while (frame := q.get()) is not None:
            getattr(self, stream).write(frame)

    def _open_writers(self):
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        size = (self.width, self.height)
        self._writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, size)
        if self.raw_path:
            self._raw_writer = cv2.VideoWriter(self.raw_path, fourcc, self.fps,
                                               self.raw_size)

    def add(self, dt, frame, entry, raw=None):
        """Record one rendered frame, and a trace line if there is one.

        The video is at the render rate and the trace is at the pose rate, and
        since the game was decoupled those are no longer the same number --
        so `entry` is None on the rendered frames that did not consume a new
        pose. `f` in each entry still indexes the video frame it was drawn on,
        which is what lines the two up again.

        The trace is written from the very first frame; only the *video* waits
        out the warm-up. Trace lines are cheap, and skipping them would leave
        a replay starting with an empty rolling baseline while the live run
        had history -- which shifts the first detected jump.
        """
        if entry is not None:
            entry["dt"] = round(dt, 4)
            self._write(entry)
            self.entries += 1
        self.frames += 1

        if self._warmup_left > 0:
            # Ignore the very first dt: it spans model warm-up, not a loop.
            if self._warmup_left < WARMUP:
                self._dts.append(dt)
            self._warmup_left -= 1
            if self._warmup_left == 0:
                mean = sum(self._dts) / len(self._dts) if self._dts else 1 / 30
                self.fps = round(max(1.0, min(120.0, 1 / mean)), 2)
                self._open_writers()
            return

        # A frame goes to both videos or to neither, so the two stay
        # frame-for-frame aligned with each other. `raw` is None only before
        # the first pose, and then there is nothing to put in the raw video.
        items = {"_writer": frame}
        if "_raw_writer" in self._qs and raw is not None:
            items["_raw_writer"] = raw
        # Checked, then put: this is the only producer, so a queue with room
        # here still has room below.
        if any(self._qs[s].full() for s in items):
            self.dropped += 1
            self._write({"kind": "drop", "f": self.frames - 1})
            return

        self.video_frames += 1
        # Copied, not queued by reference. `frame` is the reusable stage
        # canvas: the main loop starts overwriting it the moment this
        # returns, so the encoder thread was writing torn composites of
        # two game states -- and of states arbitrarily far apart whenever
        # it fell behind. ThreadedDisplay.prepare() avoids exactly this
        # hazard for the display; this is the same fix for the recorder.
        # 0.54 ms, and only ever paid with --record.
        #
        # `raw` is already a copy (jump_game takes it before mirroring),
        # so it must not be copied twice.
        items["_writer"] = frame.copy()
        for s, img in items.items():
            self._qs[s].put_nowait(img)

    def close(self, summary=None):
        for q in self._qs.values():
            q.put(None)
        for t in self._threads:
            t.join(timeout=10.0)
        for w in (self._writer, self._raw_writer):
            if w is not None:
                w.release()
        self._write({"kind": "summary", "frames": self.frames,
                     "entries": self.entries,
                     "video_frames": self.video_frames, "fps": self.fps,
                     "dropped": self.dropped, **(summary or {})})
        self._trace.close()

    def report(self):
        out = [f"recorded {self.entries} trace entries over {self.frames} frames; "
               f"{self.video_frames} video frames at {self.fps} fps",
               f"  {self.video_path}", f"  {self.trace_path}"]
        if self.raw_path:
            out.append(f"  {self.raw_path}")
        if self.dropped:
            out.append(f"  WARNING: {self.dropped} video frames dropped "
                       f"(encoder could not keep up; the trace is complete)")
        return "\n".join(out)


def frame_entry(index, t, pose, selected, jump, game, *, duck_in=False, descent=0.0):
    """Build one trace line. Keypoints are raw, so tests can re-derive anything.

    `duck_in` is what the *detector* said this frame and `duck` below is what
    the *game* did with it. They differ by the minimum hold, the release
    debounce and the rules in JumpGame -- and it is exactly that difference a
    replay needs to see, so both are written.
    """
    people = []
    for i in range(min(len(pose.xy), len(pose.box_xyxy))):
        kp = pose.kp_conf[i] if pose.kp_conf is not None else None
        people.append({
            "conf": round(float(pose.box_conf[i]), 3),
            "box": [round(float(v), 1) for v in pose.box_xyxy[i]],
            "kp": [
                [round(float(pose.xy[i][j][0]), 1), round(float(pose.xy[i][j][1]), 1),
                 round(float(kp[j]), 3) if kp is not None else 1.0]
                for j in range(len(pose.xy[i]))
            ],
        })
    return {
        "f": index,
        "t": round(t, 3),
        "people": people,
        "sel": selected,
        "jump": round(jump.height, 4) if jump else None,
        "duck_in": bool(duck_in),
        "duck": bool(getattr(game, "crouched", False)),
        "descent": round(descent, 4),
        "state": game.state,
        "score": game.score,
        "lift": round(game.player.lift, 4),
    }


class RecordedPose:
    """Duck-types the `Pose` that TakeoffDetector and JumpGame consume.

    Lets a test drive the real detector and the real game off a recording,
    with no model, no CUDA and no OpenCV capture in the picture.
    """

    __slots__ = ("xy", "kp_conf", "box_xyxy", "box_conf")

    def __init__(self, entry):
        people = entry["people"]
        self.xy = [[(k[0], k[1]) for k in p["kp"]] for p in people]
        self.kp_conf = [[k[2] for k in p["kp"]] for p in people]
        self.box_xyxy = [p["box"] for p in people]
        self.box_conf = [p["conf"] for p in people]

    def __len__(self):
        return len(self.xy)


def load(path):
    """Read a trace: returns (header, [frame entries], summary or None)."""
    header, frames, summary = None, [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.get("kind")
            if kind == "header":
                header = obj
            elif kind == "summary":
                summary = obj
            elif kind is None:
                frames.append(obj)
            # "drop" lines mark video frames the encoder lost; a replay
            # only needs the poses.
    return header, frames, summary
