"""Takeoff detection for the game, on the ankle signal.

jump_detector.py stays the reference implementation and keeps its tests; this
is the variant the *game* runs on, and it differs in one way that measurement
justified. Traced over both test videos with yolo11n-pose, per-frame lift
normalised the same way (rolling-median baseline / bbox height):

    signal    kp conf    noise    peak     SNR
    hip         0.98     0.047    0.26     5.2 / 6.0
    knee        0.86     0.060    0.37     6.6 / 5.9
    ankle       0.77     0.025    0.54    27.1 / 17.7

Ankles win on both terms at once: roughly twice the excursion and *half* the
noise, for 3-5x the signal-to-noise, even though ankle keypoints are the
least confident of the three. Standing still, feet are planted and the true
signal really is zero, while hips bob with breathing and weight shifts; and
in a jump people tuck their legs, so ankles rise from the body translating
upward *and* from the knees folding.

That headroom is spent on latency. Sweeping rise/confirm over both clips
(8 real jumps, scored against jump_detector's own firing frames):

    rise  confirm   found  false  frames earlier
    0.08     2       8/8     2        1.62
    0.10     2       8/8     0        1.25   <- chosen
    0.12     2       8/8     0        1.00
    0.15     2       8/8     0        0.12
    0.10     3       8/8     0        0.25

So RISE=0.10 with CONFIRM=2: still 4x the noise floor, no false positives on
either clip, and ~1.25 frames (~45 ms here) sooner off the ground. Dropping
to 0.08 buys a third of a frame and costs two false positives -- not a trade
worth making in a game where a phantom jump wastes your one clearance.

Two things the extra sensitivity costs, both handled below:

* Feet leave the frame. Ankles are the first keypoints to be occluded or to
  drop off the bottom edge when the player stands close, so there is a hip
  fallback.
* A single bad box wrecks the baseline. One spurious detection (a box 133px
  tall against a median of 860) turned a 0.26 peak into 3.33 in the original
  trace. A relative box-size guard rejects those.
"""

import statistics
from collections import deque
from dataclasses import dataclass

import duck as duck_mod
from coco import (LEFT_ANKLE, LEFT_HIP, LEFT_SHOULDER, RIGHT_ANKLE, RIGHT_HIP,
                  RIGHT_SHOULDER)

CONF_MIN = 0.50  # person confidence floor, as in jump_detector
KP_MIN = 0.30  # per-keypoint floor; ankles are legitimately less certain
BASE_WIN = 45  # rolling baseline; 45 frames, which is ~2.4 s at the ~18 fps
# the device actually delivers, not the ~1.5 s this comment used to claim.
RISE = 0.10  # 4x the ankle noise floor, well under the 0.54 peak
FALL = 0.04
CONFIRM = 2
REFRAC = 10

# A detection whose person is suddenly a very different size from the recent
# median is a different person, or a spurious box -- not a jump.
BOX_LO, BOX_HI = 0.6, 1.6

# Player lock. Identity between frames is matched on horizontal position and
# box width only -- never on vertical position, and never on whole-box
# distance. A jump moves the player's box a long way up and barely at all
# sideways, so y is the one axis that cannot be trusted for identity. It is
# the same trap select() avoids by refusing nearest-to-previous: the
# stationary mirror reflection in both test videos wins on proximity at
# exactly the moment the player leaves the ground.
LOCK_DX = 0.75  # box widths of sideways drift allowed between frames
LOCK_W = (0.6, 1.6)  # and how much the width may change, as BOX_LO/BOX_HI do
LOCK_EMA = 0.3  # how fast the lock follows the player walking about


@dataclass
class Jump:
    frame: int
    time_s: float
    height: float  # lift at takeoff, in body heights


class TakeoffDetector:
    """Stateful, one per player. Feed it a `Pose` per frame."""

    def __init__(
        self,
        fps=30.0,
        *,
        signal="ankle",
        duck_signal="shoulder",
        conf_min=CONF_MIN,
        base_win=BASE_WIN,
        rise=RISE,
        fall=FALL,
        confirm=CONFIRM,
        refrac=REFRAC,
        duck_drop=duck_mod.DROP,
        duck_rise_back=duck_mod.RISE_BACK,
        duck_confirm=duck_mod.CONFIRM,
    ):
        self.fps = fps
        self.signal = signal
        self.duck_signal = duck_signal
        self.conf_min = conf_min
        self.rise = rise
        self.fall = fall
        self.confirm = confirm
        self.refrac = refrac

        self._ys = deque(maxlen=base_win)
        self._boxes = deque(maxlen=base_win)
        self._state = "IDLE"
        self._run = 0
        self._cooldown = 0
        self.frame_index = -1
        self.lift = 0.0  # exposed for the HUD, and for tuning by eye
        self.selected = None  # index of the person we are tracking
        self.using = signal
        # The locked player's signature, (centre x, box width), or None for
        # "anyone may play". Held across frames; see lock().
        self._lock = None
        self._sig = None  # this frame's signature for `selected`
        self._key = None  # whose baseline `_ys` currently describes

        # The duck runs a second state machine on a second signal, sharing
        # this object's person selection, lock and box bookkeeping. Two
        # detectors would duplicate all of it -- and, worse, could disagree
        # about who the player is, because the lock EMA below is mutated every
        # frame, so two instances fed the same poses genuinely would diverge.
        # A duck detector with its own idea of the player would crouch the
        # avatar when a bystander bent down.
        self._dys = deque(maxlen=base_win)  # duck baseline, shoulder midpoint
        self._duck = None if duck_signal == "off" else duck_mod.DuckState(
            drop=duck_drop, rise_back=duck_rise_back, confirm=duck_confirm)
        self.descent = 0.0  # exposed for the HUD and for tuning by eye
        self.last_duck = None

    def select(self, pose):
        """Highest-confidence person above the floor, or None.

        Deliberately not nearest-to-previous: both test videos contain a
        stationary mirror reflection that beats the real player on proximity
        exactly when they jump. See the README.
        """
        best, best_conf = None, 0.0
        for i in range(min(len(pose.xy), len(pose.box_xyxy))):
            c = float(pose.box_conf[i])
            if c >= self.conf_min and c > best_conf:
                best, best_conf = i, c
        return best

    @property
    def ducking(self):
        """True while the player is holding a crouch. Mirrors AIRBORNE."""
        return self._duck is not None and self._duck.ducking

    @property
    def locked(self):
        """True while one player owns the game."""
        return self._lock is not None

    def lock(self):
        """Give the game to whoever is selected right now.

        Called when a run starts, so the person whose jump started it keeps
        the game even if someone else walks in front of the camera with a
        better detection score.
        """
        if self._sig is not None:
            self._lock = self._sig
            # Carry the baseline across: this is the same body we were
            # already tracking, so wiping ~1.5 s of ankle history here would
            # leave the player unable to jump just as their run begins.
            self._key = "lock"

    def release(self):
        """Hand the game back to the room."""
        self._lock = None
        self._key = self.selected  # same body; keep its baseline

    @staticmethod
    def _signature(pose, i):
        """(centre x, width) of person `i`'s box, or None if degenerate."""
        x1, x2 = float(pose.box_xyxy[i][0]), float(pose.box_xyxy[i][2])
        w = x2 - x1
        return None if w <= 0 else ((x1 + x2) / 2, w)

    def _match_lock(self, pose):
        """The locked player among this frame's detections, or None.

        None means "not visible this frame", which is deliberately different
        from "gone": update() holds its state rather than measuring someone
        else, and the caller sees selected is None and can pause the world.
        """
        cx0, w0 = self._lock
        best, best_d = None, None
        for i in range(min(len(pose.xy), len(pose.box_xyxy))):
            if float(pose.box_conf[i]) < self.conf_min:
                continue
            sig = self._signature(pose, i)
            if sig is None:
                continue
            cx, w = sig
            if not LOCK_W[0] < w / w0 < LOCK_W[1]:
                continue
            d = abs(cx - cx0) / w0
            if d <= LOCK_DX and (best_d is None or d < best_d):
                best, best_d = i, d
        return best

    @staticmethod
    def _pair(pose, i, a, b):
        """Midpoint y of keypoints `a` and `b` for person `i`, or None."""
        kp = pose.kp_conf[i] if pose.kp_conf is not None else None
        pts = pose.xy[i]
        if kp is not None and (float(kp[a]) < KP_MIN or float(kp[b]) < KP_MIN):
            return None
        ya, yb = float(pts[a][1]), float(pts[b][1])
        if ya <= 0 or yb <= 0:
            return None
        return (ya + yb) / 2

    @staticmethod
    def _box_h(pose, i):
        y1, y2 = float(pose.box_xyxy[i][1]), float(pose.box_xyxy[i][3])
        return y2 - y1

    def _measure(self, pose, i):
        """(y, box_h, which_signal) for person `i`, or None if unusable."""
        box_h = self._box_h(pose, i)
        if box_h <= 0:
            return None

        def pair(a, b):
            return self._pair(pose, i, a, b)

        if self.signal == "ankle":
            y = pair(LEFT_ANKLE, RIGHT_ANKLE)
            if y is not None:
                return y, box_h, "ankle"
            # Feet occluded or out of frame -- fall back rather than go blind.
            y = pair(LEFT_HIP, RIGHT_HIP)
            return None if y is None else (y, box_h, "hip")

        y = pair(LEFT_HIP, RIGHT_HIP)
        return None if y is None else (y, box_h, "hip")

    def _measure_duck(self, pose, i):
        """(y, box_h) for the duck signal, or None if unusable.

        No fallback chain, unlike _measure. Shoulders are never below the
        confidence floor in a session's worth of real play, and the fallbacks
        available are the ones duck.py's docstring rules out.
        """
        box_h = self._box_h(pose, i)
        if box_h <= 0:
            return None
        if self.duck_signal == "boxtop":
            # Experimental, for the tuning stage only -- see duck.py.
            return float(pose.box_xyxy[i][1]), box_h
        y = self._pair(pose, i, LEFT_SHOULDER, RIGHT_SHOULDER)
        return None if y is None else (y, box_h)

    def update(self, pose):
        """Advance one frame. Returns a `Jump` on takeoff, else None."""
        i = self._match_lock(pose) if self._lock is not None else self.select(pose)
        self.selected = i
        self._sig = None if i is None else self._signature(pose, i)

        # The baseline is one person's history of ankle heights. Carrying it
        # across a change of person is what used to invent jumps out of
        # nothing: the newcomer's ankles get compared against the last
        # person's median, and a difference in where they happen to be
        # standing reads as a leap. Locked, the key never changes and the
        # baseline survives; unlocked, any change of person resets it.
        # Only a *different person* resets the baseline. A frame where
        # nobody was matched is not a change of person -- it is the same
        # "dropped frames are not evidence" rule update_lift() follows, and
        # clearing on them starves the baseline until no jump ever registers.
        if i is not None:
            key = "lock" if self._lock is not None else i
            if key != self._key:
                self._key = key
                self._ys.clear()
                self._boxes.clear()
                # Same rule, same reason: a newcomer measured against the last
                # person's shoulder baseline reads as a crouch, and a crouch
                # latches where a jump would only misfire once.
                self._dys.clear()
                if self._duck is not None:
                    self._duck.reset()

        if i is not None and self._lock is not None and self._sig is not None:
            cx0, w0 = self._lock
            cx, w = self._sig
            self._lock = (cx0 + LOCK_EMA * (cx - cx0), w0 + LOCK_EMA * (w - w0))

        m = None if i is None else self._measure(pose, i)
        d = None if i is None else self._measure_duck(pose, i)

        # One guard, two verdicts. The strict floor is the takeoff path's and
        # is exactly what it always was; the duck path gets a lower one,
        # because a crouch shrinks the box *on purpose* and 0.6 would throw a
        # real duck away as a spurious detection. See duck.BOX_LO.
        base = statistics.median(self._boxes) if self._boxes else None

        def box_ok(box_h, lo):
            return base is None or lo < box_h / base < BOX_HI

        # Last frame's verdict. While DOWN nothing updates any baseline, so a
        # held crouch cannot drag the median down and expire itself: at 18 fps
        # a 45-frame window is dominated by crouch samples after ~1.2 s, and
        # the descent would decay to zero with the player still crouched.
        # Freezing beats clearing because the deques still hold pre-crouch
        # samples when the player stands up, so the descent reads ~0 at once
        # rather than after a re-learning gap.
        frozen = self.ducking

        if m is not None and box_ok(m[1], BOX_LO):
            y, box_h, which = m
            # Switching signal mid-flight would compare against a baseline
            # built from a different keypoint, so start that baseline over.
            if which != self.using:
                self.using = which
                self._ys.clear()
            if not frozen:
                self._boxes.append(box_h)
                self._ys.append(y)
            lift = (None if not self._ys
                    else (statistics.median(self._ys) - y) / box_h)
            jump = self.update_lift(lift)
        else:
            jump = self.update_lift(None)

        self._update_duck(d, box_ok, frozen)

        # You cannot be airborne and crouched, and the countermovement dip
        # before a jump is the largest non-duck excursion there is -- so if
        # one ever does confirm a duck, the takeoff that follows cancels it.
        if jump is not None and self._duck is not None:
            self._duck.reset()
        return jump

    def _update_duck(self, d, box_ok, frozen):
        """Advance the duck machine on frames the takeoff path may have dropped.

        A deep crouch fails the strict box guard, and the feet can be out of
        shot entirely, so this deliberately does not piggyback on the takeoff
        path's decisions -- only on its person selection.
        """
        if self._duck is None:
            return
        if d is None or not box_ok(d[1], duck_mod.BOX_LO):
            self._duck.update(None)
            return

        y, _box_h = d
        if not frozen:
            self._dys.append(y)
        # Normalised by the rolling *median* box height, not this frame's.
        # A crouch pulls the box top down with the head while the feet stay
        # put, so the live box falls with the signal; dividing by it inflates
        # entry and poisons exit, because as the player rises the denominator
        # grows back and the ratio shrinks faster than the body actually does.
        # Hysteresis cannot fix a denominator racing the numerator.
        #
        # The takeoff path above deliberately keeps the instantaneous divisor:
        # RISE and Player.LIFT_MIN/LIFT_MAX are calibrated in those units, and
        # a jump translates the box rather than shrinking it.
        scale = statistics.median(self._boxes) if self._boxes else None
        if not scale or len(self._dys) < duck_mod.WARM:
            self._duck.update(None)
            return
        self.descent = (y - statistics.median(self._dys)) / scale
        event = self._duck.update(self.descent)
        if event == "enter":
            self.last_duck = duck_mod.Duck(
                self.frame_index, self.frame_index / self.fps, self.descent)
        elif event == "expire":
            # The baseline is frozen and still describes standing, so the
            # descent is *still* over DROP at the moment of expiry. Rebuild it
            # from the current posture or the machine re-enters DOWN two
            # frames later and oscillates at the MAX_DOWN period forever.
            self._dys.clear()

    def update_lift(self, lift):
        """The state machine alone. No model, no frame -- directly testable."""
        self.frame_index += 1

        if lift is None:
            # A dropped frame is not evidence either way; hold state.
            if self._cooldown > 0:
                self._cooldown -= 1
            return None

        self.lift = lift
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        if self._state == "IDLE":
            self._run = self._run + 1 if lift > self.rise else 0
            if self._run >= self.confirm:
                self._state = "AIRBORNE"
                self._run = 0
                return Jump(self.frame_index, self.frame_index / self.fps, lift)
        elif self._state == "AIRBORNE":
            if lift < self.fall:
                self._state = "IDLE"
                self._cooldown = self.refrac
        return None
