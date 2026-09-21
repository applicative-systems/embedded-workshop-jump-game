"""Duck detection for the game, on the shoulder signal.

The mirror of takeoff.py: same shape of state machine, opposite direction, a
different signal and a different normaliser. Everything below was measured by
replaying tests/fixtures/draft2.jsonl -- 1425 frames of real play, 26 jumps,
one player, yolo11n-pose at 18.35 fps -- with

    descent = (y - rolling_median(y)) / rolling_median(box_h)

positive when the body drops. "Quiet" excludes 15 frames before and 35 after
any takeoff; "jump worst" is the largest descent anywhere in a takeoff window.

    signal              quiet sd   quiet p95   quiet max   jump worst
    shoulder mid          0.029       0.043       0.047      0.090
    hip mid               0.018       0.055       0.068      0.068
    nose                  0.045       0.096       0.115      0.087
    box top (y1)          0.014       0.019       0.024      0.084
    shoulder over feet    0.038       0.044       0.048      0.111

Shoulders win, and not on the noise column -- note that quiet noise is *not*
the binding constraint for any candidate, since every one of them has a
jump-window artifact several times its own quiet p95. Choosing on quiet sd is
what makes box-top look best, and it optimises the wrong number.

* **Ankles**, the takeoff signal, are excluded by physics: in a duck the feet
  are planted, so the true signal is zero.
* **Nose** is the noisiest of all, and its confidence collapses in exactly the
  pose we are detecting -- it falls below KP_MIN on 7.2% of frames, against
  0.0% for either shoulder, because a player who ducks tucks their chin.
* **Box top** is not a body landmark at all. It is the top of a box drawn
  round every keypoint including the wrists, so raising the arms lifts it; and
  it is pinned to the frame edge on 18.2% of frames, which is what its
  flattering quiet number is really measuring.
* **Shoulder height above the box bottom** is the intuitive choice, being
  scale-free by construction, and it is the worst of the lot (0.111): at
  landing the feet plant while the torso keeps compressing, so a landing *is*
  a crouch by that measure.
* **Hips** are the genuine runner-up, and would be a fine second signal. They
  lose on dynamic range at the other end: in a waist-bend duck -- legs
  straight, hinge at the hips, the fastest duck a tired player performs -- the
  hips do not move at all.

**The hazard is the countermovement, not the landing.** You dip before you
leap, on 26 of 26 takeoffs, for about 5 frames, peaking 6 frames (~0.33 s)
before the jump is detected:

    frames before takeoff    -8     -6     -4     -1     +1     +7
    median descent          .042   .068   .016  -.127  -.148   .041
    max descent             .072   .090   .053  -.103  -.129   .064

The two largest descents in the whole session (0.090, frames 435 and 475) are
both countermovements. Landing compression is smaller (max 0.066). This rules
out the obvious defence -- gating the duck on "not airborne" does nothing,
because the dip happens *before* takeoff. What works is that a confirmed Jump
resets this machine: physically true, and it caps any countermovement-induced
false duck at the handful of frames before the takeoff that follows it.

**One session was not enough, and this is the part worth reading.** Tuned on
draft2 alone the threshold came out at 0.12, and it was wrong. Replayed
against the other two recordings in the repo -- the same clips
tests/fixtures/*.csv are cut from, both of them jumping only -- the peak
non-duck descent is:

    session                    frames   jumps   peak descent
    draft2                      1425      26       0.090
    jump-framework-builtin       301       3       0.103
    jump-logitech                314       4       0.141   <-- 1.6x draft2

A different body in a different room pre-loads a jump half again as deeply,
and 0.12 fired a phantom duck on it (frames 27-32, six frames of it). Nothing
about that was a baseline artifact: the shoulders really do travel 117 px down
over ten frames there, and then straight back up into the jump.

False ducks across all three sessions, 2040 frames and 33 jumps with not one
real duck in them, so every hit is a false positive (RISE_BACK = 0.06):

    DROP      0.10   0.11   0.12   0.13   0.14   0.15   0.16
    CONFIRM=1    4      2      1      1      1      0      0
    CONFIRM=2    3      2      1      1      0      0      0
    CONFIRM=3    2      1      1      1      0      0      0
    CONFIRM=4    1      1      1      0      0      0      0

DROP = 0.16 with CONFIRM = 3 ships, which is clean with headroom on *both*
axes rather than sitting on the edge of either: no frame in any session
reaches 0.16 at all (13% above the worst), and the offending dip holds 0.12
for only four frames where a real duck holds its depth for 10-40. Duration is
the more trustworthy of the two, because it is bounded by physics -- a
countermovement is a ballistic pre-load with a jump on the end of it, so it
cannot be held, while a crouch is nothing but held. Amplitude only tells you
how hard this particular player loads their legs.

The cost is 3 frames, ~0.16 s at the ~18 fps the device delivers, against
0.81 s of sighting time at the game's top speed. Latency is not the binding
constraint on this mechanic anywhere; the minimum hold is.

Against the other end: standing shoulders sit 0.808 box heights above the box
bottom (p5 0.797), which matches the anthropometric acromion height and
confirms the box is stature when standing. So a deep knee bend is a descent of
~0.31, a full waist bend ~0.29, and a half squat 0.12-0.16. DROP = 0.16
therefore asks for a real crouch -- shoulders down to 0.65 of stature -- and
not a nod. Whether that is too much to ask of a tired player is the one thing
these three sessions cannot answer, because none of them contains a duck. It
is the first question for the recording described in the plan, and DROP is a
constructor kwarg so tools/sweep_duck.py can answer it without a code change.

Constants are in frames, never seconds. The detector is handed `fps` from the
requested camera rate rather than the measured loop rate, so seconds here
would be a lie; the game applies the minimum hold, and it has real dt.
"""

from dataclasses import dataclass

DROP = 0.16  # descent, in standing body heights, to start a duck
RISE_BACK = 0.08  # and to end one; half of DROP, as FALL is to RISE
CONFIRM = 3  # consecutive frames above DROP to enter DOWN
CLEAR = 2  # consecutive frames below RISE_BACK to leave it
# Safety valve for a DOWN state that latches and will not leave -- a frozen
# baseline gone stale, say, because the player walked toward the camera while
# crouched.
#
# 600, not the 90 (~5 s) it started at. 90 was sized against "longer than any
# gameplay duck", and that stopped being true the moment the obstacle mix went
# to an even split: ceiling bars are now independent coin flips, so they come
# in streaks, and a streak is held with one continuous crouch. Simulated over
# 20 seeds the longest run is 19 bars spanning 23.5 s. A valve shorter than
# that stands the player up into a bar, which is a far commoner failure than
# the one it was guarding against.
#
# It can afford to be generous because it is not the only way out: a confirmed
# Jump resets this machine, and jumping is exactly what a player does to
# restart after dying. That is the real escape hatch; this is the backstop.
# 600 frames is ~33 s at the ~18 fps the device delivers and ~20 s if a
# TensorRT engine takes it to 30.
MAX_DOWN = 600
# Baseline samples before a duck may fire. The takeoff path has no warm-up and
# does not need one: it detects a one-frame event, while this one *latches*, so
# it is worth ~0.8 s of silence after a change of person rather than risking a
# stuck crouch built on a median of three samples.
WARM = 15
# The duck's box-height floor, against takeoff.py's 0.6. A crouch shrinks the
# bounding box on purpose -- the head comes down while the feet stay put -- so
# 0.6 would throw real ducks away as spurious detections. Still 4.5x above the
# box this guard exists to reject (87.7 px against a median of 869, ratio 0.10,
# in tests/fixtures/jump-logitech-yolo11s-pose.csv at frame 192).
BOX_LO = 0.45


@dataclass
class Duck:
    """A confirmed duck. Unlike a Jump this is the *start* of a held state."""

    frame: int
    time_s: float
    depth: float  # descent at the frame it was confirmed


class DuckState:
    """The duck state machine alone. No model, no frame -- directly testable.

    IDLE <-> DOWN, mirroring TakeoffDetector's IDLE <-> AIRBORNE. Unlike a
    jump, a duck is held: the caller reads `ducking` every frame and applies
    its own minimum hold.

    There is no refractory period. Takeoff needs one because the landing
    bounce can re-trigger it; the duck's equivalent -- the rebound as the
    player stands up -- is a *rise*, and a rise cannot start a duck.
    """

    def __init__(self, *, drop=DROP, rise_back=RISE_BACK, confirm=CONFIRM,
                 clear=CLEAR, max_down=MAX_DOWN):
        self.drop = drop
        self.rise_back = rise_back
        self.confirm = confirm
        self.clear = clear
        self.max_down = max_down
        self._state = "IDLE"
        self._run = 0
        self._held = 0

    @property
    def ducking(self):
        """True while the player is holding a crouch."""
        return self._state == "DOWN"

    def reset(self):
        """Back to IDLE, counters cleared. Called when a takeoff is confirmed."""
        self._state = "IDLE"
        self._run = 0
        self._held = 0

    def update(self, descent):
        """Advance one frame. Returns "enter", "exit", "expire", or None.

        `descent` of None means "no measurement this frame", which holds the
        state rather than ending a duck -- the same rule the takeoff machine
        applies to a dropped frame.

        "expire" is distinguished from "exit" only so the caller knows to
        clear the baseline. It must: with a frozen standing baseline the
        descent is still above DROP at the moment of expiry, so a bare
        force-exit would re-enter DOWN two frames later and oscillate forever.
        """
        if descent is None:
            return None

        if self._state == "IDLE":
            self._run = self._run + 1 if descent > self.drop else 0
            if self._run >= self.confirm:
                self._state = "DOWN"
                self._run = 0
                self._held = 0
                return "enter"
            return None

        self._held += 1
        if self._held >= self.max_down:
            self.reset()
            return "expire"
        self._run = self._run + 1 if descent < self.rise_back else 0
        if self._run >= self.clear:
            self.reset()
            return "exit"
        return None
