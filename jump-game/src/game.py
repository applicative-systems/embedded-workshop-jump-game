"""The game itself: obstacles scroll at the player, who jumps to clear them.

Kept apart from jump_game.py, and free of any model or camera code, for the
same reason jump_detector.py's state machine is: all of it can be stepped by
hand in a test with no GPU and no webcam. `update()` takes a time delta and
two booleans and is completely deterministic apart from obstacle spawning,
which takes an injectable `rng`.

Every length here is a **fraction of the frame** -- x of the width, y of the
height, with y counted from the top like OpenCV does -- and every speed is
fractions per second, integrated against real elapsed time. So the game plays
identically at any resolution and any frame rate, which matters because the
frame rate is whatever the pose model leaves us.
"""

import random
import statistics
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

import coco
import gameover
import sprite

# Colours (BGR).
# The avatar wears the palette of the logo it carries for a head: the two
# NixOS blues, light on the limbs and dark down the spine, interleaved the
# way the snowflake's own lambdas are. Death stays red -- it has to read as
# an alarm against the blue, and a third blue would not.
C_LIMB = (242, 184, 95)  # #5fb8f2
C_CORE = (183, 111, 77)  # #4d6fb7
C_OBSTACLE = (40, 200, 255)
# The string a hanging obstacle hangs from. White, like the ground line and
# like both instruction words: against a lit room the game's own furniture has
# to be the brightest thing on screen, and the two verbs are already told
# apart by where the obstacle is and by the word itself, so spending a colour
# on that as well would only compete with the artwork.
C_STRING = (255, 255, 255)
C_GROUND = (210, 210, 210)
C_TEXT = (255, 255, 255)
C_DEAD = (60, 60, 255)


@dataclass
class Obstacle:
    """A block travelling right to left, occupying [y, y + h] above the ground.

    `y` is the bottom edge. Ground obstacles leave it at 0 and are cleared by
    jumping over them; a ceiling bar sits it at head height and runs off the
    top of the frame, so the only way past is to duck under it. One interval
    rather than two shapes, because it makes the collision test symmetric --
    see JumpGame._hits().

    `art` is the Sprite drawn in place of the block, or None for a plain
    rectangle. It is chosen once at spawn rather than per frame so an
    obstacle keeps its identity all the way across the screen.
    """

    x: float  # left edge
    w: float
    h: float  # height of the collision box
    y: float = 0.0  # bottom edge, above the ground line
    # How tall the artwork itself is drawn, from `y` upward. Equal to `h` for
    # a ground obstacle, where the picture *is* the obstacle. A ceiling bar is
    # a picture of the same size with a wall stacked on top of it, so its box
    # runs off the top of the frame while its art does not -- which is the
    # whole point: the two kinds look alike and only their height differs.
    art_h: float = 0.0
    passed: bool = False
    art: object = None

    @property
    def ceiling(self):
        return self.y > 0.0


class Player:
    """The red line. Rests on the ground; a jump throws it on a ballistic arc.

    Ballistic rather than a live mirror of the player's own height, because
    the game needs a *predictable* hang time: an obstacle is only fair if
    clearing it depends on jumping at the right moment, not on the player
    physically staying airborne for as long as the obstacle takes to pass.
    """

    # Peak height and hang time of a reference jump, which together fix
    # gravity: peak = v0^2/2g and airtime = 2*v0/g.
    #
    # The hang time is deliberately long. What has to hold for the game to be
    # fair is that the time the arc spends above an obstacle exceeds the time
    # that obstacle spends overlapping the player -- and the second term is
    # (player_width + obstacle_width) / speed, which gets *bigger* as the game
    # gets slower. So the binding case is the opening speed, not the top one,
    # and the arc has to be long enough to sit out a slow obstacle.
    # Hang time is held close to a real jump (~0.45-0.55 s) rather than
    # stretched for extra forgiveness. An arc that floats much longer than
    # the player does makes the avatar land visibly after they do, and that
    # mismatch is more disorienting than a tight timing window -- the whole
    # point of mirroring their pose is that avatar and body read as the same
    # movement. These sit a touch above that range, deliberately: the arc
    # needs enough height and air to feel like a jump rather than a hop, and
    # ~70 ms of overshoot at the landing is a fair price for that.
    #
    # Scaling the stage up scales REF_PEAK but *not* REF_AIR, which is the
    # whole trick: hang time is a property of the player's body, not of the
    # picture, so it is the one quantity that must not grow with the drawing.
    # Holding it while the peak grows is exactly what raises gravity, and a
    # uniform spatial scale with gravity scaled to match leaves every timing
    # in the game identical -- the same play, drawn bigger.
    REF_PEAK = 0.39  # 1.5x the stage scale-up
    REF_AIR = 0.55
    GRAVITY = 8 * REF_PEAK / (REF_AIR**2)

    # Measured lift (body heights) -> arc peak (frame heights). The floor is
    # generous because a takeoff is detected before the real apex, so the
    # lift handed to us always understates the jump -- and because even the
    # feeblest hop has to clear the tallest obstacle with room to spare.
    # LIFT_MIN is the takeoff threshold itself (takeoff.RISE): a smaller lift
    # cannot trigger a jump at all, so mapping from below it only wasted the
    # bottom of the arc range. Real jumps measure 0.24-0.32 here.
    LIFT_MIN, LIFT_MAX = 0.10, 0.32
    # PEAK_MIN is set by fairness rather than by looks: obstacles grew by 2x
    # against the stage's 1.5x, so they are relatively taller than they were
    # and the feeblest jump has to clear the tallest one with the timing
    # window intact. 0.34 holds that window at 0.198 s against the old
    # 0.211 s. It is ~1.2 of the avatar's own height, so it still reads as a
    # jump rather than as being fired out of a cannon.
    #
    # PEAK_MAX is capped by the frame, not by feel: measured against a
    # standing pose it puts the top of the avatar's head 7% of the way down
    # the picture, and every extra 0.01 of peak costs another ~1%. There is
    # no fairness cost to keeping it modest -- clearance is decided by
    # PEAK_MIN, the *weakest* jump -- so the headroom is worth more than the
    # extra height.
    #
    # PEAK_MIN is set by the tallest obstacle, and by a wider measure than
    # just clearing it. At 0.42 the arc did out-top the 269 px beaver -- it
    # peaked at 403 px -- and the game was still unplayable, because what
    # decides whether a jump lands is not the height reached but the time
    # spent above the obstacle: 0.105 s, six frames, less than the takeoff
    # detector's own latency. 0.50 takes that to 0.179 s (eleven frames).
    #
    # PEAK_MAX is what the frame allows, given where the avatar's head ends
    # up. It is paired with the default ground line rather than fixed on its
    # own -- JumpGame clamps it against whatever --ground is actually set to,
    # so a lower line cannot silently throw the avatar off the top.
    PEAK_MIN, PEAK_MAX = 0.50, 0.575

    # Fixed horizontal position, as a fraction of stage width. This is about
    # as far left as it can go: X is the *box's* left edge, but the figure is
    # drawn centred in that box and its arms reach roughly half a body height
    # further out again. Measured with the arms fully outstretched, 0.06
    # leaves 32 px of margin and 0.04 clips the hand off the screen.
    X = 0.06
    # Widths are in fractions of frame *height*, not width, and are divided
    # by the stage aspect where they are used. The stage's shape is a runtime
    # choice now (--stage), and a width quoted as a fraction of it would make
    # the avatar fatten or thin with the monitor it happens to be on.
    W_H = 0.14  # drawn width, in frame heights
    # Collision uses a narrower box than the line we draw, centred on it.
    # Standard platformer forgiveness: clipping the very end of the bar
    # against the very edge of a block should not end the run.
    HIT_W_H = 0.056

    def __init__(self):
        self.lift = 0.0
        self.vel = 0.0
        self.airborne = False

    def launch(self, lift, peak_max=None):
        """Start a jump. `peak_max` caps the arc; None means PEAK_MAX.

        The cap is passed in rather than read off the class because it
        depends on where the ground line is, which is a runtime choice.
        """
        if self.airborne:
            return  # already in the air; no double jumps
        top = self.PEAK_MAX if peak_max is None else max(self.PEAK_MIN, peak_max)
        span = self.LIFT_MAX - self.LIFT_MIN
        t = min(1.0, max(0.0, (lift - self.LIFT_MIN) / span))
        peak = self.PEAK_MIN + t * (top - self.PEAK_MIN)
        self.vel = (2 * self.GRAVITY * peak) ** 0.5
        self.airborne = True

    def step(self, dt):
        if not self.airborne:
            return
        self.vel -= self.GRAVITY * dt
        self.lift += self.vel * dt
        if self.lift <= 0:
            self.lift, self.vel, self.airborne = 0.0, 0.0, False

    def reset(self):
        self.lift, self.vel, self.airborne = 0.0, 0.0, False


class StickFigure:
    """The avatar's body: a small figure that copies the player's live pose.

    Only the *pose* is copied. Where the figure is on screen stays under the
    game's control (Player's ballistic arc), because the two answer different
    questions: the arc is what keeps obstacle timing fair and predictable,
    while the limbs are what let the player see, continuously and in every
    idle movement they make, how far behind them the picture actually is.
    That is the whole point of a figure over a bar -- lag you can watch in
    your own arms while waiting is lag you can learn to lead.

    Poses are anchored at the **hips**, not the feet. Hips are the most
    reliably detected pair (0.98 mean confidence against the ankles' 0.77)
    and, more importantly, anchoring at the feet would cancel out exactly the
    thing worth showing: when the player tucks their legs mid-jump, a
    hip-anchored figure tucks with them, while a foot-anchored one would
    squash its torso instead.
    """

    HEIGHT = 0.2775  # of frame height
    # How far the drawn figure reaches above its own feet, in figure heights
    # -- hips, torso, neck and head stacked up. It is now a *target* rather
    # than an observation: draw() squashes the pose vertically so the top of
    # the head lands exactly here, which is what lets JumpGame use one number
    # for both the drawing and the collision (see player_h).
    #
    # 1.05, not the 1.20 this used to claim. 1.20 was never what the figure
    # drew: replayed over tests/fixtures/draft2.jsonl the real extent is 1.022
    # median, 1.036 p75, 1.089 p95 -- so the old constant over-stated the
    # avatar by ~17%. Harmless while it only fed the headroom clamp (which is
    # capped by PEAK_MAX anyway at the default ground line), but fatal the
    # moment a ceiling bar is placed against it: the player would have died
    # with 47 px of daylight visible above their head. 1.05 sits just above
    # the observed p75, so the squash is a few percent either way rather than
    # a distortion.
    TOP_EXTENT = 1.05
    # The same measure, crouched. Because the squash is unconditional, this is
    # exactly what the crouched avatar draws -- it is a gameplay choice, not a
    # measurement of anybody's body, and the physical depth a real crouch
    # needs lives in the duck detector instead. 0.58/1.05 = 0.55: a low, clear
    # crouch, chosen with BAR_GAP so both margins in JumpGame stay ~40 px+.
    CROUCH_EXTENT = 0.58
    # Fallback standing hip height above the feet, in figure heights, used
    # only until an ankle has been seen. The live value comes off the pose --
    # see draw() -- because a hardcoded one cancels every crouch.
    HIP_FRAC = 0.48
    SMOOTH = 0.6  # EMA on normalised keypoints; high = responsive
    KP_MIN = 0.30
    # The pose is normalised by a rolling *standing* box height, not by this
    # frame's. A crouch shrinks the bounding box by roughly the factor it
    # shrinks the pose, so dividing by the live box cancels the crouch almost
    # exactly -- which is why the figure never tucked its legs mid-jump
    # either, despite the docstring above having claimed it did for as long
    # as it has existed.
    #
    # A quantile rather than a median, and a long window rather than
    # takeoff.py's 45. A jump is brief, so a median survives it; a crouch is
    # *held*, and a median over a short window would drag down to the
    # crouched height and cancel the crouch all over again. At a bar every
    # ~4.5 s the crouch duty cycle is ~20%, so the 75th percentile is
    # robustly the standing box.
    SCALE_WIN = 240  # ~4 s at 60 fps, ~13 s at 18 fps
    SCALE_Q = 0.75
    SCALE_EVERY = 6  # recompute every N accepted frames; hold in between
    SCALE_EMA = 0.25  # so the figure's unit of length does not shimmer
    # Same outlier guard as takeoff.py's: one spurious 133px box against an
    # 860px median must not become the scale the avatar is drawn at.
    BOX_LO, BOX_HI = 0.6, 1.6
    HEAD_R = 0.070  # fallback circle radius, of figure height
    # The logo head is deliberately far bigger than the circle it replaces,
    # and bigger than anatomy would have it. Drawn at the circle's size the
    # NixOS snowflake came out ~25 px across at 1280x960, which reads as a
    # blue smudge -- its arms only separate at about twice that. The cost is
    # a figure ~15% taller overall, which is cheap for a head you can name.
    HEAD_H = 0.30  # logo height, of figure height
    NECK = 0.02  # gap from the neck joint to the bottom of the head

    def __init__(self):
        self.pts = None  # (17, 2) normalised to hip origin, in standing heights
        self.ok = [False] * coco.N
        # The rolling standing box height, and the samples it is drawn from.
        # Deliberately *not* shared with TakeoffDetector's deque: that one is
        # cleared on a change of person and on a change of signal, and a clear
        # mid-crouch would make the avatar visibly change size.
        self._boxes = deque(maxlen=self.SCALE_WIN)
        self.scale = None
        self._since = 0
        self._foot = None  # last known hip height above the planted foot

    def _rescale(self):
        """Refresh `scale` from the window, EMA'd so it does not shimmer."""
        vals = sorted(self._boxes)
        i = min(len(vals) - 1, int(self.SCALE_Q * (len(vals) - 1) + 0.5))
        target = vals[i]
        if self.scale is None:
            self.scale = target
        else:
            self.scale += self.SCALE_EMA * (target - self.scale)

    def update(self, xy, kp_conf, box_xyxy):
        """Absorb one detection. Pass xy=None when nobody was found."""
        if xy is None:
            return  # hold the last pose rather than collapsing to nothing
        box_h = float(box_xyxy[3]) - float(box_xyxy[1])
        if box_h <= 0:
            return

        # Guard before insertion, so a spurious box never enters the scale.
        if self._boxes:
            ratio = box_h / statistics.median(self._boxes)
            if not self.BOX_LO < ratio < self.BOX_HI:
                return
        self._boxes.append(box_h)
        if self.scale is None or self._since >= self.SCALE_EVERY:
            self._rescale()
            self._since = 0
        else:
            self._since += 1
        size = self.scale

        hx = (float(xy[coco.LEFT_HIP][0]) + float(xy[coco.RIGHT_HIP][0])) / 2
        hy = (float(xy[coco.LEFT_HIP][1]) + float(xy[coco.RIGHT_HIP][1])) / 2
        fresh = [
            ((float(xy[j][0]) - hx) / size, (float(xy[j][1]) - hy) / size)
            for j in range(coco.N)
        ]
        good = [
            (kp_conf is None or float(kp_conf[j]) >= self.KP_MIN)
            and (float(xy[j][0]) > 0 or float(xy[j][1]) > 0)
            for j in range(coco.N)
        ]

        if self.pts is None:
            self.pts = list(fresh)
            self.ok = good
            return
        a = self.SMOOTH
        for j in range(coco.N):
            if not good[j]:
                continue  # a keypoint that dropped out keeps its last place
            px, py = self.pts[j]
            self.pts[j] = (px + a * (fresh[j][0] - px), py + a * (fresh[j][1] - py))
            self.ok[j] = True

    def draw(self, frame, cx, feet_y, c_limb, c_core, head=None, head_tint=None,
             extent=None):
        """Draw with the figure's feet at (cx, feet_y), in pixels.

        Arms and legs are drawn in `c_limb`, the spine and neck in `c_core`.
        `head` is a Sprite to use in place of the plain circle, or None.

        `extent` is how far above the feet the top of the head must land, in
        figure heights; the pose is squashed vertically to make that exact.
        It is the game's number, not the pose's -- see JumpGame.player_h.

        A proper stick figure: one spine, limbs hanging off its two ends. The
        shoulder and hip pairs are collapsed to their midpoints rather than
        drawn as a box, so arms and legs radiate from a single joint each --
        at this size a shoulders/hips trapezoid just reads as a blob with
        sticks attached.
        """
        if self.pts is None:
            return
        h = frame.shape[0]
        size = self.HEIGHT * h
        extent = self.TOP_EXTENT if extent is None else extent

        # Hips above the *planted* foot, live off the pose. max(), not the
        # mean: the more-positive y is the foot on the ground, and a mean
        # would lift the whole body off the ground line whenever one foot is
        # raised. Held across a dropout, because ankles are the least
        # confident keypoints here (0.77 mean) and the first to leave frame.
        ankles = [self.pts[j][1] for j in (coco.LEFT_ANKLE, coco.RIGHT_ANKLE)
                  if self.ok[j]]
        if ankles:
            self._foot = max(ankles)
        foot = self.HIP_FRAC if self._foot is None else self._foot

        # Squash about the feet so the head lands exactly on `extent`. Solved
        # rather than eased into: this is the number the collision test uses,
        # and a drawing that disagreed with it -- in either direction -- would
        # be the worst bug this file could have. x is left alone, so the
        # figure widens as it compresses, which is what reads as a crouch.
        span = None
        if self.ok[coco.LEFT_SHOULDER] and self.ok[coco.RIGHT_SHOULDER]:
            neck_dy = (self.pts[coco.LEFT_SHOULDER][1]
                       + self.pts[coco.RIGHT_SHOULDER][1]) / 2
            span = foot - neck_dy
        head_stack = self.NECK + self.HEAD_H
        k = 1.0 if not span or span <= 0 else (extent - head_stack) / span
        k = min(3.0, max(0.2, k))  # a degenerate pose must not explode
        hip_y = feet_y - k * foot * size

        def at(j):
            return cx + self.pts[j][0] * size, hip_y + k * self.pts[j][1] * size

        def mid(a, b):
            (ax, ay), (bx, by) = at(a), at(b)
            return (ax + bx) / 2, (ay + by) / 2

        def px(pt):
            return int(pt[0]), int(pt[1])

        have = lambda *js: all(self.ok[j] for j in js)  # noqa: E731

        neck = mid(coco.LEFT_SHOULDER, coco.RIGHT_SHOULDER) if have(
            coco.LEFT_SHOULDER, coco.RIGHT_SHOULDER) else None
        pelvis = mid(coco.LEFT_HIP, coco.RIGHT_HIP) if have(
            coco.LEFT_HIP, coco.RIGHT_HIP) else None

        # Thin: at this size the limbs are only ~0.15*size apart, so a fat
        # stroke merges them into a blob rather than reading as a pose.
        limbs = max(2, round(size * 0.028))

        segments = []
        if neck and pelvis:
            segments.append((neck, pelvis, c_core))  # the body core, one line
        for shoulder, elbow, wrist in (
            (coco.LEFT_SHOULDER, coco.LEFT_ELBOW, coco.LEFT_WRIST),
            (coco.RIGHT_SHOULDER, coco.RIGHT_ELBOW, coco.RIGHT_WRIST),
        ):
            root = neck or (at(shoulder) if self.ok[shoulder] else None)
            if root and self.ok[elbow]:
                segments.append((root, at(elbow), c_limb))
                if self.ok[wrist]:
                    segments.append((at(elbow), at(wrist), c_limb))
        for hip, knee, ankle in (
            (coco.LEFT_HIP, coco.LEFT_KNEE, coco.LEFT_ANKLE),
            (coco.RIGHT_HIP, coco.RIGHT_KNEE, coco.RIGHT_ANKLE),
        ):
            root = pelvis or (at(hip) if self.ok[hip] else None)
            if root and self.ok[knee]:
                segments.append((root, at(knee), c_limb))
                if self.ok[ankle]:
                    segments.append((at(knee), at(ankle), c_limb))

        # Outline everything first, then the colour, so the dark edging never
        # lands on top of a neighbouring limb.
        for a, b, _ in segments:
            cv2.line(frame, px(a), px(b), (0, 0, 0), limbs + 3, cv2.LINE_AA)
        for a, b, c in segments:
            cv2.line(frame, px(a), px(b), c, limbs, cv2.LINE_AA)

        # Head sits on the neck joint, not on the nose -- the nose vanishes
        # the moment the player turns away from the camera.
        if not neck:
            return

        if head is None:
            r = max(3, int(self.HEAD_R * size))
            top = (int(neck[0]), int(neck[1] - r * 1.5))
            cv2.line(frame, px(neck), top, (0, 0, 0), limbs + 3, cv2.LINE_AA)
            cv2.line(frame, px(neck), top, c_core, limbs, cv2.LINE_AA)
            cv2.circle(frame, top, r + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, top, r, c_core, -1, cv2.LINE_AA)
            return

        head_h = self.HEAD_H * size
        cy = neck[1] - self.NECK * size - head_h / 2
        # The neck stub stops at the logo's bottom edge rather than running to
        # its centre: the snowflake is hollow in the middle, so a line aimed
        # at the centre would show straight through the gap.
        stub = (int(neck[0]), int(cy + head_h / 2))
        cv2.line(frame, px(neck), stub, (0, 0, 0), limbs + 3, cv2.LINE_AA)
        cv2.line(frame, px(neck), stub, c_core, limbs, cv2.LINE_AA)
        head.draw(frame, neck[0], cy, head_h, head_tint)


class JumpGame:
    """Obstacle-dodging state machine.

    WAITING -> (jump) -> PLAYING -> (hit) -> OVER -> (jump, after a lockout)
    -> PLAYING.
    """

    WAITING, PLAYING, OVER = "waiting", "playing", "over"

    # Speeds scale with the stage: the world is 1.5x bigger and the clock is
    # not, so the same play needs 1.5x the frame widths per second. They then
    # go a little beyond that, because an obstacle twice as wide overlaps the
    # player for twice as long at a given speed, and that overlap is what the
    # jump arc has to outlast.
    # In stage widths per second, tuned on the 16:9 stage. Faster than the
    # stage scale alone would need, because an obstacle twice as wide
    # overlaps the player for twice as long and the arc has to outlast that.
    #
    # SPEED_MAX is held down by reaction time rather than by fairness: going
    # faster only *widens* the jump window (the overlap shrinks), but it eats
    # the runway between an obstacle appearing at the right edge and reaching
    # the player. 1.15 keeps that at 0.65 s at the very top speed.
    # Slower openings are *harder*, not easier, and it is worth being clear
    # about why: an obstacle overlaps the player for (hit + width) / speed
    # seconds and the arc has a fixed time above it, so the slowest moment in
    # the game is the one with the least room for a mistimed jump. The
    # opening is therefore the binding case for fairness, and dropping it 25%
    # takes the worst-case window from 0.174 s to 0.105 s. What it buys back
    # is sighting time: 1.29 s from an obstacle appearing to reaching the
    # player, against 0.98 s before.
    # Scaled by 0.884 from the 0.633/0.011/1.15 these were tuned at, and the
    # reason is a clock fix rather than a design change. Until the fixed
    # timestep landed, `dt = min(dt, 0.05)` fired on ~100% of frames, so the
    # world advanced at 0.918 / 0.849 / 0.851 of wall clock across the three
    # recorded device sessions -- these numbers were quoted in stage widths
    # per *second* and were never delivering that. Fixing the clock made the
    # game 1.13x faster overnight; scaling by the mean 0.884 puts the pace
    # back where it was actually played and tuned by eye.
    #
    # Only the speeds are scaled. Every *duration* here -- REF_AIR, DUCK_MIN,
    # the reaction budgets -- is anchored to a human body, and those were 13%
    # too long under the clamp. Leaving them alone is the fix, not a
    # regression.
    SPEED0 = 0.559
    SPEED_STEP = 0.0097  # added per obstacle cleared
    SPEED_MAX = 1.016

    # Obstacles are spawned on a time gap rather than a distance gap, so
    # speeding the game up shortens the reaction window smoothly instead of
    # suddenly. GAP_MIN is the floor that keeps the game *possible*: the
    # player needs a full arc (REF_AIR) plus time to land and react.
    GAP_MIN, GAP_MAX = 1.7, 2.7
    GAP_FLOOR = 1.2
    GAP_TIGHTEN = 0.03  # per obstacle cleared

    # Comfortably under Player.PEAK_MIN, so that even a weak jump leaves a
    # real window rather than needing frame-perfect timing.
    # Both dimensions in fractions of frame HEIGHT, width included and for
    # the same reason as Player.W_H: the stage aspect is a runtime choice.
    OBST_H = (0.160, 0.280)
    OBST_WMAX = 0.256  # widest an obstacle may be drawn; see _spawn()

    # Artwork is what the obstacles are made of; the list is what they are
    # picked from, uniformly and independently, so two in a row can match.
    # Listed rather than globbed, so a file can sit in images/ without
    # immediately turning up in the game.
    ART = ("ansible.png", "bobr.png", "docker.png", "homebrew.png",
           "debian.png", "opensuse.png", "ubuntu.png", "redhat.png")

    # A hit is only counted once the obstacle is properly overlapping, and
    # the player's line has to be genuinely below its top -- a pixel of
    # clearance should not kill you.
    FORGIVE = 0.048

    # Every obstacle carries its own instruction, just above the picture.
    # Permanent rather than a one-off tutorial prompt: this is a game people
    # walk up to at a booth and play once, so there is no second run in which
    # they would already know.
    LABEL_SCALE = 0.8
    LABEL_GAP = 0.030  # of frame height, from the top of the art to the text
    STRING_W = 0.011  # the hanging string's thickness, of frame height

    # The avatar is small, off to one side, and shares the screen with a live
    # picture of a room -- so the first time anyone sees this, it is worth
    # saying which figure is theirs. Shown in the lobby and for the opening
    # seconds of the session's first run, then never again: by then they have
    # jumped, watched the figure jump with them, and know.
    YOU_HOLD = 2.5  # seconds into the first run
    YOU_DX, YOU_DY = 0.075, 0.10  # arrow tail offset from the head, of frame height

    # The simulation's fixed slice, and how much banked time one frame may
    # burn catching up.
    #
    # STEP is set by tunnelling, since the collision test is a discrete
    # overlap check. The narrowest obstacle the spawner can build is the
    # homebrew mug (trimmed aspect 0.656) at the bottom of OBST_H:
    # 0.160 * 0.656 / (16/9) = 0.059 stage widths, which with the 0.0315-wide
    # hitbox gives a 0.0905-width window. At SPEED_MAX that window lasts
    # 78.7 ms -- so the old 50 ms clamp cleared it by only 1.57x, and only by
    # luck. At 1/120 the player is tested against it 9.4 times.
    #
    # MAX_CATCHUP is one human reaction, the same 0.25 s the BAR_SPACING sum
    # below already budgets for a countermovement. Simulating a longer stall
    # than that only kills the player with an obstacle they never saw. It
    # cannot fire above 4 fps.
    STEP = 1.0 / 120
    MAX_CATCHUP = 0.25

    OVER_LOCKOUT = 1.6  # seconds before a jump can restart the game
    ABSENT_GRACE = 0.5  # seconds without a player before the world pauses

    # -- ducking -----------------------------------------------------------
    #
    # The player's collision height, standing and crouched, in frame heights.
    # Derived from the avatar rather than tuned, because the avatar is drawn
    # at exactly these numbers -- see player_h.
    H_STAND = StickFigure.TOP_EXTENT * StickFigure.HEIGHT  # 0.2914
    H_CROUCH = StickFigure.CROUCH_EXTENT * StickFigure.HEIGHT  # 0.1610

    # Where a ceiling bar's underside sits. The band is fixed by the two
    # heights and FORGIVE:
    #
    #   a crouch must fit:   BAR_GAP >= H_CROUCH - FORGIVE = 0.113
    #   standing must not:   BAR_GAP <  H_STAND  - FORGIVE = 0.243
    #
    # 0.20 rather than the middle of that band, chosen by *visible daylight*.
    # On the 960-tall stage it leaves 37 px of air between the crouched head
    # and the slab, buries 43 px of a standing head in it, and gives the
    # crouch an 84 px margin. The midpoint would leave the crouched head
    # touching a bar it survived, which teaches the player nothing; the
    # asymmetry the other way is deliberate, since a duck the detector
    # believes must always be safe while standing into a bar is never
    # accidental.
    BAR_GAP = 0.20

    # Minimum hold once a crouch starts.
    #
    # The hold has to outlast the widest obstacle's crossing by the same
    # margin the jump arc keeps over its own worst case. A ceiling bar is
    # drawn from the same sprite pool and sized by the same rule as a ground
    # obstacle -- that is what makes the two read as one family -- so it can
    # be as wide as OBST_WMAX, which at SPEED0 crosses the player in 0.314 s.
    #
    #   DUCK_MIN   16:9     4:3 (--stage off)
    #     0.35     1.11x        0.84x   <- the original; 4:3 was already unfair
    #     0.48     1.53x        1.15x   <- fine until the speeds were rescaled
    #     0.53     1.69x        1.27x   <- chosen
    #
    # 0.53 rather than 0.48 because the speed rescale above slowed the
    # obstacles by 11.6%, and a slower obstacle spends *longer* overlapping
    # the player, not less -- the same inversion that makes the game's opening
    # its binding fairness case rather than its top speed.
    #
    # A longer hold costs nothing in play: a jump cancels the crouch instantly
    # (R2), so the only thing it delays is standing up idle.
    DUCK_MIN = 0.53
    # How long a crouch survives the detector saying "not ducking". Covers two
    # consecutive dropped detections at 18.35 fps (109 ms) and seven frames at
    # 60. Deliberately no longer: this is visible lag in a mirror the player
    # is standing in front of, which the arc's own comments argue is worse
    # than a tight window. DUCK_MIN does the work against flicker.
    DUCK_RELEASE = 0.12
    # Entry blend. Long enough that the crouch reads as a movement and that
    # the countermovement dip before a jump barely shows, short enough that
    # the player is geometrically safe 34 ms after committing.
    CROUCH_BLEND = 0.10

    # An even mix: every obstacle after the first is an independent coin flip,
    # so the two verbs are equally common and neither is the "normal" one that
    # the other interrupts.
    #
    # Independent, which means bars *do* come in streaks -- about a quarter of
    # them are followed by another. That is a consequence of the ratio rather
    # than a choice: forbidding two in a row caps bars at 50% only by strict
    # alternation, and an alternating game telegraphs every obstacle. Streaks
    # are the easy case anyway, since one held crouch covers a whole run of
    # them.
    #
    # The first obstacle of a run is always a ground one regardless: start()
    # clears next_is_bar, so the jump that begins the game is followed by
    # something to jump over.
    BAR_P = 0.5

    # The gap a bar needs *after* it, in seconds. The binding transition is
    # crouch -> jump at the opening speed, worst case being a player who
    # ducked at the very instant of contact and so still owes the whole
    # minimum hold when the bar has gone:
    #
    #   0.277  the widest bar finishes crossing
    # + 0.203  the rest of DUCK_MIN (0.48 less the 0.277 already served)
    # + 0.120  the release debounce (DUCK_RELEASE)
    # + 0.250  countermovement into the next jump
    # + 0.163  takeoff confirmed (3 frames at 18 fps)
    # + 0.200  the arc rises to clearance
    # = 1.213 s
    #
    # GAP_FLOOR is 1.2, which clears that by nothing at all. 1.6 gives 1.32x.
    #
    # Only after a bar, and only into a ground obstacle. The other three
    # transitions already fit inside GAP_FLOOR: jump -> crouch costs 0.81 s,
    # and anything -> crouch after a crouch costs nothing at all, because the
    # player can simply stay down. Widening every gap next to a bar would, at
    # an even mix, apply to three quarters of them and quietly retire
    # GAP_TIGHTEN -- leaving the difficulty ramp to speed alone.
    BAR_SPACING = 1.6

    def __init__(self, ground=0.96, rng=None, aspect=4 / 3, bars=True):
        self.ground = ground
        self.bars = bars  # master switch; --no-duck turns the mechanic off
        # How high the arc may actually go here. PEAK_MAX is tuned against
        # the default ground line; a line drawn higher up the frame leaves
        # less room above it, and without this the avatar's head would simply
        # go off the top. Never below PEAK_MIN -- clearing obstacles wins
        # over staying in frame, since one is a rule and the other is looks.
        # TOP_EXTENT, deliberately -- the *standing* extent, never the live or
        # crouched one. The arc budget must not change when the player
        # crouches: a crouch only lowers the head, so it can only gain
        # headroom, and letting it raise peak_max would make the same jump
        # reach different heights depending on what the player did before it.
        headroom = ground - StickFigure.TOP_EXTENT * StickFigure.HEIGHT - 0.04
        self.peak_max = max(Player.PEAK_MIN, min(Player.PEAK_MAX, headroom))
        # Frame width/height. Needed because a sprite's proportions are fixed
        # while the game's two axes are scaled independently -- turning "as
        # wide as it is tall" into a width *fraction* takes the frame's own
        # aspect. Obstacles already changed shape with it (w is a fraction of
        # width, h of height), so this narrows no invariant that held before.
        self.aspect = aspect
        # Horizontal player geometry, converted from heights to fractions of
        # this stage's width once, here, rather than at every use.
        self.pw = Player.W_H / aspect
        self.hit_w = Player.HIT_W_H / aspect
        self.rng = rng or random.Random()
        self.player = Player()
        self.figure = StickFigure()
        # None when images/ is absent -- the figure falls back to its circle.
        self.head = sprite.Sprite.load("nixos.png")
        self.over = gameover.GameOverScreen()
        self.taunt = None  # chosen at death, held for the whole screen
        self.killer = None
        self.art = [s for s in map(sprite.Sprite.load, self.ART) if s is not None]
        self.state = self.WAITING
        self.obstacles = []
        self.score = 0
        self.best = 0
        self.speed = self.SPEED0
        self.next_gap = 0.0
        self.since_over = 0.0
        self.absent = 0.0
        # Duck state. `crouched` is the latch, `duck_t` how long it has been
        # held, `duck_off` how long the detector has been quiet inside it, and
        # `crouch_k` the 0..1 blend that both the drawing and the collision
        # height are derived from.
        self.crouched = False
        self.duck_t = 0.0
        self.duck_off = 0.0
        self.crouch_k = 0.0
        # One boolean of lookahead. Deciding the *next* obstacle's kind when
        # this one's gap is scheduled is what buys a >= BAR_SPACING gap on
        # both sides of every bar, with no lookbehind and no rescan.
        self.next_is_bar = False
        # Session-level, like `best`: the onboarding prompt stops for good the
        # first time the player actually clears a bar.
        self.ducks_cleared = 0
        self.runs = 0  # runs started this session; the first one gets the hint
        self.elapsed = 0.0  # seconds into the current run
        self._accum = 0.0  # unspent simulation time, carried between frames
        self._pending_launch = None

    def set_pose(self, pose, index):
        """Hand the avatar the player's current keypoints (index may be None).

        Ignored once the run is over, which freezes the figure in the pose it
        died in. The death screen is a photograph of the crash -- the avatar
        mid-stride against the obstacle that got it -- and a figure that goes
        on mirroring the player while they walk away and shrug is not that.
        """
        if self.state == self.OVER:
            return
        if index is None:
            self.figure.update(None, None, None)
        else:
            self.figure.update(
                pose.xy[index],
                pose.kp_conf[index] if pose.kp_conf is not None else None,
                pose.box_xyxy[index],
            )

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        self.obstacles = []
        self.score = 0
        self.speed = self.SPEED0
        self.player.reset()
        self.crouched = False
        self.duck_t = self.duck_off = self.crouch_k = 0.0
        self.next_is_bar = False
        self._accum = 0.0
        self._pending_launch = None
        self.runs += 1
        self.elapsed = 0.0
        # Give the player a moment before the first obstacle arrives.
        self.next_gap = 1.6
        self.state = self.PLAYING

    def die(self, obstacle=None):
        """End the run. `obstacle` is whatever hit you, for the taunt.

        The taunt is drawn now, through the game's own rng, rather than at
        draw time -- the screen is redrawn every frame and would otherwise
        reshuffle its own joke sixty times a second, and picking here keeps a
        seeded replay identical.
        """
        self.best = max(self.best, self.score)
        self.since_over = 0.0
        self.killer = obstacle.art if obstacle is not None else None
        # Keyed on the artwork, and only on the artwork. Both kinds of
        # obstacle are drawn from the same sprites, so the joke is about the
        # logo that got you and never about which way you failed to move --
        # which is also why none of the taunts names a verb any more.
        self.taunt = self.over.pick(
            self.killer.name if self.killer is not None else None, self.rng)
        self.state = self.OVER

    # -- per frame ---------------------------------------------------------

    @property
    def player_h(self):
        """The player's height now, in frame heights.

        The one number that is both what is drawn and what kills you. draw()
        hands it to the figure as an extent and _hits() tests against it, so
        there is no frame in which the avatar's silhouette and its collision
        box can disagree.
        """
        return self.H_STAND + (self.H_CROUCH - self.H_STAND) * self.crouch_k

    def _step_duck(self, dt, ducking):
        """Advance the crouch latch. `ducking` is the detector's raw verdict."""
        if self.crouched:
            self.duck_t += dt
            self.duck_off = 0.0 if ducking else self.duck_off + dt
            if self.duck_t >= self.DUCK_MIN and self.duck_off >= self.DUCK_RELEASE:
                self.crouched = False
                self.duck_t = self.duck_off = 0.0
        elif ducking:
            self.crouched = True
            self.duck_t = self.duck_off = 0.0
        # Entry is blended, release is instant. A release blend would leave
        # the drawn head low while the collision height had already risen --
        # drawn safe, actually dead, the one mismatch that must never happen.
        self.crouch_k = (min(1.0, self.duck_t / self.CROUCH_BLEND)
                         if self.crouched else 0.0)

    def _duck_input(self, ducking, player_present):
        """What the crouch latch is actually fed this frame.

        Two rules fold in here. You cannot crouch in the air (R3), so a
        takeoff's countermovement cannot latch a crouch that outlives it. And
        an absent player is not evidence either way -- the same rule the
        takeoff detector applies to a dropped frame -- so a player who steps
        out of shot while crouched under a frozen bar does not stand up in
        the freeze and die on the frame they come back.
        """
        if not player_present:
            return self.crouched
        return ducking and not self.player.airborne

    def update(self, dt, jump_lift=None, player_present=True, ducking=False):
        """Advance by `dt` seconds.

        `jump_lift` is None for "no takeoff this frame", or the measured lift
        in body heights for a confirmed one -- which is what scales the arc.
        `ducking` is the detector's per-frame verdict; the minimum hold and
        the release debounce are applied here, not there.
        """
        jumped = jump_lift is not None
        # There is deliberately no `dt = min(dt, 0.05)` here any more.
        #
        # That clamp was the only thing standing between a slow frame and an
        # obstacle teleporting through the player, and it fired on ~100% of
        # frames -- so the world ran at 70-92% of wall clock by a factor that
        # changed with the frame rate, and none of the seconds quoted
        # anywhere in this file were true. The substep loop below stops the
        # teleporting now, and MAX_CATCHUP bounds a stall.
        #
        # Keeping both would have been worse than either: at 0.05 the clamp
        # capped what the accumulator could ever bank, so MAX_CATCHUP's 0.25
        # was unreachable and the catch-up rule was quietly dead. One bound,
        # in one place, is the whole point.

        if self.state == self.WAITING:
            # Stepped outside PLAYING too, so the lobby doubles as a practice
            # ground and the crouch is visible before it can cost anything.
            self._step_duck(dt, self._duck_input(ducking, player_present))
            if jumped:
                self.start()
            return

        if self.state == self.OVER:
            self.since_over += dt
            # Neither the arc nor the crouch advances. The arc used to finish
            # its flight here, on the grounds that it looked better than
            # stopping dead -- but that was when the death panel covered the
            # middle of the screen and the avatar was mostly hidden behind
            # it. Now the crash is on show in the left third, and what looks
            # better is the frozen moment of impact.
            if jumped and self.since_over >= self.OVER_LOCKOUT:
                self.start()
            return

        # PLAYING: fixed substeps, with the remainder carried.
        self.elapsed += dt
        # If the player has walked out of frame, freeze the world rather than
        # killing them for something they cannot see or react to.
        self.absent = 0.0 if player_present else self.absent + dt

        if jumped:
            # Held rather than applied here, because a tick shorter than one
            # substep runs the loop below zero times and would otherwise drop
            # the takeoff on the floor.
            self._pending_launch = jump_lift
        self._accum = min(self._accum + dt, self.MAX_CATCHUP)
        while self._accum >= self.STEP and self.state == self.PLAYING:
            self._accum -= self.STEP
            self._substep(self.STEP, ducking, player_present)

    def _substep(self, dt, ducking, player_present):
        """One fixed slice of the world. The only place obstacles move."""
        if self._pending_launch is not None:
            self.player.launch(self._pending_launch, self.peak_max)
            # A jump always beats a duck. Takeoff is the measured,
            # false-positive-free event, while a crouch left latched across a
            # real takeoff is fatal against the next ground obstacle.
            self.crouched = False
            self.duck_t = self.duck_off = self.crouch_k = 0.0
            self._pending_launch = None
        self.player.step(dt)
        # Recomputed per substep rather than once per frame: `airborne`
        # changes inside the arc, and rule R3 keys the crouch off it.
        self._step_duck(dt, self._duck_input(ducking, player_present))
        if self.absent > self.ABSENT_GRACE:
            return
        self._advance_obstacles(dt)

    def _spawn(self, ceiling=False):
        """One new obstacle at the right-hand edge.

        Both kinds are the same picture, sized by the same rule; the only
        thing `ceiling` changes is how high off the ground it is hung, and
        that a wall is stacked on top of it. That is the whole design: the
        player reads *where* the obstacle is, not what it is made of.

        Height is normally the free variable and width follows from the
        artwork's proportions: height is what decides whether a jump clears
        the thing, so it is the dimension that has to stay in the tuned range.

        Wide artwork is capped the other way round -- by width, losing height
        to keep its proportions. Width is not free either. An obstacle
        overlaps the player for (HIT_W + w) / speed seconds and both the jump
        arc and the duck's minimum hold are budgeted to outlast exactly that,
        so a picture wider than the widest block the timings were tuned
        against eats the margin: the Docker whale at full height would be
        0.092 frames wide, near double OBST_W. Capping costs it some height
        instead, which both budgets have room for.
        """
        art = self.rng.choice(self.art) if self.art else None
        if art is None:
            w_h = self.rng.uniform(self.OBST_WMAX * 0.58, self.OBST_WMAX)
            w, h = w_h / self.aspect, self.rng.uniform(*self.OBST_H)
        else:
            # Tallest this art may stand before it is over-wide, with the
            # floor held in the same proportion so squat art still varies in
            # size rather than collapsing to one. h * art.aspect is the width
            # in heights, which is what OBST_WMAX bounds; the divide by the
            # stage aspect at the end is the only place the monitor's shape
            # enters.
            hi = min(self.OBST_H[1], self.OBST_WMAX / art.aspect)
            lo = hi * self.OBST_H[0] / self.OBST_H[1]
            h = self.rng.uniform(lo, hi)
            w = h * art.aspect / self.aspect

        if not ceiling:
            return Obstacle(x=1.02, w=w, h=h, art_h=h, art=art)

        # Hung at head height, with the collision box running off the top of
        # the frame. The box has to: a jump puts the player's head 0.87 above
        # the ground line, so anything less and a good jump simply clears the
        # obstacle and the duck becomes optional.
        #
        # What is *drawn* up there is a string, not a filled box, and that is
        # a deliberate trade -- it looks like an object hung from the ceiling
        # rather than a slab, at the cost of the drawing covering less than
        # the collision does. The gap only bites a player who jumps at a
        # hanging obstacle, who was going to die either way; a player who
        # ducks never meets the top of the box at all.
        gap = min(self.BAR_GAP, max(0.10, self.ground - 0.06))
        return Obstacle(x=1.02, w=w, y=gap, h=self.ground - gap + 0.04,
                        art_h=h, art=art)

    def _next_kind(self, _last_was_bar):
        """Is the obstacle after this one a ceiling bar? A coin flip.

        Takes the previous kind and ignores it, deliberately: an even mix and
        a no-repeats rule cannot both hold, and of the two the mix is what
        keeps either verb from feeling like the exception.
        """
        return self.bars and self.rng.random() < self.BAR_P

    def _advance_obstacles(self, dt):
        for o in self.obstacles:
            o.x -= self.speed * dt
        self.obstacles = [o for o in self.obstacles if o.x + o.w > -0.05]

        self.next_gap -= dt
        if self.next_gap <= 0:
            o = self._spawn(ceiling=self.next_is_bar)
            self.obstacles.append(o)
            self.next_is_bar = self._next_kind(o.ceiling)
            tighten = self.GAP_TIGHTEN * self.score
            lo = max(self.GAP_FLOOR, self.GAP_MIN - tighten)
            hi = max(lo + 0.2, self.GAP_MAX - tighten)
            gap = self.rng.uniform(lo, hi)
            # Standing up and jumping is the one transition GAP_FLOOR does
            # not comfortably cover, so it is the one that gets widened.
            # Deciding the next kind here rather than at the next spawn is
            # what makes the test cheap: both sides of the gap are known.
            if o.ceiling and not self.next_is_bar:
                gap = max(gap, self.BAR_SPACING)
            self.next_gap = gap

        pad = (self.pw - self.hit_w) / 2
        px0, px1 = Player.X + pad, Player.X + self.pw - pad
        for o in self.obstacles:
            overlapping = o.x < px1 and o.x + o.w > px0
            if overlapping and self._hits(o):
                self.die(o)
                return
            if not o.passed and o.x + o.w <= px0:
                o.passed = True
                self.score += 1
                if o.ceiling:
                    self.ducks_cleared += 1
                self.speed = min(self.SPEED_MAX, self.speed + self.SPEED_STEP)

    def _hits(self, o):
        """Do the player and `o` overlap vertically?

        One AABB for both shapes. For a ground obstacle (y = 0) the second
        term is true for any player height at all, so this is exactly the
        `lift < o.h - FORGIVE` it replaces; for a ceiling bar the first term
        is trivially true and it collapses to "your head is in the slab".
        FORGIVE now serves both faces of the box, in the same currency.
        """
        lo = self.player.lift
        hi = lo + self.player_h
        return lo < o.y + o.h - self.FORGIVE and hi > o.y + self.FORGIVE

    # -- drawing -----------------------------------------------------------

    def draw(self, frame):
        h, w = frame.shape[:2]
        ground_y = int(self.ground * h)
        cv2.line(frame, (0, ground_y), (w, ground_y), C_GROUND, 2, cv2.LINE_AA)

        for o in self.obstacles:
            x0, x1 = int(o.x * w), int((o.x + o.w) * w)
            bot = int(ground_y - o.y * h)
            top = int(bot - o.h * h)
            # The artwork occupies the bottom `art_h` of the box; for a ground
            # obstacle that is the whole of it, and for a ceiling bar the rest
            # is wall.
            art_top = int(bot - (o.art_h or o.h) * h)
            if o.ceiling:
                # The string it hangs from, drawn before the picture so the
                # picture sits on the end of it. Thick, because it is the only
                # thing telling the player this obstacle is suspended rather
                # than floating, and outlined so it survives a lit room.
                sx = (x0 + x1) // 2
                thick = max(4, round(h * self.STRING_W))
                cv2.line(frame, (sx, top), (sx, art_top), (0, 0, 0),
                         thick + 4, cv2.LINE_AA)
                cv2.line(frame, (sx, top), (sx, art_top), C_STRING,
                         thick, cv2.LINE_AA)
            if o.art is not None:
                # Sized by height and standing on its own edge, so the art
                # fills exactly the part of the box that is not wall.
                o.art.draw(frame, (x0 + x1) / 2, (art_top + bot) / 2,
                           bot - art_top)
            else:
                cv2.rectangle(frame, (x0, art_top), (x1, bot), C_OBSTACLE, -1)
                cv2.rectangle(frame, (x0, art_top), (x1, bot), (0, 0, 0), 2)
            # The instruction, in the same place for both kinds: up and to
            # the left of the picture, clear of the middle. Offset rather
            # than centred because a hanging obstacle's string comes down
            # exactly there, and a white word on a white string is a white
            # smudge. Ground obstacles take the same offset so the two kinds
            # go on reading alike, which is the whole point of them being the
            # same picture in the first place.
            label = "DUCK" if o.ceiling else "JUMP"
            size = self.LABEL_SCALE * h / 700
            (tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           size, 2)
            # Right edge of the word just clear of the string, whatever the
            # word and whatever the resolution.
            clear = max(4, round(h * self.STRING_W)) / 2 + max(3, round(h * 0.006))
            self._draw_text(frame, label,
                            ((x0 + x1) / 2 - tw / 2 - clear) / w,
                            (art_top - self.LABEL_GAP * h) / h,
                            scale=self.LABEL_SCALE)

        feet_y = ground_y - self.player.lift * h
        cx = (Player.X + self.pw / 2) * w
        dead = self.state == self.OVER
        c_limb = C_DEAD if dead else C_LIMB
        c_core = C_DEAD if dead else C_CORE
        if self.figure.pts is None:
            # Nobody seen yet -- fall back to the bar so there is still
            # something standing on the ground line.
            x0, x1 = int(Player.X * w), int((Player.X + self.pw) * w)
            thick = max(4, round(h * 0.014))
            cv2.line(frame, (x0, int(feet_y)), (x1, int(feet_y)),
                     c_limb, thick, cv2.LINE_AA)
            # A vertical stroke too, so the collision box is legible in
            # blocks-and-circles mode -- otherwise a bar kills a bar of a
            # player with nothing on screen to explain why.
            cx0 = int((Player.X + self.pw / 2) * w)
            cv2.line(frame, (cx0, int(feet_y)),
                     (cx0, int(feet_y - self.player_h * h)), c_limb, thick,
                     cv2.LINE_AA)
        else:
            self.figure.draw(frame, cx, feet_y, c_limb, c_core, self.head,
                             C_DEAD if dead else None,
                             extent=self.player_h / StickFigure.HEIGHT)

        # Not on the death screen: `elapsed` is frozen there along with
        # everything else, so without the state test the arrow would sit on
        # the crash for as long as the panel is up -- and the crash is the
        # thing the panel moved aside to show.
        if self.state == self.WAITING or (
                self.state == self.PLAYING
                and self.runs <= 1 and self.elapsed < self.YOU_HOLD):
            self._draw_you(frame, cx, feet_y - self.player_h * h)

        if self.state != self.OVER:
            self._draw_text(frame, f"{self.score}", 0.5, 0.10, scale=1.6, weight=3)
        if self.state == self.WAITING:
            self._draw_text(frame, "JUMP TO START", 0.5, 0.30, scale=1.1)

        elif self.state == self.OVER and self.taunt is not None:
            self.over.draw(frame, self.taunt, self.killer, self.score, self.best,
                           self.since_over, self.OVER_LOCKOUT)
        elif self.absent > self.ABSENT_GRACE:
            self._draw_text(frame, "STEP INTO FRAME", 0.5, 0.30, scale=1.1)

    def _draw_you(self, frame, cx, head_y):
        """Point at the avatar and name it, for someone who has just walked up.

        Aimed down-left at the head from up-right, which is the one direction
        with nothing in it: the camera panel is above, the obstacles arrive
        from the right along the ground, and the stage edge is immediately to
        the left.
        """
        h = frame.shape[0]
        tail = (int(cx + self.YOU_DX * h), int(head_y - self.YOU_DY * h))
        tip = (int(cx + 0.012 * h), int(head_y - 0.02 * h))
        cv2.arrowedLine(frame, tail, tip, (0, 0, 0),
                        max(5, round(h * 0.008)), cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(frame, tail, tip, C_TEXT,
                        max(3, round(h * 0.004)), cv2.LINE_AA, tipLength=0.3)
        self._draw_text(frame, "YOU", (tail[0] + 0.022 * h) / frame.shape[1],
                        (tail[1] - 0.005 * h) / h, scale=1.0)

    @staticmethod
    def _draw_text(frame, text, fx, fy, scale=1.0, colour=C_TEXT, weight=2):
        """Centre `text` on (fx, fy), outlined so it reads over any background."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = scale * h / 700
        (tw, th), _ = cv2.getTextSize(text, font, size, weight)
        org = (int(fx * w - tw / 2), int(fy * h + th / 2))
        cv2.putText(frame, text, org, font, size, (0, 0, 0), weight + 4, cv2.LINE_AA)
        cv2.putText(frame, text, org, font, size, colour, weight, cv2.LINE_AA)
