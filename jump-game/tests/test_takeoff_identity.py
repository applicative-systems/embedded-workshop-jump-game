"""Whose baseline is it? Person identity in TakeoffDetector while unlocked.

The session this pins: two people in and out of shot in the lobby, then one
left alone who could not start the game. The baseline had been keyed on the
detection index, and index 0 is simply whoever scores best, so the lone player
inherited the other person's box median. Their box was over BOX_HI of it,
every frame was rejected, rejected frames never reach the median -- and no
jump could ever fire again.
"""

import coco
from takeoff import TakeoffDetector

FPS = 18.35
FLOOR = 950.0


def person(cx, h, lift=0.0, conf=0.9):
    """A standing figure `h` px tall, feet `lift` px off the floor."""
    w = 0.35 * h
    top, bot = FLOOR - h - lift, FLOOR - lift
    kp = [[cx, top + 0.5 * h, conf] for _ in range(coco.N)]
    for j in (coco.LEFT_SHOULDER, coco.RIGHT_SHOULDER):
        kp[j] = [cx, top + 0.15 * h, conf]
    for j in (coco.LEFT_HIP, coco.RIGHT_HIP):
        kp[j] = [cx, top + 0.5 * h, conf]
    for j in (coco.LEFT_ANKLE, coco.RIGHT_ANKLE):
        kp[j] = [cx, bot - 0.03 * h, conf]
    return kp, [cx - w / 2, top, cx + w / 2, bot], conf


class Pose:
    def __init__(self, *people):
        self.xy = [[p[:2] for p in kp] for kp, _box, _c in people]
        self.kp_conf = [[p[2] for p in kp] for kp, _box, _c in people]
        self.box_xyxy = [box for _kp, box, _c in people]
        self.box_conf = [c for _kp, _box, c in people]


def run(det, frames, *people):
    return [j for _ in range(frames) if (j := det.update(Pose(*people)))]


def jump(det, cx, h, *others):
    """One takeoff-and-landing, feet up to 35% of body height."""
    return [j for f in (0, 0.15, 0.3, 0.35, 0.3, 0.15, 0)
            if (j := det.update(Pose(person(cx, h, f * h), *others)))]


def test_a_lone_player_after_a_smaller_one_can_still_jump():
    det = TakeoffDetector(fps=FPS)
    # Far away and scoring higher, so they are index 0 and the one selected.
    far = person(400, 420, conf=0.93)
    close = person(1200, 840, conf=0.90)
    run(det, 60, far, close)
    run(det, 40, close)
    assert jump(det, 1200, 840)


def test_a_lone_player_after_a_bigger_one_can_still_jump():
    det = TakeoffDetector(fps=FPS)
    run(det, 60, person(1200, 840, conf=0.93), person(400, 420, conf=0.90))
    run(det, 40, person(400, 420))
    assert jump(det, 400, 420)


def test_someone_else_on_the_same_spot_can_still_jump():
    """Same place, other distance: nothing sideways to tell them apart by."""
    det = TakeoffDetector(fps=FPS)
    run(det, 60, person(600, 420))
    run(det, 40, person(600, 840))
    assert jump(det, 600, 840)


def test_a_jump_and_a_dropout_keep_the_baseline():
    det = TakeoffDetector(fps=FPS)
    run(det, 60, person(600, 800))
    body = det._body
    assert jump(det, 600, 800)
    run(det, 20)  # out of shot: not evidence of anybody new
    run(det, 20, person(610, 790))
    assert det._body == body
    assert len(det._ys) > 40, "the baseline was rebuilt for the same person"
