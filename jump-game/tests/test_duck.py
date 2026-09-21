"""The duck state machine, and the detector plumbing around it.

Driven by hand, the way test_jump_detector.py drives update_pose: no model, no
frame, no GPU. The detector-level tests build poses out of literal numbers
rather than replaying a fixture, because the behaviour they pin -- a held
crouch, a shrunken bounding box -- is exactly what no recorded session
contains yet.
"""

import pytest

import coco
import duck as duck_mod
from duck import DuckState
from takeoff import TakeoffDetector


# -- the state machine alone ----------------------------------------------

DEEP = 0.5  # a descent well past any plausible threshold


def feed(d, values):
    return [d.update(v) for v in values]


def enter(d):
    """Feed exactly enough frames to confirm a duck. Returns the events."""
    return feed(d, [DEEP] * d.confirm)


def test_fewer_than_confirm_frames_is_not_a_duck():
    d = DuckState()
    feed(d, [DEEP] * (d.confirm - 1))
    assert not d.ducking


def test_confirm_consecutive_frames_enter():
    d = DuckState()
    assert enter(d) == [None] * (d.confirm - 1) + ["enter"]
    assert d.ducking


def test_the_run_must_be_consecutive():
    d = DuckState()
    for _ in range(5):
        feed(d, [DEEP] * (d.confirm - 1) + [0.0])
    assert not d.ducking


def test_hysteresis_holds_between_the_thresholds():
    """Above RISE_BACK but below DROP: sustains a duck, never starts one."""
    between = (duck_mod.DROP + duck_mod.RISE_BACK) / 2
    idle = DuckState()
    feed(idle, [between] * 20)
    assert not idle.ducking

    down = DuckState()
    enter(down)
    feed(down, [between] * 20)
    assert down.ducking


def test_a_single_frame_below_rise_back_does_not_exit():
    d = DuckState()
    enter(d)
    d.update(0.0)
    assert d.ducking
    d.update(0.0)
    assert not d.ducking


def test_none_holds_state_in_both_directions():
    """A dropped frame is not evidence either way."""
    d = DuckState()
    assert d.update(None) is None
    enter(d)
    assert feed(d, [None] * 50) == [None] * 50
    assert d.ducking


def test_max_down_expires_and_lands_in_idle():
    """The safety valve, so a latched false DOWN heals itself."""
    d = DuckState(max_down=10)
    enter(d)
    for i in range(10):
        event = d.update(DEEP)
        if event == "expire":
            assert i == 9
            assert not d.ducking
            break
    else:
        pytest.fail("MAX_DOWN never fired")


def test_expiry_re_enters_unless_the_caller_clears_the_baseline():
    """Documents why "expire" is a distinct event from "exit".

    The state machine on its own oscillates: the descent it is being fed is
    still over DROP at the moment of expiry, so it re-enters two frames later.
    Breaking that loop is the caller's job -- TakeoffDetector clears the duck
    baseline on "expire" so the next descent is measured from the crouch.
    """
    d = DuckState(max_down=10)
    enter(d)
    events = feed(d, [DEEP] * 30)
    assert "expire" in events
    assert events.count("enter") >= 1  # it came straight back
    assert d.ducking


def test_reset_clears_a_partial_run():
    d = DuckState()
    feed(d, [DEEP] * (d.confirm - 1))
    d.reset()
    feed(d, [DEEP] * (d.confirm - 1))
    assert not d.ducking


# -- the detector around it ------------------------------------------------

STAND_H = 800.0


def pose(shoulder_y, box_h=STAND_H, *, ankle_y=None, conf=0.9):
    """One person, standing at a fixed spot, with the given shoulder height.

    The box bottom is pinned so a shorter box means a lower head, which is
    what a crouch actually looks like to the model.
    """
    bottom = 900.0
    kp = [[500.0, 400.0, conf] for _ in range(coco.N)]
    for j in (coco.LEFT_SHOULDER, coco.RIGHT_SHOULDER):
        kp[j] = [500.0, shoulder_y, conf]
    for j in (coco.LEFT_ANKLE, coco.RIGHT_ANKLE):
        kp[j] = [500.0, bottom if ankle_y is None else ankle_y, conf]
    for j in (coco.LEFT_HIP, coco.RIGHT_HIP):
        kp[j] = [500.0, bottom - 0.47 * box_h, conf]

    class P:
        xy = [[(k[0], k[1]) for k in kp]]
        kp_conf = [[k[2] for k in kp]]
        box_xyxy = [[440.0, bottom - box_h, 560.0, bottom]]
        box_conf = [conf]

        def __len__(self):
            return 1

    return P()


def stand(det, n):
    """n frames of standing, to build a baseline."""
    for _ in range(n):
        det.update(pose(900.0 - 0.808 * STAND_H))


def test_a_held_crouch_does_not_expire_itself():
    """The regression that pins the frozen baseline.

    A rolling median follows whatever you do. Without the freeze this crouch
    ends about 23 frames in, with the player still crouched -- which is the
    whole reason the baseline is held while DOWN.
    """
    det = TakeoffDetector(fps=18.35)
    stand(det, 60)
    crouch_y = 900.0 - 0.808 * STAND_H + 0.25 * STAND_H
    # Comfortably past the 45-frame baseline window -- which is the point;
    # an unfrozen median is dominated by crouch samples after ~23 of them.
    # Stops short of MAX_DOWN, which is a separate rule with its own test.
    held = duck_mod.MAX_DOWN - 5
    seen = []
    for _ in range(held):
        det.update(pose(crouch_y, box_h=0.72 * STAND_H))
        seen.append(det.ducking)
    assert seen[3:] == [True] * (len(seen) - 3), (
        f"duck ended after {seen.index(False, 3)} frames of a held crouch")


def test_a_deep_crouch_is_still_measured_when_the_takeoff_path_drops_it():
    """box_h at 0.50 of standing fails takeoff's guard and passes the duck's."""
    det = TakeoffDetector(fps=18.35)
    stand(det, 60)
    crouch_y = 900.0 - 0.808 * STAND_H + 0.30 * STAND_H
    for _ in range(6):
        jump = det.update(pose(crouch_y, box_h=0.50 * STAND_H))
        assert jump is None
    assert det.ducking


def test_a_spurious_box_is_rejected_by_both_paths():
    det = TakeoffDetector(fps=18.35)
    stand(det, 60)
    before = det.descent
    for _ in range(5):
        det.update(pose(880.0, box_h=0.10 * STAND_H))
    assert not det.ducking
    assert det.descent == before


def test_the_descent_is_scale_invariant():
    """The same crouch twice as far from the camera reads the same."""
    def run(scale):
        det = TakeoffDetector(fps=18.35)
        box = STAND_H * scale
        for _ in range(60):
            det.update(pose(900.0 - 0.808 * box, box_h=box))
        for _ in range(4):
            det.update(pose(900.0 - 0.808 * box + 0.25 * box, box_h=0.75 * box))
        return det.descent

    assert run(1.0) == pytest.approx(run(0.5), abs=1e-9)


def test_a_confirmed_jump_cancels_a_duck():
    det = TakeoffDetector(fps=18.35)
    stand(det, 60)
    crouch_y = 900.0 - 0.808 * STAND_H + 0.25 * STAND_H
    for _ in range(4):
        det.update(pose(crouch_y, box_h=0.75 * STAND_H))
    assert det.ducking
    det._duck.reset()  # what update() does on a Jump
    assert not det.ducking


def test_duck_signal_off_never_ducks():
    det = TakeoffDetector(fps=18.35, duck_signal="off")
    stand(det, 60)
    crouch_y = 900.0 - 0.808 * STAND_H + 0.30 * STAND_H
    for _ in range(30):
        det.update(pose(crouch_y, box_h=0.70 * STAND_H))
    assert not det.ducking
