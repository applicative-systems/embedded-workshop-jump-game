"""Replay a real recorded session through the real detector and the real game.

`tests/fixtures/draft2.jsonl` is one session captured with `--record`: 1425
frames, 26 jumps, one player, yolo11n-pose at 18.35 fps. It holds raw
keypoints, so everything below runs with no GPU, no model and no camera --
the same trick `tests/fixtures/*.csv` plays, extended to whole poses.

The golden list in TAKEOFFS is the safety net for restructuring
`TakeoffDetector.update()`. It was captured *before* any duck code existed and
must not move: a duck path added alongside the takeoff path is only correct if
the takeoff path still fires on exactly these frames.
"""

import pathlib

import pytest

import duck as duck_mod
from record import RecordedPose, load
from takeoff import TakeoffDetector

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "draft2.jsonl"
FPS = 18.35

# Every frame today's detector fires on, replaying the fixture unlocked.
# Identical to the `jump` events the live session actually recorded.
TAKEOFFS = [92, 152, 196, 233, 274, 310, 382, 441, 481, 525, 576, 621, 653,
            688, 741, 801, 863, 917, 975, 1017, 1067, 1120, 1160, 1225, 1283,
            1329]


@pytest.fixture(scope="module")
def frames():
    _header, frames, _summary = load(FIXTURE)
    return frames


def replay(frames, **kw):
    """Drive a fresh detector over the whole trace; return it and its takeoffs."""
    det = TakeoffDetector(fps=FPS, **kw)
    fired = [j.frame for e in frames if (j := det.update(RecordedPose(e)))]
    return det, fired


def test_the_fixture_is_the_session_we_think_it_is(frames):
    header, _frames, summary = load(FIXTURE)
    assert header["model"] == "yolo11n-pose"
    assert header["signal"] == "ankle"
    assert summary["jumps"] == 26
    assert len(frames) == 1425
    # One person throughout, and never lost: the trace exercises the detector,
    # not its person-selection fallbacks.
    assert all(e["sel"] is not None for e in frames)


def test_takeoffs_are_unchanged(frames):
    """The regression that guards every future change to update().

    If this fails, the takeoff path moved. Nothing about ducking is allowed to
    move it -- the duck runs on a different signal, a different threshold and
    (deliberately) a looser box guard, and shares only the person selection.
    """
    _det, fired = replay(frames)
    assert fired == TAKEOFFS


def test_the_replay_matches_what_the_live_session_recorded(frames):
    """Not just self-consistent -- consistent with the run that produced it."""
    assert [e["f"] for e in frames if e["jump"] is not None] == TAKEOFFS


# Every recorded session in the repo, all of them jumping only. The peaks are
# what DROP was chosen against; jump-logitech is the one that matters, because
# it pre-loads its jumps half again as deeply as draft2 does and it is what
# caught a threshold tuned on draft2 alone. jump-framework-builtin read 0.103
# while the baseline was keyed on the detection index: at frame 80 the mirror
# reflection outscores the player, became index 0, and was measured into the
# player's baseline. See test_takeoff_identity.py.
SESSIONS = [("draft2.jsonl", 0.090),
            ("jump-framework-builtin.jsonl", 0.094),
            ("jump-logitech.jsonl", 0.141)]


def descents(name):
    _header, frames, summary = load(FIXTURE.parent / name)
    det = TakeoffDetector(fps=(summary or {}).get("fps") or FPS)
    out = []
    for e in frames:
        det.update(RecordedPose(e))
        out.append((det.descent, det.ducking))
    return out


@pytest.mark.parametrize("name, _peak", SESSIONS)
def test_a_session_with_no_ducks_produces_no_ducks(name, _peak):
    """Not "few" false ducks. None.

    A phantom duck is as expensive as a phantom jump: it crouches the avatar
    into a ground obstacle it would otherwise have cleared.
    """
    assert [i for i, (_d, on) in enumerate(descents(name)) if on] == []


@pytest.mark.parametrize("name, peak", SESSIONS)
def test_the_descent_never_gets_close_to_the_threshold(name, peak):
    """The canary.

    Every one of these peaks is a countermovement before a jump, not a
    landing. If new weights, a new camera or a new body erode the margin, this
    fails before a player ever sees a phantom duck.
    """
    worst = max(d for d, _on in descents(name))
    assert worst == pytest.approx(peak, abs=0.002)
    assert worst < duck_mod.DROP


def test_the_countermovement_never_confirms_a_duck(frames):
    """You dip before you leap -- on 26 of 26 takeoffs in this session.

        frames before takeoff    -8     -6     -4     -1     +1     +7
        median descent          .042   .068   .016  -.127  -.148   .041
        max descent             .072   .090   .053  -.103  -.129   .064

    It peaks about 6 frames before the takeoff fires and it is the largest
    non-duck excursion there is -- larger than the landing compression that
    is the obvious thing to worry about. Gating the duck on "not airborne"
    would not help, because this happens *before* the jump.
    """
    descent = [d for d, _on in descents("draft2.jsonl")]
    for t in TAKEOFFS:
        window = descent[max(0, t - 12):t + 1]
        assert max(window) < duck_mod.DROP, f"takeoff at frame {t}"
