"""Both test videos contain exactly 4 jumps."""

import collections
import csv
import os

import pytest

from jump_detector import JumpDetector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = f"{ROOT}/tests/fixtures"

VIDEOS = ["jump-framework-builtin", "jump-logitech"]
MODELS = ["yolo11n-pose", "yolo11s-pose"]
EXPECTED_JUMPS = 4

# Frames where the player is visually at the top of each jump. Detection is
# expected to fire slightly *before* these, never after.
APEXES = {
    "jump-framework-builtin": [22, 74, 128, 176],
    "jump-logitech": [42, 98, 146, 190],
}


def load(video, model):
    """Fixture rows grouped by frame, as (conf, hip_y, box_h) tuples."""
    by_frame = collections.defaultdict(list)
    path = f"{FIXTURES}/{video}-{model}.csv"
    with open(path) as fh:
        for row in csv.DictReader(fh):
            by_frame[int(row["frame"])].append(
                (float(row["conf"]), float(row["hip_y"]), float(row["box_h"]))
            )
    return by_frame


def replay(video, model, **kwargs):
    by_frame = load(video, model)
    det = JumpDetector(**kwargs)
    jumps = []
    for f in range(max(by_frame) + 1):
        if jump := det.update_detections(by_frame.get(f, [])):
            jumps.append(jump)
    return jumps


@pytest.mark.parametrize("video", VIDEOS)
@pytest.mark.parametrize("model", MODELS)
def test_detects_exactly_four_jumps(video, model):
    jumps = replay(video, model)
    assert len(jumps) == EXPECTED_JUMPS


@pytest.mark.parametrize("video", VIDEOS)
@pytest.mark.parametrize("model", MODELS)
def test_fires_just_before_each_apex(video, model):
    """Each jump is reported 0-8 frames ahead of the apex, in order."""
    frames = [j.frame for j in replay(video, model)]
    for fired, apex in zip(frames, APEXES[video]):
        assert 0 <= apex - fired <= 8, f"fired {fired}, apex {apex}"


@pytest.mark.parametrize("video", VIDEOS)
@pytest.mark.parametrize("model", MODELS)
def test_jump_metadata_is_sane(video, model):
    for jump in replay(video, model):
        assert jump.height > 0
        assert jump.time_s == pytest.approx(jump.frame / 30.0)


def test_logitech_survives_the_stationary_decoy():
    """Regression: nearest-to-previous selection scored 3/4 here.

    jump-logitech carries a decoy detection parked near hip_y 650 that beats
    the real player on proximity mid-jump. Confidence-based selection must not
    be replaced by proximity tracking.
    """
    assert len(replay("jump-logitech", "yolo11s-pose")) == EXPECTED_JUMPS


def test_low_confidence_junk_is_never_selected():
    """Junk detections sit far from the body; the floor must exclude them."""
    det = JumpDetector()
    junk = (0.32, 210.0, 800.0)
    real = (0.93, 640.0, 850.0)
    assert det.select([junk]) is None
    assert det.select([junk, real]) == real
    # A junk box can outrank the floor but must still lose on confidence.
    assert det.select([(0.58, 226.0, 800.0), real]) == real


def test_no_detections_yields_no_jump():
    det = JumpDetector()
    assert det.update_detections([]) is None
    assert det.update_pose(None, None) is None


def test_slow_drift_is_not_a_jump():
    """Walking toward the camera moves the hip steadily without a jump."""
    det = JumpDetector()
    jumps = [det.update_pose(600 + i, 850) for i in range(120)]
    assert not any(jumps)
    det = JumpDetector()
    jumps = [det.update_pose(600 - i * 0.5, 850) for i in range(120)]
    assert not any(jumps)


@pytest.mark.gpu
@pytest.mark.parametrize("video", VIDEOS)
def test_end_to_end_on_real_video(video):
    """Full frame -> keypoint -> jump path. Needs an NVIDIA GPU."""
    import cv2

    det = JumpDetector()
    cap = cv2.VideoCapture(f"{ROOT}/test_inputs/{video}.webm")
    assert cap.isOpened()
    jumps = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if jump := det.update(frame):
            jumps.append(jump)
    cap.release()
    assert len(jumps) == EXPECTED_JUMPS
