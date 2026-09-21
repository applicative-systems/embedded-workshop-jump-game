"""Detect jumps from a video stream, frame by frame.

Feed frames in with `update()`; it returns a `Jump` on the frame where a jump is
confirmed and `None` otherwise, so `if det.update(frame):` reads naturally.

The signal is the hip height, normalised by the person's bounding-box height so
it is invariant to how far away the player stands, and measured against a
rolling-median baseline so slow drift (walking toward the camera) is not
mistaken for a jump.
"""

import os
import statistics
from collections import deque
from dataclasses import dataclass

# COCO keypoint indices used below.
LEFT_HIP, RIGHT_HIP = 11, 12

# Defaults chosen from a sweep over both test videos and two pose models; each
# sits in the middle of its verified-safe range. See the plan for the grid.
CONF_MIN = 0.50  # safe anywhere in 0.30..0.90
BASE_WIN = 45  # 1.5 s at 30 fps; >= 25 required
RISE = 0.08  # safe in 0.04..0.12, in body heights
FALL = 0.03  # insensitive
CONFIRM = 3  # >= 2 required; 1 produces a false positive
REFRAC = 10  # insensitive


@dataclass
class Jump:
    """A confirmed jump. Truthy, so it can be used directly in an `if`."""

    frame: int
    time_s: float
    height: float  # peak lift so far, in body heights


class JumpDetector:
    """Stateful, one instance per player. Not thread-safe."""

    def __init__(
        self,
        model="yolo11n-pose",
        fps=30.0,
        *,
        yolo=None,
        imgsz=640,
        conf_min=CONF_MIN,
        base_win=BASE_WIN,
        rise=RISE,
        fall=FALL,
        confirm=CONFIRM,
        refrac=REFRAC,
    ):
        self._yolo = yolo
        self._model_name = model
        self.fps = fps
        self.imgsz = imgsz
        self.conf_min = conf_min
        self.rise = rise
        self.fall = fall
        self.confirm = confirm
        self.refrac = refrac

        self._hips = deque(maxlen=base_win)
        self._state = "IDLE"
        self._run = 0
        self._cooldown = 0
        self._peak = 0.0
        self.frame_index = -1

    @property
    def yolo(self):
        """Loaded on first use so the state machine can be tested without a GPU."""
        if self._yolo is None:
            from ultralytics import YOLO

            weights = os.environ.get("POSE_WEIGHTS_DIR", ".")
            self._yolo = YOLO(f"{weights}/{self._model_name}.pt")
        return self._yolo

    def update(self, frame):
        """Run pose on `frame` and advance the detector. Returns `Jump` or None."""
        r = self.yolo.predict(frame, imgsz=self.imgsz, device="cuda", verbose=False)[0]

        dets = []
        for i in range(len(r.boxes)):
            y1, y2 = float(r.boxes.xyxy[i][1]), float(r.boxes.xyxy[i][3])
            k = r.keypoints.xy[i].tolist()
            hip_y = (k[LEFT_HIP][1] + k[RIGHT_HIP][1]) / 2
            dets.append((float(r.boxes.conf[i]), hip_y, y2 - y1))
        return self.update_detections(dets)

    def select(self, dets):
        """Pick the player from one frame's detections, or None.

        Highest confidence above the floor wins. Deliberately NOT
        nearest-to-previous: these videos contain a stationary decoy detection
        that beats the real player on proximity exactly when they jump, which
        is the one moment we must not lose them.
        """
        above = [d for d in dets if d[0] >= self.conf_min]
        return max(above, key=lambda d: d[0]) if above else None

    def update_detections(self, dets):
        """Advance from one frame's `(conf, hip_y, box_h)` detections."""
        picked = self.select(dets)
        if picked is None:
            return self.update_pose(None, None)
        return self.update_pose(picked[1], picked[2])

    def update_pose(self, hip_y, box_h):
        """Advance the state machine from raw numbers. No GPU, no model."""
        self.frame_index += 1

        if hip_y is None or not box_h:
            # Keep state; a dropped frame is not evidence either way.
            if self._cooldown > 0:
                self._cooldown -= 1
            return None

        self._hips.append(hip_y)
        baseline = statistics.median(self._hips)
        lift = (baseline - hip_y) / box_h

        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        if self._state == "IDLE":
            self._run = self._run + 1 if lift > self.rise else 0
            if self._run >= self.confirm:
                self._state = "AIRBORNE"
                self._run = 0
                self._peak = lift
                return Jump(
                    frame=self.frame_index,
                    time_s=self.frame_index / self.fps,
                    height=lift,
                )
        elif self._state == "AIRBORNE":
            self._peak = max(self._peak, lift)
            if lift < self.fall:
                self._state = "IDLE"
                self._cooldown = self.refrac
        return None
