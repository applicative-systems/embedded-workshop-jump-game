"""The handover between the pose worker and the render loop.

Deliberately free of cv2, torch and numpy at import time, for the reason
record.py's lazy cv2 import documents: `nix flake check` builds a python with
pytest and nothing else, and everything in here is pure enough to test there.

The whole design is one distinction. Most of what the renderer needs from a
camera frame is a **level** -- the pose, the softened backdrop, the camera
panel, whether the player is crouching, whether they are in shot. Reading a
level twice, or skipping one, changes nothing, because JumpGame integrates
levels against seconds and counts no frames.

A confirmed takeoff is an **edge**. Reading it twice re-launches the arc;
dropping it costs the player a run. So it does not ride on the frame at all --
it goes in a mailbox that is drained exactly once.
"""

import threading
from collections import deque
from dataclasses import dataclass, field


@dataclass
class StageFrame:
    """One camera frame's worth of everything the renderer needs from it.

    `bgr` is the whole stage with the backdrop already softened into it and
    the letterbox bars already black -- the renderer copies it and draws the
    game on top. `panel` is the camera inset with its skeleton already on it.
    Neither is ever copied across the thread boundary; ownership of a
    preallocated buffer moves instead.
    """

    bgr: object
    slot: object  # a view into bgr: where the camera's softened room goes
    panel: object = None
    seq: int = 0
    pose: object = None
    selected: object = None
    ducking: bool = False
    present: bool = False
    descent: float = 0.0
    using: str = ""


@dataclass
class Intent:
    """What the renderer wants the detector to do, sampled by the worker."""

    playing: bool = False
    arm_lock: bool = False
    ack_seq: int = -1


class Shared:
    """One lock, one condition, one ring of stage buffers.

    Newest-wins on the frame, exactly as ThreadedCapture and ThreadedDisplay
    already do: a published frame nobody claimed is stale by definition, and
    holding it back would only add latency. Edges are the exception and are
    queued, not overwritten.
    """

    def __init__(self, buffers):
        self._cv = threading.Condition()
        self._free = list(buffers)
        self._published = None
        self._jumps = deque()
        self._intent = Intent()
        self._done = False
        self.overruns = 0  # published frames the renderer never saw

    # -- worker side -------------------------------------------------------

    def acquire(self, timeout=1.0):
        """A buffer to write the next camera frame into, or None."""
        with self._cv:
            if not self._free:
                self._cv.wait_for(lambda: self._free or self._done, timeout)
            return self._free.pop() if self._free else None

    def publish(self, sf):
        """Hand `sf` to the renderer and take back a buffer to fill next."""
        with self._cv:
            if self._published is not None:
                # Never claimed. Straight back to the free list -- the
                # renderer wants the newest room, not a queue of old ones.
                self._free.append(self._published)
                self.overruns += 1
            self._published = sf
            self._cv.notify_all()
            return self._free.pop() if self._free else None

    def post_jump(self, jump):
        with self._cv:
            self._jumps.append(jump)
            self._cv.notify_all()

    def intent(self):
        with self._cv:
            i = self._intent
            return i.playing, i.arm_lock, i.ack_seq

    def finish(self):
        with self._cv:
            self._done = True
            self._cv.notify_all()

    # -- render side -------------------------------------------------------

    def take(self, held):
        """The newest published frame, or None. Recycles `held` if one came."""
        with self._cv:
            sf = self._published
            if sf is None:
                return None
            self._published = None
            if held is not None:
                self._free.append(held)
            self._cv.notify_all()
            return sf

    def drain_jumps(self):
        """Every takeoff since the last call, in order. Never the same twice."""
        with self._cv:
            out = list(self._jumps)
            self._jumps.clear()
            return out

    def set_intent(self, playing, arm_lock, ack_seq):
        with self._cv:
            self._intent = Intent(playing, arm_lock, ack_seq)

    def wait(self, timeout):
        """Sleep until a frame is published, the worker finishes, or timeout.

        Waking on publish is what keeps the split latency-neutral: a pose that
        lands mid-interval is drawn at once rather than at the next deadline.
        """
        if timeout <= 0:
            return
        with self._cv:
            self._cv.wait_for(lambda: self._published is not None or self._done,
                              timeout)

    @property
    def done(self):
        with self._cv:
            return self._done and self._published is None
