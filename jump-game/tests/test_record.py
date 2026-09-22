"""The recorder's queueing, with the encoders faked out -- no OpenCV needed."""

import threading

import record
from record import Recorder, load


class FakeWriter:
    def __init__(self, gate=None):
        self.frames = []
        self.gate = gate

    def write(self, frame):
        if self.gate is not None:
            self.gate.wait()
        self.frames.append(frame)

    def release(self):
        pass


def recorder(tmp_path, monkeypatch, gate=None, queue_size=4):
    writers = {}

    def open_writers(self):
        self._writer = writers["game"] = FakeWriter()
        self._raw_writer = writers["raw"] = FakeWriter(gate)

    monkeypatch.setattr(Recorder, "_open_writers", open_writers)
    rec = Recorder(str(tmp_path / "s"), 4, 3, meta={}, raw=True, queue_size=queue_size)
    return rec, writers


def feed(rec, n):
    for i in range(n):
        rec.add(1 / 60, FakeFrame(i), {"f": i, "people": []}, raw=("raw", i))


class FakeFrame:
    def __init__(self, i):
        self.i = i

    def copy(self):
        return ("game", self.i)


def test_both_videos_get_every_frame_when_encoders_keep_up(tmp_path, monkeypatch):
    rec, w = recorder(tmp_path, monkeypatch, queue_size=64)  # holds the whole burst
    feed(rec, record.WARMUP + 50)
    rec.close()
    assert rec.dropped == 0
    assert [i for _, i in w["game"].frames] == [i for _, i in w["raw"].frames]
    assert len(w["game"].frames) == rec.video_frames == 50


def test_a_slow_encoder_drops_from_both_videos_and_marks_the_trace(tmp_path, monkeypatch):
    gate = threading.Event()  # the raw encoder is stuck until this is set
    rec, w = recorder(tmp_path, monkeypatch, gate)
    feed(rec, record.WARMUP + 50)
    gate.set()
    rec.close()

    assert rec.dropped > 0
    assert rec.video_frames + rec.dropped == 50
    # Dropped from both, so the two videos still line up frame for frame.
    kept = [i for _, i in w["game"].frames]
    assert kept == [i for _, i in w["raw"].frames]

    lines = (tmp_path / "s.jsonl").read_text()
    drops = [i for i in range(record.WARMUP, record.WARMUP + 50) if i not in kept]
    assert lines.count('"kind":"drop"') == len(drops) == rec.dropped
    assert all(f'{{"kind":"drop","f":{i}}}' in lines for i in drops)

    # A replay sees poses only, drop lines or not.
    _, frames, summary = load(tmp_path / "s.jsonl")
    assert len(frames) == record.WARMUP + 50
    assert summary["dropped"] == rec.dropped
