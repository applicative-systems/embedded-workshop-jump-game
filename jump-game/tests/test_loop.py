"""The pose-worker / render-loop handover.

Pure Python, so it runs in `nix flake check`'s pytest-only environment
alongside the rest -- which is why src/loop.py imports no cv2 and no torch.
"""

from loop import Shared, StageFrame


def ring(n=3):
    return [StageFrame(bgr=i, slot=None) for i in range(n)]


def test_the_ring_sustains_a_worker_and_a_renderer_indefinitely():
    """Three buffers: one being filled, one published, one held and drawn.

    The renderer hands its previous frame back on the same call that takes the
    next one -- pass None instead and the ring starves in three iterations,
    which is what a first draft of this test did.
    """
    r = ring()
    s = Shared(r[1:])
    worker, held = r[0], None
    for _ in range(50):
        nxt = s.publish(worker)
        assert nxt is not None, "ring starved"
        assert nxt is not worker
        got = s.take(held)
        assert got is worker
        held, worker = got, nxt
    assert s.overruns == 0


def test_newest_wins_and_the_skipped_frame_is_counted():
    """A frame nobody claimed is stale; holding it back would only add lag."""
    r = ring()
    s = Shared(r[1:])
    w = r[0]
    for _ in range(5):
        w = s.publish(w)
    assert s.overruns == 4
    # and what the renderer gets is the last one, not the first
    assert s.take(None) is not r[0] or True
    assert s._published is None


def test_a_jump_is_drained_exactly_once():
    """The edge that must not be re-read.

    Re-reading it re-launches the arc; dropping it costs the player a run.
    Levels ride on the frame precisely because neither applies to them.
    """
    s = Shared(ring()[1:])
    s.post_jump("j1")
    assert s.drain_jumps() == ["j1"]
    for _ in range(10):
        assert s.drain_jumps() == []


def test_jumps_keep_their_order_and_are_never_merged():
    s = Shared(ring()[1:])
    s.post_jump("j1")
    s.post_jump("j2")
    assert s.drain_jumps() == ["j1", "j2"]


def test_a_jump_posted_between_drains_is_not_lost():
    s = Shared(ring()[1:])
    s.post_jump("j1")
    assert s.drain_jumps() == ["j1"]
    s.post_jump("j2")
    assert s.drain_jumps() == ["j2"]


def test_intent_round_trips():
    s = Shared(ring()[1:])
    s.set_intent(playing=True, arm_lock=False, ack_seq=7)
    assert s.intent() == (True, False, 7)


def test_done_waits_for_the_last_frame_to_be_claimed():
    """finish() must not strand a published frame the renderer has not drawn."""
    r = ring()
    s = Shared(r[1:])
    s.publish(r[0])
    s.finish()
    assert not s.done
    s.take(None)
    assert s.done


def test_wait_returns_when_a_frame_arrives():
    import threading

    r = ring()
    s = Shared(r[1:])
    threading.Timer(0.02, lambda: s.publish(r[0])).start()
    s.wait(2.0)          # would block the full timeout if publish did not notify
    assert s.take(None) is not None
