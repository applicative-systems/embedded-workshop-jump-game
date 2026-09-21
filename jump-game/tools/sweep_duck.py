#!/usr/bin/env python3
"""Tune the duck detector against recorded sessions.

Replays `--record` traces (`NAME.jsonl`) through the *real* TakeoffDetector,
parameterised through its duck_* kwargs -- deliberately not through a copy of
the state machine, because a sweep that measures a reimplementation is how a
tuned constant rots.

    python tools/sweep_duck.py duck1.jsonl duck2.jsonl     # the grid
    python tools/sweep_duck.py --label duck1.jsonl         # propose intervals
    python tools/sweep_duck.py --dump duck1.jsonl          # per-frame descent

Ground truth comes from the recording protocol rather than frame-by-frame
labelling: each block of the session contains a known number of ducks, so
--label proposes intervals at a permissive threshold and the human's job is to
confirm the count per block and trim. It writes NAME.labels.json next to the
trace; the grid reads it if it is there. A trace with no labels file still
contributes its false positives, which is what makes a jumping-only session
like tests/fixtures/draft2.jsonl useful here.

No GPU, no model, no camera -- everything runs off the raw keypoints in the
trace.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import duck as duck_mod  # noqa: E402
from record import RecordedPose, load  # noqa: E402
from takeoff import TakeoffDetector  # noqa: E402

# Permissive on purpose: --label is proposing candidates for a human to trim,
# so it should over-offer rather than hide a duck the sweep will then score as
# a miss.
LABEL_DROP = 0.10
LABEL_RUN = 3


def trace_fps(path):
    _header, _frames, summary = load(path)
    return (summary or {}).get("fps") or 30.0


def descents(path, **kw):
    """Per-frame (descent, ducking) for one trace."""
    _header, frames, _summary = load(path)
    det = TakeoffDetector(fps=trace_fps(path), **kw)
    out = []
    for e in frames:
        det.update(RecordedPose(e))
        out.append((det.descent, det.ducking))
    return out


def intervals(flags):
    """[start, end] runs of True in a bool sequence."""
    runs, start = [], None
    for i, on in enumerate(flags):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i - 1])
            start = None
    if start is not None:
        runs.append([start, len(flags) - 1])
    return runs


def labels_for(path):
    lp = os.path.splitext(path)[0] + ".labels.json"
    return json.load(open(lp)) if os.path.exists(lp) else None


def do_label(path):
    d = [x for x, _ in descents(path)]
    flags = [x > LABEL_DROP for x in d]
    runs = [r for r in intervals(flags) if r[1] - r[0] + 1 >= LABEL_RUN]
    fps = trace_fps(path)
    print(f"{path}: {len(runs)} candidate ducks at descent > {LABEL_DROP}")
    for a, b in runs:
        peak = max(d[a:b + 1])
        print(f"  [{a:5d}, {b:5d}]  {a / fps:6.2f}s  {(b - a + 1) / fps:5.2f}s"
              f"  peak {peak:.3f}")
    out = os.path.splitext(path)[0] + ".labels.json"
    print(f"\nconfirm the count per block, trim, and save as {out}:")
    print(json.dumps(runs))


def do_dump(path):
    d = descents(path)
    print("frame\tdescent\tducking")
    for i, (x, on) in enumerate(d):
        print(f"{i}\t{x:.4f}\t{int(on)}")


def score(paths, drop, rise_back, confirm):
    """found / missed / false / latency / hold fidelity / early exits."""
    found = missed = false = early = 0
    lat, fid = [], []
    for path in paths:
        flags = [on for _, on in descents(
            path, duck_drop=drop, duck_rise_back=rise_back,
            duck_confirm=confirm)]
        got = intervals(flags)
        truth = labels_for(path)
        if truth is None:
            false += len(got)
            continue
        used = set()
        for a, b in truth:
            hits = [i for i, (ga, gb) in enumerate(got)
                    if ga <= b and gb >= a]
            if not hits:
                missed += 1
                continue
            found += 1
            used.update(hits)
            if len(hits) > 1:
                early += len(hits) - 1
            lat.append(got[hits[0]][0] - a)
            covered = sum(min(b, got[i][1]) - max(a, got[i][0]) + 1
                          for i in hits)
            fid.append(covered / (b - a + 1))
        false += len(got) - len(used)
    mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    return found, missed, false, mean(lat), mean(fid), early


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traces", nargs="+")
    p.add_argument("--label", action="store_true",
                   help="propose duck intervals for a human to confirm")
    p.add_argument("--dump", action="store_true",
                   help="per-frame descent, for plotting")
    a = p.parse_args()

    if a.label or a.dump:
        for path in a.traces:
            (do_label if a.label else do_dump)(path)
        return

    for path in a.traces:
        d = [x for x, _ in descents(path)]
        lab = labels_for(path)
        print(f"{path}: {len(d)} frames, "
              f"{'no labels (false positives only)' if lab is None else str(len(lab)) + ' labelled ducks'}"
              f", peak descent {max(d):.3f}")
    print()

    print(f"{'drop':>5} {'conf':>4} {'back':>5} "
          f"{'found':>5} {'miss':>4} {'false':>5} {'lat':>6} {'hold':>5} {'early':>5}")
    for rise_back in (0.04, 0.06, 0.08):
        for confirm in (1, 2, 3, 4):
            for i in range(8, 21):
                drop = i / 100
                f, m, fp, lat, fid, e = score(a.traces, drop, rise_back, confirm)
                print(f"{drop:5.2f} {confirm:4d} {rise_back:5.2f} "
                      f"{f:5d} {m:4d} {fp:5d} {lat:6.2f} {fid:5.2f} {e:5d}")


if __name__ == "__main__":
    main()
