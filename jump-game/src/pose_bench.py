"""Play a video frame by frame, paint the detected skeleton, and benchmark it.

Runs one pass per model and writes one annotated video per model, so frame
rates and detection quality can be compared side by side. Detection is
GPU-only: without a usable NVIDIA GPU this crashes, by design.

--export compiles --models to TensorRT .engine files instead (no video
needed); --engine then benchmarks those instead of the .pt weights.
"""

import argparse
import grp
import os
import pwd
import shutil
import sys
import time

import cv2
import torch
from ultralytics import YOLO

WEIGHTS_DIR = os.environ.get("POSE_WEIGHTS_DIR", ".")
# Compiled TensorRT engines are locked to one GPU/TensorRT-version/precision/
# imgsz combo and have to be built on the device that will run them, so
# unlike WEIGHTS_DIR this must be writable -- default it to cwd rather than
# the (read-only, Nix store) WEIGHTS_DIR.
ENGINE_DIR = os.environ.get("POSE_ENGINE_DIR", ".")

# Device nodes CUDA needs on Jetson; missing group membership on these (not a
# driver problem) is the usual cause of "device=0 not available" there.
JETSON_DEVICE_NODES = [
    "/dev/nvmap",
    "/dev/nvhost-ctrl",
    "/dev/nvhost-gpu",
    "/dev/nvhost-as-gpu",
    "/dev/nvhost-vic",
]


def print_system_info():
    """Print user/group and CUDA visibility, so permission issues (e.g. the
    Jetson user not being in the 'video' group) show up before the cryptic
    ultralytics traceback does."""
    user = pwd.getpwuid(os.getuid()).pw_name
    primary_group = grp.getgrgid(pwd.getpwuid(os.getuid()).pw_gid).gr_name
    groups = sorted({primary_group, *(g.gr_name for g in grp.getgrall() if user in g.gr_mem)})
    print(f"user={user} groups={groups}", flush=True)
    print(f"python={sys.version.split()[0]} torch={torch.__version__}", flush=True)
    print(
        f"cuda available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    for dev in JETSON_DEVICE_NODES:
        if os.path.exists(dev):
            print(f"{dev}: read={os.access(dev, os.R_OK)} write={os.access(dev, os.W_OK)}", flush=True)
    print(flush=True)


def export_engine(name, imgsz):
    """Compile `name`'s .pt weights into a TensorRT .engine in ENGINE_DIR.

    Engines are baked for one exact GPU/TensorRT-version/precision/imgsz --
    re-export after any of those change. ultralytics writes the .engine next
    to the .pt it's exporting, and WEIGHTS_DIR is a read-only Nix store path,
    so copy the weights into (writable) ENGINE_DIR first.
    """
    os.makedirs(ENGINE_DIR, exist_ok=True)
    pt_copy = f"{ENGINE_DIR}/{name}.pt"
    shutil.copyfile(f"{WEIGHTS_DIR}/{name}.pt", pt_copy)
    YOLO(pt_copy).export(format="engine", half=True, imgsz=imgsz, device="cuda")
    os.remove(pt_copy)


def run(video, name, outdir, imgsz, limit, engine):
    """Annotate `video` with `name`'s skeleton; return per-frame stats."""
    if engine:
        model = YOLO(f"{ENGINE_DIR}/{name}.engine")
    else:
        model = YOLO(f"{WEIGHTS_DIR}/{name}.pt")
    # half=True only makes sense for the .pt/eager-CUDA path -- an exported
    # engine already has its precision baked in.
    predict_kwargs = dict(imgsz=imgsz, device="cuda", verbose=False)
    if not engine:
        predict_kwargs["half"] = True

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(outdir, exist_ok=True)
    out = cv2.VideoWriter(
        f"{outdir}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    # Warm up: the first inference pays CUDA context init and model upload,
    # which would otherwise dominate the average.
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"{video} has no frames")
    model.predict(frame, **predict_kwargs)
    # Reopen rather than seek: seeking is unreliable in VP8/WebM.
    cap.release()
    cap = cv2.VideoCapture(video)

    # infer_times covers only model.predict(), so a faster backend (e.g.
    # TensorRT) shows up here even when total ms/frame does not move --
    # r.plot() is CPU-side skeleton drawing and costs the same regardless of
    # backend, so it can dominate total time and hide an inference speedup.
    times, infer_times, confs, detected = [], [], [], 0
    wall = time.perf_counter()
    while limit is None or len(times) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        r = model.predict(frame, **predict_kwargs)[0]
        # CUDA is async; without this the timings are meaninglessly low.
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        annotated = r.plot()
        times.append(time.perf_counter() - t0)
        infer_times.append(t1 - t0)

        out.write(annotated)
        if len(r.boxes):
            detected += 1
            confs.append(float(r.boxes.conf.mean()))
    wall = time.perf_counter() - wall

    cap.release()
    out.release()

    avg = sum(times) / len(times)
    avg_infer = sum(infer_times) / len(infer_times)
    return {
        "model": name,
        "frames": len(times),
        "avg_ms": avg * 1000,
        "infer_ms": avg_infer * 1000,
        "fps": 1 / avg,
        # What jump_detector.py would actually see: it never calls r.plot(),
        # so total FPS above (which includes that CPU-side drawing cost) is
        # not what real gameplay throughput looks like -- this is.
        "infer_fps": 1 / avg_infer,
        "wall_s": wall,
        "conf": sum(confs) / len(confs) if confs else 0.0,
        "det_rate": detected / len(times),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", nargs="?", help="input video; not needed with --export")
    # Comma-separated rather than nargs="+": a greedy list would swallow the
    # video positional, so `--models a,b,c video.webm` would not parse.
    p.add_argument(
        "--models",
        default="yolo11n-pose",
        help="comma-separated, e.g. yolo11n-pose,yolo11s-pose",
    )
    p.add_argument("--outdir", default="out")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--limit", type=int, help="stop after N frames")
    p.add_argument(
        "--export",
        action="store_true",
        help="compile --models to TensorRT .engine files in POSE_ENGINE_DIR instead of benchmarking",
    )
    p.add_argument(
        "--engine",
        action="store_true",
        help="benchmark the TensorRT .engine (from POSE_ENGINE_DIR) instead of the .pt weights",
    )
    a = p.parse_args()

    models = [m for m in a.models.split(",") if m]

    if a.export:
        for name in models:
            print(f"exporting {name} -> {ENGINE_DIR}/{name}.engine ...", flush=True)
            export_engine(name, a.imgsz)
        return

    if not a.video:
        p.error("VIDEO is required unless --export is given")

    print_system_info()

    stats = []
    for name in models:
        print(f"running {name} ...", flush=True)
        s = run(a.video, name, a.outdir, a.imgsz, a.limit, a.engine)
        print(f"  -> {a.outdir}/{name}.mp4", flush=True)
        stats.append(s)

    print(f"\n{'model':<16}{'frames':>7}{'ms/frame':>10}{'infer ms':>10}{'FPS':>8}"
          f"{'infer FPS':>10}{'conf':>7}{'det%':>7}{'wall s':>8}")
    for s in stats:
        print(f"{s['model']:<16}{s['frames']:>7}{s['avg_ms']:>10.2f}"
              f"{s['infer_ms']:>10.2f}{s['fps']:>8.1f}{s['infer_fps']:>10.1f}"
              f"{s['conf']:>7.2f}{s['det_rate'] * 100:>7.0f}{s['wall_s']:>8.1f}")


if __name__ == "__main__":
    main()
