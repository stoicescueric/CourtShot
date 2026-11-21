"""
Run one job on this machine, the way the worker does, and print how long each
stage took and the peak memory.

    cd inference
    uv run python scripts/run_local.py ../videos/final_f.mp4 ../videos/final_f_calibration.json /tmp/courtshot-e2e

The calibration file is one written by the reference tool; its
"labeled_keypoints" are the clicked points. work/ and out/ are made under the
given folder. Exit code 2 means a JobFailure (an input problem).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def peak_footprint_mb() -> float | None:
    """This process's lifetime peak physical footprint in MB, GPU memory
    included (macOS: proc_pid_rusage, RUSAGE_INFO_V4); None elsewhere."""
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    except OSError:
        return None
    info = (ctypes.c_uint64 * 38)()  # struct rusage_info_v4: a 16-byte uuid, then 36 counters
    if libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
        return None
    return info[2 + 28] / 2**20  # ri_lifetime_max_phys_footprint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("calibration", type=Path, help="calibration JSON with labeled_keypoints")
    ap.add_argument("root", type=Path, help="work/ and out/ are made here")
    ap.add_argument("--device", default=None, help="mps, cpu (default: detect)")
    ap.add_argument("--fit-workers", type=int, default=None)
    args = ap.parse_args()

    from courtshot import __version__
    from courtshot.errors import JobFailure
    from courtshot.job import run_job

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    saved = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration = {
        "points": saved["labeled_keypoints"],
        "frame_width": saved["frame_width"],
        "frame_height": saved["frame_height"],
    }
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    spans: dict[str, list[float]] = {}  # stage -> [first call, last call]
    shown: dict[str, int] = {}

    def on_progress(stage: str, done: float) -> None:
        now = time.monotonic()
        spans.setdefault(stage, [now, now])[1] = now
        step = int(done * 10)
        if shown.get(stage) != step:
            shown[stage] = step
            print(f"[progress] {stage} {100 * done:.0f}%", flush=True)

    start = time.monotonic()
    try:
        names = run_job(
            args.video,
            calibration,
            args.root / "work",
            args.root / "out",
            on_progress,
            pipeline_version=f"{__version__}+{commit or 'unknown'}",
            device=args.device,
            fit_workers=args.fit_workers,
        )
    except JobFailure as exc:
        print(f"JobFailure (retryable={exc.retryable}): {exc.message}", file=sys.stderr)
        return 2
    total = time.monotonic() - start

    print("\nstage timings")
    for stage, (a, b) in spans.items():
        print(f"  {stage:<11}{b - a:8.1f} s")
    print(f"  {'total':<11}{total:8.1f} s")
    # macOS reports ru_maxrss in bytes; the children are the 3D-fit processes and ffmpeg
    footprint = peak_footprint_mb()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    print(f"\npeak memory: this process {footprint or 0:.0f} MB footprint ({rss:.0f} MB resident),")
    print(f"             largest child process {child:.0f} MB resident")
    print("\nfiles")
    for name in names:
        print(f"  {name:<16}{(args.root / 'out' / name).stat().st_size / 2**20:9.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
