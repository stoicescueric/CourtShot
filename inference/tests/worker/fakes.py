"""Fake pipeline runners with the signature of courtshot.job.run_job, named in job.json's "runner"."""

from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from courtshot.errors import JobFailure

Progress = Callable[[str, float], None]


def result_files(report: dict[str, Any]) -> dict[str, bytes]:
    """What run_ok writes: deterministic bytes, so a test can compare what the server received."""
    rng = random.Random(1)
    return {
        "report.json": json.dumps(report).encode(),
        "summary.json": b"{}",
        "positions.csv": b"frame,player,x,y\n0,P1,1.0,2.0\n",
        "tracked.mp4": rng.randbytes(300_000),
        "plain.mp4": rng.randbytes(150_000),
        "scene.json": b'{"schema_version": 1}',
        "notes.txt": b"not a result file",
    }


def run_ok(video: Path, calibration: dict[str, Any], work_dir: Path, out_dir: Path, on_progress: Progress,
           **options: Any) -> list[str]:
    for stage in ("tracking", "appearance", "analysis", "ball3d", "render"):
        on_progress(stage, 0.5)
        on_progress(stage, 1.0)
    print("fake pipeline ran", flush=True)
    report = {"video": video.name, "calibration": calibration, "work_dir": str(work_dir),
              "options": {key: None if value is None else str(value) for key, value in options.items()}}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result_files(report).items():
        (out_dir / name).write_bytes(data)
    return sorted(result_files(report))


def run_user_error(video: Path, calibration: dict[str, Any], work_dir: Path, out_dir: Path, on_progress: Progress,
                   **options: Any) -> list[str]:
    print("checking the calibration", flush=True)
    raise JobFailure("the court calibration is mirrored")


def run_crash(video: Path, calibration: dict[str, Any], work_dir: Path, out_dir: Path, on_progress: Progress,
              **options: Any) -> list[str]:
    on_progress("tracking", 0.2)
    print("about to crash", flush=True)
    raise RuntimeError("boom")


IGNORE_SIGTERM = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"


def run_killed(video: Path, calibration: dict[str, Any], work_dir: Path, out_dir: Path, on_progress: Progress,
               **options: Any) -> list[str]:
    """SIGKILLed, like the memory pressure killer does, with a child left behind like a 3D-fit pool process."""
    child = subprocess.Popen([sys.executable, "-c", IGNORE_SIGTERM])
    (work_dir / "pids.json").write_text(json.dumps({"job": os.getpid(), "child": child.pid}))
    print("using too much memory", flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
    raise AssertionError("not reached")


def run_hang(video: Path, calibration: dict[str, Any], work_dir: Path, out_dir: Path, on_progress: Progress,
             **options: Any) -> list[str]:
    """Never ends; leaves a grandchild that ignores SIGTERM, like a stuck pool process or ffmpeg."""
    child = subprocess.Popen([sys.executable, "-c", IGNORE_SIGTERM])
    on_progress("tracking", 0.3)
    (work_dir / "pids.json").write_text(json.dumps({"job": os.getpid(), "child": child.pid}))
    time.sleep(600)
    return []
