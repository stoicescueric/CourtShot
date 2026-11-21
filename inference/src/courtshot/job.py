"""
One job: a prepared video and the clicked court points in, the result files out.

    names = run_job(video, calibration, work_dir, out_dir, on_progress, pipeline_version="0.2.0+1a2b3c4")

Input problems the user has to fix raise JobFailure(retryable=False): an
unusable calibration, a frame size other than the one the court was marked
on, four players never on court together, players on court too briefly. Any
other exception is a crash, and a retry may succeed; it finds the caches of
the finished stages in work_dir.

Nothing heavy is imported at import time: on macOS every process of the 3D-fit
pool re-imports the main module of the job's process, and loading PyTorch in
each of them would exhaust a small Mac.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from courtshot.errors import JobFailure

STAGES = ("tracking", "appearance", "analysis", "ball3d", "render")
SCHEMA_VERSION = 1
RESULT_FILES = (
    "report.json",
    "scene.json",
    "tracked.mp4",
    "plain.mp4",
    "summary.json",
    "positions.csv",
    "shots.csv",
    "ball_events.csv",
)


def default_fit_workers() -> int:
    """Processes for the 3D fit: at most 2, and at most half the CPUs."""
    return max(1, min(2, (os.cpu_count() or 2) // 2))


def run_job(
    video: Path,
    calibration: dict,
    work_dir: Path,
    out_dir: Path,
    on_progress: Callable[[str, float], None],
    *,
    pipeline_version: str,
    device: str | None = None,
    fit_workers: int | None = None,
    max_error_m: float = 0.25,
    max_height: int = 720,
    ball_model: Path | None = None,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
) -> list[str]:
    """Run the pipeline on one prepared video and write the result files.

    video:        constant frame rate, upright, start time 0, square pixels
    calibration:  {"points": {name: [x, y]}, "frame_width": int, "frame_height": int}
    work_dir:     the match's work folder; caches go here and survive retries
    out_dir:      the result files go here (created if missing)
    on_progress:  (stage in STAGES, share of that stage done, 0..1)
    ffmpeg, ffprobe: the commands for the videos; None means $COURTSHOT_FFMPEG /
                  $COURTSHOT_FFPROBE, else the command on PATH

    Returns the names of the files written to out_dir.
    """
    from courtshot.court import checked_calibration
    from courtshot.pipeline import CourtShot
    from courtshot.report.match_report import report_payload, write_data_files
    from courtshot.report.scene3d import scene_payload

    cal = checked_calibration(
        calibration["points"], int(calibration["frame_width"]), int(calibration["frame_height"]), max_error_m
    )
    width, height = decoded_frame_size(video)
    if (width, height) != (cal.frame_width, cal.frame_height):
        raise JobFailure(
            f"The video is {width}x{height} but the court was marked on a "
            f"{cal.frame_width}x{cal.frame_height} frame. Redo the calibration."
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in RESULT_FILES:  # nothing left over from an earlier attempt
        (out / name).unlink(missing_ok=True)
        (out / f"{name}.tmp").unlink(missing_ok=True)

    shot = CourtShot(
        video,
        cal,
        work_dir,
        ball_model_path=ball_model,
        device=device,
        on_progress=on_progress,
        fit_workers=fit_workers or default_fit_workers(),
    )
    analysis = shot.analyse()
    report = report_payload(analysis)
    scene = scene_payload(analysis)
    names = shot.render(analysis, out, max_height, ffmpeg=ffmpeg, ffprobe=ffprobe)
    names += write_data_files(out, report)
    write_json(
        out / "report.json",
        {**report.payload, "schema_version": SCHEMA_VERSION, "pipeline_version": pipeline_version},
    )
    names.append("report.json")
    if scene is not None:
        write_json(out / "scene.json", {**scene, "schema_version": SCHEMA_VERSION})
        names.append("scene.json")
    return names


def decoded_frame_size(video: Path) -> tuple[int, int]:
    """(width, height) of the first frame as OpenCV decodes it."""
    from courtshot.io.video import VideoReader

    with VideoReader(video) as reader:
        frame = reader.read_frame(0)
    if frame is None:
        raise RuntimeError(f"Cannot decode the first frame of {video}")
    return frame.shape[1], frame.shape[0]


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")
