"""Prepare the downloaded video for the pipeline (design 11.2).

The pipeline assumes time = frame / fps and reads stored pixels, so a video with a variable frame rate,
a rotation flag, a start offset or non-square pixels is first copied, corrected, at high quality.
Any other video is used untouched: re-encoding costs the ball detector a little.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

from courtshot.errors import JobFailure

logger = logging.getLogger(__name__)

TIMING_WINDOW_S = 10  # packet timestamps are judged over the first seconds
# ponytail: a frame interval more than 10 % (and 1.1 ms, a 1 ms timebase's rounding) off the typical one is
# variable frame rate. One dropped frame counts: over 15 minutes such drops add up to seconds.
INTERVAL_TOLERANCE = 0.10
INTERVAL_FLOOR_S = 0.0011
STANDARD_RATES = tuple(
    Fraction(r) for r in ("24000/1001", "24", "25", "30000/1001", "30", "50", "60000/1001", "60", "100", "120", "240")
)
FFPROBE_TIMEOUT_S = 120


@dataclass(frozen=True)
class VideoInfo:
    width: int  # stored pixels, what OpenCV decodes
    height: int
    duration_s: float
    start_s: float
    rotation: int  # degrees, 0..359
    sar: Fraction  # sample (pixel) aspect ratio
    rate: Fraction  # frames per second: the nominal rate, or the measured one when variable
    variable_rate: bool


def prepare_video(
    source: Path,
    prepared: Path,
    calibration: Mapping[str, Any],
    limits: Mapping[str, Any],
    *,
    ffmpeg: str,
    ffprobe: str,
    on_progress: Callable[[float], None],
) -> Path:
    """Return the video the pipeline should read: `source`, or the corrected copy at `prepared`.

    Raises JobFailure (not retryable) for a length outside the limits, an unreadable video, or frames
    whose size differs from the size the court was marked on.
    """
    info = probe(ffprobe, source)
    low, high = limits["min_video_seconds"], limits["max_video_seconds"]
    if not low <= info.duration_s <= high:
        raise JobFailure(f"the video is {info.duration_s:.1f} s long; the limits are {low} s to {high} s")
    reasons = corrections(info)
    video = source
    if reasons:
        logger.info("correcting the video: %s", ", ".join(reasons))
        if not prepared.exists():
            write_corrected(ffmpeg, source, prepared, info, on_progress)
        video, info = prepared, probe(ffprobe, prepared)
    on_progress(1.0)
    expected = (int(calibration["frame_width"]), int(calibration["frame_height"]))
    if (info.width, info.height) != expected:
        raise JobFailure(
            f"the video's frames are {info.width}x{info.height} pixels but the court was marked "
            f"on a {expected[0]}x{expected[1]} frame"
        )
    return video


def probe(ffprobe: str, video: Path) -> VideoInfo:
    data = _ffprobe_json(ffprobe, video, "-show_streams", "-show_format")
    if not data.get("streams"):
        raise JobFailure("the file has no video stream")
    stream, container = data["streams"][0], data.get("format", {})
    duration = _number(container.get("duration")) or _number(stream.get("duration"))
    if not duration or not stream.get("width") or not stream.get("height"):
        raise JobFailure("the video's length or frame size cannot be read")
    nominal = _fraction(stream.get("r_frame_rate")) or _fraction(stream.get("avg_frame_rate")) or Fraction(25)
    interval, variable = _frame_timing(_packet_times(ffprobe, video))
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration_s=duration,
        start_s=_number(stream.get("start_time")) or 0.0,
        rotation=_rotation(stream),
        sar=_fraction(stream.get("sample_aspect_ratio"), sep=":") or Fraction(1),
        rate=_standard_rate(1 / interval) if variable and interval else nominal,
        variable_rate=variable,
    )


def corrections(info: VideoInfo) -> list[str]:
    """Why the video needs a corrected copy; empty when it can be used as it is."""
    reasons = []
    if info.variable_rate:
        reasons.append("variable frame rate")
    if info.rotation:
        reasons.append(f"rotated {info.rotation} degrees")
    if abs(info.start_s) > 0.5 / info.rate:  # less than half a frame changes no frame's time
        reasons.append(f"starts at {info.start_s:.3f} s")
    if info.sar != 1:
        reasons.append(f"non-square pixels ({info.sar.numerator}:{info.sar.denominator})")
    return reasons


def write_corrected(
    ffmpeg: str, source: Path, dest: Path, info: VideoInfo, on_progress: Callable[[float], None]
) -> None:
    """Constant frame rate, upright (ffmpeg applies the rotation), start at 0, square pixels.

    Non-square pixels are stretched the way browsers size them: the shorter side grows. 4:4:4 chroma and
    no size rounding, so the frame size equals what the browser showed when the court was marked.
    """
    tmp = dest.with_name(dest.stem + ".tmp" + dest.suffix)
    command = [ffmpeg, "-nostdin", "-y", "-v", "error", "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn"]
    if info.sar != 1:
        command += ["-vf", "scale='if(gte(sar,1),round(iw*sar),iw)':'if(gte(sar,1),ih,round(ih/sar))',setsar=1"]
    command += [
        "-fps_mode", "cfr", "-r", str(info.rate),
        "-c:v", "libx264", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv444p", "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats", str(tmp),
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:  # stderr: the job log
        assert process.stdout is not None
        for line in process.stdout:
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and value.isdigit():
                on_progress(int(value) / 1e6 / info.duration_s)
    if process.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not write the corrected copy (exit code {process.returncode})")
    os.replace(tmp, dest)


def _ffprobe_json(ffprobe: str, video: Path, *entries: str) -> dict[str, Any]:
    command = [ffprobe, "-v", "error", "-select_streams", "v:0", *entries, "-of", "json", str(video)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_S)
    if result.returncode != 0:
        raise JobFailure(f"the video cannot be read: {result.stderr.strip()[:200]}")
    data: dict[str, Any] = json.loads(result.stdout or "{}")
    return data


def _packet_times(ffprobe: str, video: Path) -> list[float]:
    """Presentation times of the video packets in the first seconds, in display order."""
    data = _ffprobe_json(ffprobe, video, "-read_intervals", f"%+{TIMING_WINDOW_S}", "-show_entries", "packet=pts_time")
    times = (_number(packet.get("pts_time")) for packet in data.get("packets", []))
    return sorted(t for t in times if t is not None)


def _frame_timing(times: list[float]) -> tuple[float | None, bool]:
    """(typical frame interval, whether any interval strays from it)."""
    intervals = [b - a for a, b in pairwise(times)]
    if len(intervals) < 2:
        return None, False
    typical = statistics.median(intervals)
    tolerance = max(INTERVAL_TOLERANCE * typical, INTERVAL_FLOOR_S)
    return typical, any(abs(i - typical) > tolerance for i in intervals)


def _standard_rate(measured: float) -> Fraction:
    nearest = min(STANDARD_RATES, key=lambda rate: abs(rate - Fraction(measured)))
    if abs(nearest - Fraction(measured)) <= 0.01 * nearest:
        return nearest
    return Fraction(measured).limit_denominator(1001)


def _rotation(stream: Mapping[str, Any]) -> int:
    values = [stream.get("tags", {}).get("rotate")]
    values += [entry.get("rotation") for entry in stream.get("side_data_list", [])]
    for value in values:
        if (number := _number(value)) is not None and round(number) % 360:
            return round(number) % 360
    return 0


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fraction(value: Any, sep: str = "/") -> Fraction | None:
    """'30000/1001' or '4:3'; None for missing, zero or malformed values ('0/0', '0:1', 'N/A')."""
    numerator, _, denominator = str(value).partition(sep)
    try:
        fraction = Fraction(int(numerator), int(denominator))
    except (ValueError, ZeroDivisionError):
        return None
    return fraction or None
