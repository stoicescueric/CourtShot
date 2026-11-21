"""
Project: CourtShot
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: video.py
Description:
    Video I/O utilities.
    VideoReader reads video files and yields frames with timestamps.
    VideoWriter writes frames to a browser-playable H.264 MP4 with ffmpeg.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Reads video files and yields frames with metadata."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

    def frames(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
        step: int = 1,
    ) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """Yield (frame_id, timestamp_ms, frame) tuples.

        Args:
            start_frame: First frame to read.
            end_frame: Last frame (exclusive). None = all frames.
            step: Read every Nth frame (for sampling).
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_id = start_frame
        end = end_frame or self.frame_count

        while frame_id < end:
            ret, frame = self._cap.read()
            if not ret:
                break

            if (frame_id - start_frame) % step == 0:
                timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                yield frame_id, timestamp_ms, frame

            frame_id += 1

    def read_frame(self, frame_id: int) -> np.ndarray | None:
        """Read a single specific frame by index."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self._cap.read()
        return frame if ret else None

    @property
    def fps(self) -> float:
        """Video frames per second."""
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frame_size(self) -> tuple[int, int]:
        """Returns (width, height)."""
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *args) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def __del__(self) -> None:
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()


def ffmpeg_path() -> str:
    """$COURTSHOT_FFMPEG, else ffmpeg on PATH."""
    return os.environ.get("COURTSHOT_FFMPEG") or "ffmpeg"


def ffprobe_path() -> str:
    """$COURTSHOT_FFPROBE, else ffprobe on PATH."""
    return os.environ.get("COURTSHOT_FFPROBE") or "ffprobe"


def output_size(width: int, height: int, max_height: int) -> tuple[int, int]:
    """(width, height) scaled to at most max_height high, both even (H.264
    with yuv420p needs even dimensions), keeping the aspect ratio."""
    h = max(2, min(height, max_height) // 2 * 2)
    w = max(2, round(width * h / height / 2) * 2)
    return w, h


class VideoWriter:
    """Writes BGR frames to an H.264 MP4 through an ffmpeg process.

    The video is scaled to an even size at most max_height high (yuv420p,
    fast start, no audio). It is written under a temporary name; on close()
    the file is checked with ffprobe (codec, pixel format, size, frame count)
    and renamed to ``path``. ffmpeg's messages go to this process's stderr.
    ``ffmpeg`` and ``ffprobe`` are the commands to run; None means
    ffmpeg_path() / ffprobe_path().

        with VideoWriter(out, fps=25.0, frame_size=(1920, 1080)) as w:
            w.write(frame)
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        max_height: int = 720,
        ffmpeg: str | None = None,
        ffprobe: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(self.path.name + ".tmp")
        self.frame_size = frame_size
        self.output_size = output_size(*frame_size, max_height)
        self._frame_count = 0
        self._ffprobe = ffprobe
        w, h = self.output_size
        cmd = [
            ffmpeg or ffmpeg_path(),
            "-nostats",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{frame_size[0]}x{frame_size[1]}",
            "-framerate",
            repr(float(fps)),
            "-i",
            "-",
            "-an",
            "-vf",
            f"scale={w}:{h}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(self._tmp),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame (BGR, frame_size)."""
        if frame.shape[:2] != (self.frame_size[1], self.frame_size[0]):
            raise ValueError(f"frame is {frame.shape[1]}x{frame.shape[0]}, expected {self.frame_size}")
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            raise RuntimeError(f"ffmpeg stopped while writing {self.path.name}: {self._why()}") from None
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count

    def close(self) -> None:
        """Finish the file, check it and move it into place."""
        self._close_stdin()
        try:
            if self._proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path.name}: {self._why()}")
            self._check()
        except BaseException:
            self._tmp.unlink(missing_ok=True)
            raise
        os.replace(self._tmp, self.path)

    def abort(self) -> None:
        """Stop ffmpeg and remove the unfinished file."""
        self._proc.kill()
        self._close_stdin()
        self._proc.wait()
        self._tmp.unlink(missing_ok=True)

    def _why(self) -> str:
        return f"exit {self._proc.wait()}, see ffmpeg's messages on stderr"

    def _close_stdin(self) -> None:
        with contextlib.suppress(BrokenPipeError):  # ffmpeg is gone; its exit code tells why
            self._proc.stdin.close()

    def _check(self) -> None:
        info = probe_video(self._tmp, self._ffprobe)
        want = {
            "codec_name": "h264",
            "pix_fmt": "yuv420p",
            "width": self.output_size[0],
            "height": self.output_size[1],
            "nb_read_packets": self._frame_count,
        }
        wrong = {k: info.get(k) for k, v in want.items() if info.get(k) != v}
        if wrong:
            raise RuntimeError(f"{self.path.name} came out wrong: {wrong}, expected {want}")

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, *args) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def probe_video(path: str | Path, ffprobe: str | None = None) -> dict:
    """ffprobe's view of the first video stream: codec_name, pix_fmt, width,
    height and nb_read_packets (the frame count), as numbers where numeric.
    ffprobe is the command to run; None means ffprobe_path()."""
    out = subprocess.run(
        [
            ffprobe or ffprobe_path(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,nb_read_packets",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    ).stdout
    streams = json.loads(out).get("streams") or [{}]
    return {k: int(v) if k == "nb_read_packets" else v for k, v in streams[0].items()}
