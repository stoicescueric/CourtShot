"""Every path and every file read or write of the server (design 8.3).

No client-supplied text becomes part of a path: extensions and result names come from
the fixed tables below, everything else is an integer id.
"""

import json
import os
import shutil
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starlette.responses import FileResponse

from .settings import GIB, MIB

VIDEO_TYPES = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "m4v": "video/x-m4v",
    "webm": "video/webm",
}


@dataclass(frozen=True)
class ResultFile:
    content_type: str
    max_bytes: int
    required: bool
    download: bool = False


CSV = "text/csv; charset=utf-8"
RESULT_FILES = {
    "report.json": ResultFile("application/json", 64 * MIB, True),
    "tracked.mp4": ResultFile("video/mp4", 4 * GIB, True),
    "plain.mp4": ResultFile("video/mp4", 4 * GIB, True),
    "summary.json": ResultFile("application/json", 64 * MIB, True),
    "positions.csv": ResultFile(CSV, 256 * MIB, True, download=True),
    "scene.json": ResultFile("application/json", 256 * MIB, False),
    "shots.csv": ResultFile(CSV, 64 * MIB, False, download=True),
    "ball_events.csv": ResultFile(CSV, 64 * MIB, False, download=True),
}
MAX_RUN_BYTES = 10 * GIB


class OffsetBeyondEnd(Exception):
    """A chunk's offset is past the bytes received so far."""

    def __init__(self, offset: int) -> None:
        super().__init__(f"offset is beyond the {offset} bytes received")
        self.offset = offset


class TooLarge(Exception):
    pass


class NoSpace(Exception):
    pass


# ponytail: 64 striped locks instead of one lock per file; unrelated files may wait on
# each other briefly. A dict of per-path locks if upload concurrency ever matters.
_LOCKS = [threading.Lock() for _ in range(64)]


def _lock(path: Path) -> threading.Lock:
    return _LOCKS[zlib.crc32(str(path).encode()) % len(_LOCKS)]


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _pwrite(path: Path, offset: int, data: bytes) -> None:
    # No O_CREAT on missing folders: a write for a deleted match fails with FileNotFoundError.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o640)
    try:
        view = memoryview(data)
        while view:
            written = os.pwrite(fd, view, offset)
            view = view[written:]
            offset += written
    finally:
        os.close(fd)


class Storage:
    def __init__(self, data_dir: Path, free_space_reserve_bytes: int) -> None:
        self.data_dir = data_dir
        self.root = data_dir / "matches"
        self.reserve = free_space_reserve_bytes

    def ensure_free_space(self, incoming: int) -> None:
        if shutil.disk_usage(self.data_dir).free - incoming < self.reserve:
            raise NoSpace("the server's disk is full")

    # Match folders

    def match_dir(self, match_id: int) -> Path:
        return self.root / str(match_id)

    def create_match_dir(self, match_id: int) -> None:
        # The id is new to the database, so anything already at this path is stale.
        self.delete_match_dir(match_id)
        self.root.mkdir(parents=True, exist_ok=True)
        self.match_dir(match_id).mkdir()

    def delete_match_dir(self, match_id: int) -> None:
        shutil.rmtree(self.match_dir(match_id), ignore_errors=True)

    def match_ids_on_disk(self) -> list[int]:
        if not self.root.is_dir():
            return []
        return [int(p.name) for p in self.root.iterdir() if p.is_dir() and p.name.isdigit()]

    # The uploaded video: source.part until complete, then source.<ext>

    def _source_part(self, match_id: int) -> Path:
        return self.match_dir(match_id) / "source.part"

    def source_path(self, match_id: int, ext: str) -> Path:
        if ext not in VIDEO_TYPES:
            raise ValueError(f"unknown video extension {ext!r}")
        return self.match_dir(match_id) / f"source.{ext}"

    def upload_offset(self, match_id: int, ext: str) -> int:
        final = self.source_path(match_id, ext)
        return final.stat().st_size if final.exists() else _size(self._source_part(match_id))

    def write_upload_chunk(self, match_id: int, ext: str, size_bytes: int, offset: int, data: bytes) -> int:
        """Write a chunk at `offset`; returns the new offset. At `size_bytes` the file is final."""
        if offset + len(data) > size_bytes:
            raise TooLarge("the chunk ends beyond the declared file size")
        final = self.source_path(match_id, ext)
        part = self._source_part(match_id)
        with _lock(part):
            if final.exists():
                return final.stat().st_size
            current = _size(part)
            if offset > current:
                raise OffsetBeyondEnd(current)
            _pwrite(part, offset, data)
            size = part.stat().st_size
            if size == size_bytes:
                os.replace(part, final)
            return size

    def video_response(self, match_id: int, ext: str) -> FileResponse:
        return _file_response(self.source_path(match_id, ext), VIDEO_TYPES[ext])

    # Results: runs/<job id>-<attempt>/<name>

    def runs_dir(self, match_id: int) -> Path:
        return self.match_dir(match_id) / "runs"

    def run_dir(self, match_id: int, job_id: int, attempt: int) -> Path:
        return self.runs_dir(match_id) / f"{job_id}-{attempt}"

    def _result_path(self, match_id: int, job_id: int, attempt: int, name: str) -> Path:
        if name not in RESULT_FILES:
            raise ValueError(f"unknown result file {name!r}")
        return self.run_dir(match_id, job_id, attempt) / name

    def result_offset(self, match_id: int, job_id: int, attempt: int, name: str) -> int:
        return _size(self._result_path(match_id, job_id, attempt, name))

    def write_result_chunk(self, match_id: int, job_id: int, attempt: int, name: str, offset: int, data: bytes) -> int:
        path = self._result_path(match_id, job_id, attempt, name)
        end = offset + len(data)
        if end > RESULT_FILES[name].max_bytes:
            raise TooLarge(f"{name} is larger than {RESULT_FILES[name].max_bytes} bytes")
        # mkdir without parents: a deleted match folder is not recreated.
        self.runs_dir(match_id).mkdir(exist_ok=True)
        path.parent.mkdir(exist_ok=True)
        with _lock(path):
            current = _size(path)
            if offset > current:
                raise OffsetBeyondEnd(current)
            others = sum(_size(p) for p in path.parent.iterdir() if p.name != name)
            if others + max(current, end) > MAX_RUN_BYTES:
                raise TooLarge(f"the results of one run exceed {MAX_RUN_BYTES} bytes")
            _pwrite(path, offset, data)
            return path.stat().st_size

    def result_names(self, match_id: int, job_id: int, attempt: int) -> list[str]:
        run = self.run_dir(match_id, job_id, attempt)
        return [name for name in RESULT_FILES if (run / name).is_file()]

    def missing_required(self, match_id: int, job_id: int, attempt: int) -> list[str]:
        run = self.run_dir(match_id, job_id, attempt)
        return [n for n, spec in RESULT_FILES.items() if spec.required and _size(run / n) == 0]

    def read_result_json(self, match_id: int, job_id: int, attempt: int, name: str) -> Any:
        return json.loads(self._result_path(match_id, job_id, attempt, name).read_bytes())

    def result_response(self, match_id: int, job_id: int, attempt: int, name: str) -> FileResponse:
        spec = RESULT_FILES[name]
        path = self._result_path(match_id, job_id, attempt, name)
        return _file_response(path, spec.content_type, download_name=name if spec.download else None)

    def run_folders(self) -> dict[int, list[str]]:
        """Run folder names per match folder on disk."""
        found: dict[int, list[str]] = {}
        for match_id in self.match_ids_on_disk():
            runs = self.runs_dir(match_id)
            if runs.is_dir():
                found[match_id] = [p.name for p in runs.iterdir()]
        return found

    def delete_run_folder(self, match_id: int, folder: str) -> None:
        shutil.rmtree(self.runs_dir(match_id) / folder, ignore_errors=True)


def _file_response(path: Path, content_type: str, download_name: str | None = None) -> FileResponse:
    if not path.is_file():
        raise FileNotFoundError(path)
    return FileResponse(path, media_type=content_type, filename=download_name)
