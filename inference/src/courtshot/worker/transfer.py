"""Resumable transfers: the job's video down (HTTP ranges), result files up (chunks at offsets)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from courtshot.worker.client import NETWORK_ERRORS, JobRef, ServerClient, ServerUnavailable

READ_BYTES = 64 * 1024  # what a broken connection can cost: a partial read is discarded

Checkpoint = Callable[[], None]  # raises to abort the transfer


def download_video(
    client: ServerClient, job: JobRef, dest: Path, size: int, checkpoint: Checkpoint,
    on_progress: Callable[[float], None],
) -> None:
    """Fetch the video into `dest`, continuing from whatever `dest` already holds.

    Raises ServerUnavailable when the transfer breaks off; calling again resumes.
    """
    have = dest.stat().st_size if dest.exists() else 0
    if have > size:
        dest.unlink()
        have = 0
    if have == size:
        return
    with client.open_video(job, have) as response:
        if response.status_code != 206:
            have = 0  # the server sent the whole file
        with dest.open("ab" if have else "wb") as out:
            try:
                for block in response.iter_content(READ_BYTES):
                    checkpoint()
                    out.write(block)
                    have += len(block)
                    on_progress(have / size)
            except NETWORK_ERRORS as exc:
                raise ServerUnavailable(f"download broke off at byte {have}: {exc}") from exc
    if have > size:
        dest.unlink()
        raise RuntimeError(f"the server sent {have} bytes for a video of {size}")
    if have < size:
        raise ServerUnavailable(f"download ended at byte {have} of {size}")


def upload_file(
    client: ServerClient, job: JobRef, path: Path, chunk_bytes: int, checkpoint: Checkpoint,
    on_held: Callable[[int], None],
) -> None:
    """Send `path` under its own name, from the offset the server already has. Calling again resumes.

    on_held(n) is told how many bytes of the file the server holds, after every answer.
    """
    size = path.stat().st_size
    offset = client.file_offset(job, path.name)
    if offset > size:
        raise RuntimeError(f"the server holds {offset} bytes of {path.name}, which has {size}")
    on_held(offset)
    with path.open("rb") as source:
        while offset < size:
            checkpoint()
            source.seek(offset)
            new_offset = client.send_chunk(job, path.name, offset, source.read(chunk_bytes))
            if new_offset == offset or new_offset > size:
                raise RuntimeError(f"the server answered offset {new_offset} to a chunk at {offset} of {path.name}")
            offset = new_offset
            on_held(offset)
