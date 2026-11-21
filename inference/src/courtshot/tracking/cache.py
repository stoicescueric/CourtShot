"""
The tracking cache: the raw tracking of a video, saved in the job's work
folder as <video>_tracking_cache.pkl so that a retry skips the slow part.

A key file (<video>_tracking_cache.key.json) records what the tracking was
made from. Tracking made from another video, calibration, model, setting or
package version is not reused: it is deleted and redone. A cache without its
key file is not trusted either. Both files are written under a temporary name
and renamed, so a crash never leaves a half-written cache behind.

The cache is a pickle of your own tracking; only load caches you made.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from courtshot.schemas.tracking import CourtCalibration
from courtshot.tracking.pipeline import TrackingResult

logger = logging.getLogger(__name__)


def cache_path_for(video: str | Path, cache_dir: str | Path | None = None) -> Path:
    video = Path(video)
    return Path(cache_dir or video.parent) / f"{video.stem}_tracking_cache.pkl"


def key_path_for(cache: Path) -> Path:
    return cache.with_suffix(".key.json")


def load_pickle(path: str | Path):
    """Read a pickle CourtShot saved."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tracking(cache: str | Path) -> TrackingResult:
    """Read a tracking cache."""
    return load_pickle(cache)


@contextmanager
def atomic_file(path: Path) -> Iterator[BinaryIO]:
    """A file opened for writing under a temporary name and renamed to path
    once it is complete; on an error the temporary file is removed."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_pickle(obj: object, path: Path, key: dict[str, object]) -> None:
    """Pickle obj to path, then its key next to it (the key last: a cache
    is only trusted with its key)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_file(path) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    with atomic_file(key_path_for(path)) as f:
        f.write(json.dumps(key, indent=1).encode("utf-8"))


def save_tracking(tracking: TrackingResult, cache: str | Path, key: dict[str, object]) -> None:
    save_pickle(tracking, Path(cache), key)


def tracking_key(
    video: Path,
    calibration: CourtCalibration | None,
    ball_model: str | Path | None,
    enable_pose: bool,
    max_players: int | None,
) -> dict[str, object]:
    """Everything the cached tracking depends on."""
    from courtshot import __version__
    from courtshot.tracking.device import use_pose_closeups_only

    def file_id(path: str | Path | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        return f"{p.name}:{p.stat().st_size}" if p.exists() else p.name

    cal = calibration.model_dump_json() if calibration else ""
    return {
        "video": file_id(video),
        "calibration_sha1": hashlib.sha1(cal.encode()).hexdigest(),
        "ball_model": file_id(ball_model),
        "enable_pose": enable_pose,
        "tracker": os.environ.get("COURTSHOT_TRACKER", "courtshot_bytetrack.yaml"),
        "max_players": max_players,
        "pose_closeups_only": bool(enable_pose and use_pose_closeups_only()),
        "package_version": __version__,
    }


def stale_reason(cache: Path, key: dict[str, object]) -> str | None:
    """Why the cache can't be reused for this key, or None if it can."""
    try:
        saved = json.loads(key_path_for(cache).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "its key file is missing"
    except (OSError, ValueError):
        return "its key file is unreadable"
    changed = [k for k in key if saved.get(k) != key[k]]
    return f"changed since it was made: {', '.join(changed)}" if changed else None


def discard_if_stale(cache: Path, key: dict[str, object]) -> None:
    """Delete a cache (and its key) that was not made from these inputs."""
    if not cache.exists():
        return
    reason = stale_reason(cache, key)
    if reason is not None:
        cache.unlink()
        key_path_for(cache).unlink(missing_ok=True)
        logger.warning("Not reusing %s: %s. Tracking again.", cache.name, reason)
