"""
The tracking cache: the raw tracking of a video, saved next to it as
<video>_tracking_cache.pkl so that the slow part runs once.

A key file (<video>_tracking_cache.key.json) records what the tracking was
made from. Tracking made from another video, calibration, model or setting is
not reused: it is set aside as ..._tracking_cache.old.pkl and redone.

The cache is a pickle of your own tracking; only load caches you made.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path

from padex.schemas.tracking import CourtCalibration
from padex.tracking.pipeline import TrackingResult

logger = logging.getLogger(__name__)


def cache_path_for(video: str | Path, cache_dir: str | Path | None = None) -> Path:
    video = Path(video)
    return Path(cache_dir or video.parent) / f"{video.stem}_tracking_cache.pkl"


def key_path_for(cache: Path) -> Path:
    return cache.with_suffix(".key.json")


class _LegacyField:
    """Stands in for a value older caches hold and nothing reads any more."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        pass


class _Unpickler(pickle.Unpickler):
    # Caches made before the ball's floor position and visibility state were
    # dropped from BallFrame still carry them.
    _GONE = {("padex.schemas.tracking", "Position3D"), ("padex.schemas.tracking", "BallVisibility")}

    def find_class(self, module: str, name: str):
        if (module, name) in self._GONE:
            return _LegacyField
        return super().find_class(module, name)


def load_tracking(cache: str | Path) -> TrackingResult:
    """Read a tracking cache (also one made by an earlier version)."""
    with open(cache, "rb") as f:
        return _Unpickler(f).load()


def save_tracking(tracking: TrackingResult, cache: str | Path, key: dict[str, object]) -> None:
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(tracking, f)
    key_path_for(cache).write_text(json.dumps(key, indent=1), encoding="utf-8")


def tracking_key(
    video: Path,
    calibration: CourtCalibration | None,
    ball_model: str | Path | None,
    enable_pose: bool,
    max_players: int | None,
) -> dict[str, object]:
    """Everything the cached tracking depends on."""
    from padex.tracking.device import use_pose_closeups_only

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
        "use_tracknet_v3": True,  # the only ball model now; kept so older caches still match
        "enable_pose": enable_pose,
        "tracker": os.environ.get("PADEX_TRACKER", "padex_bytetrack.yaml"),
        "max_players": max_players,
        "pose_closeups_only": bool(enable_pose and use_pose_closeups_only()),
    }


def stale_reason(cache: Path, key: dict[str, object]) -> str | None:
    """Why the cache can't be reused for this key, or None if it can."""
    key_path = key_path_for(cache)
    if not key_path.exists():
        # Cache from before keys were saved: nothing to compare with.
        logger.warning(
            "%s is missing, so the cached tracking can't be checked against "
            "the current video, calibration and models. Delete the cache to "
            "re-track.",
            key_path.name,
        )
        return None
    try:
        saved = json.loads(key_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "its key file is unreadable"
    changed = [k for k in key if saved.get(k) != key[k]]
    return f"changed since it was made: {', '.join(changed)}" if changed else None


def set_aside_if_stale(cache: Path, key: dict[str, object]) -> None:
    """Move a cache that was made from other inputs out of the way."""
    if not cache.exists():
        return
    reason = stale_reason(cache, key)
    if reason is not None:
        old = cache.with_suffix(".old.pkl")
        os.replace(cache, old)
        logger.warning("Not reusing %s: %s. Kept as %s; tracking again.", cache.name, reason, old.name)
