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

from courtshot.schemas.tracking import CourtCalibration
from courtshot.tracking.pipeline import TrackingResult

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
    # The package was called "padex" before; caches made then name its
    # classes by that path.
    _OLD_PACKAGE = "padex"
    # Caches made before the ball's floor position and visibility state were
    # dropped from BallFrame still carry them.
    _GONE = {("courtshot.schemas.tracking", "Position3D"), ("courtshot.schemas.tracking", "BallVisibility")}

    def find_class(self, module: str, name: str):
        if module == self._OLD_PACKAGE or module.startswith(self._OLD_PACKAGE + "."):
            module = "courtshot" + module[len(self._OLD_PACKAGE) :]
        if (module, name) in self._GONE:
            return _LegacyField
        return super().find_class(module, name)


def load_pickle(path: str | Path):
    """Read a pickle CourtShot saved (also one saved by an earlier version)."""
    with open(path, "rb") as f:
        return _Unpickler(f).load()


def load_tracking(cache: str | Path) -> TrackingResult:
    """Read a tracking cache (also one made by an earlier version)."""
    return load_pickle(cache)


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
    people: str = "accurate",
) -> dict[str, object]:
    """Everything the cached tracking depends on."""
    from courtshot.tracking.device import use_pose_closeups_only
    from courtshot.tracking.player import PEOPLE_PRESETS

    preset = PEOPLE_PRESETS[people]

    def file_id(path: str | Path | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        return f"{p.name}:{p.stat().st_size}" if p.exists() else p.name

    cal = calibration.model_dump_json() if calibration else ""
    return {
        "video": file_id(video),
        "calibration_sha1": hashlib.sha1(cal.encode()).hexdigest(),
        # no ball model: the classical (frame difference) detector
        "ball_model": file_id(ball_model) if ball_model is not None else "classical",
        "use_tracknet_v3": ball_model is not None,  # caches from before the classical detector say True
        "enable_pose": enable_pose,
        "tracker": os.environ.get("COURTSHOT_TRACKER", "courtshot_bytetrack.yaml"),
        "max_players": max_players,
        "pose_closeups_only": bool(enable_pose and (preset.pose_closeups_only or use_pose_closeups_only())),
        "people": preset.key(),
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
    # the tracker configs were called padex_*.yaml before the package was renamed
    tracker = saved.get("tracker")
    if isinstance(tracker, str) and tracker.startswith("padex_"):
        saved["tracker"] = "courtshot_" + tracker[len("padex_") :]
    # caches from before the people presets were made with today's "accurate"
    saved.setdefault("people", "yolo26m.pt@1280")
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
