"""
Everything that is derived from a tracking cache, computed once:

    raw tracking -> stable player IDs -> poses around the ball's events
                 -> 2D ball events -> 3D ball flights

The tracked video, the match report and the 3D replay all start from the same
MatchAnalysis, so they agree on who is who and on what the ball did.

The 3D fit is the slow part (a minute or two per minute of play). It is saved
next to the tracking cache as <video>_tracking_cache_ball3d.pkl with a key
file, and reused as long as the tracking, the player identities and the
settings it was made from are unchanged.
"""

from __future__ import annotations

import contextlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

from courtshot.analysis import ball_events, ball_flight, poses
from courtshot.analysis.identities import (
    StableIdConfig,
    cache_appearance,
    frame_rate,
    stabilize_tracking,
)
from courtshot.analysis.swing import PoseTracks
from courtshot.tracking.cache import file_sha1, key_path_for, load_pickle, load_tracking
from courtshot.tracking.pipeline import TrackingResult

logger = logging.getLogger(__name__)


@dataclass
class MatchAnalysis:
    cache: Path  # the tracking cache this was made from
    raw: TrackingResult  # tracking as the detectors gave it
    tracking: TrackingResult  # the same with stable player IDs (P_1..P_4)
    id_report: dict[str, object]  # what the identity step did
    fps: float
    ball: ball_events.BallResult | None  # None: no ball detections or no calibration
    ball3d: ball_flight.Ball3DResult | None  # None: no ball, or the 3D fit was skipped
    poses: PoseTracks | None = None  # the players' keypoints; None: read them from the tracking

    @property
    def ball_speeds(self) -> dict[int, float] | None:
        """Ball speed in km/h per frame, from the fitted 3D flights."""
        return None if self.ball3d is None else ball_flight.frame_speeds(self.ball3d, self.fps)


def analyse_cache(
    cache: str | Path,
    video: str | Path | None = None,
    raw: TrackingResult | None = None,
    id_config: StableIdConfig | None = None,
    use_ball: bool = True,
    fit_3d: bool = True,
    refit: bool = False,
    fps: float | None = None,
) -> MatchAnalysis:
    """Analyse the tracking saved in ``cache``.

    video:  the match video; only needed to make the players' appearance
            vectors and poses when they have not been saved yet (without
            them the identities go by position and motion alone, and hits
            and overheads by the ball alone)
    raw:    the tracking, if it is already in memory
    use_ball / fit_3d: switch off everything that needs the ball / the 3D fit
    refit:  redo the 3D fit even when a saved one could be reused
    fps:    frame rate, if the one read from the tracking should not be used
    """
    cache = Path(cache)
    if raw is None:
        raw = load_tracking(cache)
    fps = fps or frame_rate(raw)
    appearance = cache_appearance(cache, video=video, tracking=raw)
    stable, id_report = stabilize_tracking(raw, id_config, appearance)

    ball = ball3d = pose_tracks = poses_mark = None
    if use_ball:
        pose_tracks, poses_mark = _pose_tracks(cache, raw, stable, fps, id_report, video)
        ball = ball_events.run_on_tracking(stable, fps, poses=pose_tracks)
    if ball is not None and fit_3d:
        ball3d = ball3d_for(cache, stable, ball, fps, id_report, refit=refit, poses_mark=poses_mark)
    return MatchAnalysis(cache, raw, stable, id_report, fps, ball, ball3d, pose_tracks)


def _pose_tracks(cache: Path, raw, stable, fps: float, id_report, video) -> tuple[PoseTracks, str | None]:
    """The players' keypoints, and what identifies them if they were made
    after tracking (None when the tracking itself carries them)."""
    if any(pf.keypoints for pf in raw.player_frames):  # a cache from when tracking made the poses
        return PoseTracks(stable), None
    wanted = True
    key_file = key_path_for(cache)
    if key_file.exists():
        with contextlib.suppress(OSError, ValueError):
            wanted = bool(json.loads(key_file.read_text(encoding="utf-8")).get("enable_pose", True))
    lookup, mark = poses.poses_for(cache, stable, fps, id_report, video) if wanted else (None, None)
    return PoseTracks(stable, poses=lookup), mark


# ---------------------------------------------------------------------------
# the saved 3D fit
# ---------------------------------------------------------------------------


def fit_path(tracking_cache: Path) -> Path:
    return tracking_cache.with_name(tracking_cache.stem + "_ball3d.pkl")


def key_path(fit: Path) -> Path:
    return fit.with_suffix(".key.json")


def fit_key(tracking_cache: Path, fps: float) -> dict:
    """What a saved 3D fit depends on: the tracking it was fitted to and the
    settings of the 2D and 3D ball steps. Kept next to the fit, so a fit made
    from other tracking (or other settings) is not reused."""
    return {
        "tracking_sha1": file_sha1(tracking_cache),
        "fps": round(fps, 4),
        "ball_settings": repr(ball_events.BallSettings()),
        "fit_settings": repr(ball_flight.FitSettings()),
    }


def stale_reason(fit: Path, tracking_cache: Path, key: dict) -> str | None:
    """Why the saved 3D fit can't be reused, or None if it can."""
    if not fit.exists():
        return "missing"
    kp = key_path(fit)
    if not kp.exists():
        # Fit saved before keys existed: all we can tell is which file is newer.
        if fit.stat().st_mtime < tracking_cache.stat().st_mtime:
            return "it is older than the tracking cache"
        return None
    try:
        saved = json.loads(kp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "its key file is unreadable"
    changed = [k for k in key if saved.get(k) != key[k]]
    return f"changed since it was fitted: {', '.join(changed)}" if changed else None


def ball3d_for(
    tracking_cache,
    stable,
    ball,
    fps: float,
    id_report: dict | None = None,
    refit: bool = False,
    poses_mark: str | None = None,
):
    """The 3D ball flights for a tracking cache: the saved fit when it was
    made from this tracking, these player identities and these settings,
    otherwise a new fit, which is saved.

    poses_mark: what identifies the poses made after tracking, if the hits
    were told with those (the fit depends on who hit the ball).

    Returns None when there is no court calibration or no ball track.
    """
    tracking_cache = Path(tracking_cache)
    fit = fit_path(tracking_cache)
    key = fit_key(tracking_cache, fps)
    # who hit each ball depends on the player identities
    key["player_ids"] = poses.player_ids_key(id_report)
    if poses_mark is not None:
        key["poses"] = poses_mark
    reason = "refit asked for" if refit else stale_reason(fit, tracking_cache, key)
    if fit.exists() and reason is None:
        try:
            r3 = load_pickle(fit)  # your own CourtShot cache file
            logger.info("3D ball: loaded %s (%d flights)", fit, len(r3.flights))
            return r3
        except (ImportError, AttributeError, pickle.UnpicklingError, EOFError):
            reason = "it was saved by an earlier version of CourtShot"
    if fit.exists():
        logger.info("3D ball: not reusing %s (%s)", fit, reason)
    logger.info("3D ball: fitting flights (a minute or two per minute of play)...")
    r3 = ball_flight.run_on_tracking(stable, ball, fps)
    if r3 is None:
        return None
    with open(fit, "wb") as f:
        pickle.dump(r3, f)
    key_path(fit).write_text(json.dumps(key, indent=1), encoding="utf-8")
    logger.info("3D ball: %d/%d flights fit well; saved %s", sum(fl.ok for fl in r3.flights), len(r3.flights), fit)
    return r3
