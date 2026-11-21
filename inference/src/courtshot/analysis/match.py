"""
Everything that is derived from a tracking cache, computed once:

    raw tracking -> stable player IDs -> 2D ball events -> 3D ball flights

The tracked video, the match report and the 3D replay all start from the same
MatchAnalysis, so they agree on who is who and on what the ball did.

The 3D fit is the slow part (a minute or two per minute of play). It is saved
next to the tracking cache as <video>_tracking_cache_ball3d.pkl with a key
file, and reused as long as the tracking, the player identities and the
settings it was made from are unchanged. A fit without its key is redone.

Right after the identity step, a match the rest cannot work with stops with a
JobFailure: four players never positioned together, or players on court for
less than the movement rally window.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from courtshot.analysis import ball_events, ball_flight
from courtshot.analysis.identities import (
    SLOT_NAMES,
    StableIdConfig,
    cache_appearance,
    frame_rate,
    stabilize_tracking,
)
from courtshot.analysis.movement import Settings
from courtshot.errors import JobFailure
from courtshot.tracking.cache import load_pickle, load_tracking, save_pickle
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
    device: str | None = None,
    fit_workers: int | None = None,
    on_progress: Callable[[str, float], None] | None = None,
) -> MatchAnalysis:
    """Analyse the tracking saved in ``cache``.

    video:  the match video; only needed to make the players' appearance
            vectors when they have not been saved yet (without them the
            identities go by position and motion alone)
    raw:    the tracking, if it is already in memory
    use_ball / fit_3d: switch off everything that needs the ball / the 3D fit
    refit:  redo the 3D fit even when a saved one could be reused
    fps:    frame rate, if the one read from the tracking should not be used
    device: where the appearance model runs (default: detect_device())
    fit_workers: processes for the 3D fit (default: ball_flight._default_workers)
    on_progress: (stage, share done) for the stages "appearance", "analysis"
            and "ball3d"

    Raises JobFailure when the players cannot be told apart (see check_players).
    """
    progress = on_progress or (lambda stage, done: None)
    cache = Path(cache)
    if raw is None:
        raw = load_tracking(cache)
    fps = fps or frame_rate(raw)
    progress("appearance", 0.0)
    appearance = cache_appearance(
        cache, video=video, tracking=raw, device=device, on_progress=partial(progress, "appearance")
    )
    progress("appearance", 1.0)
    progress("analysis", 0.0)
    stable, id_report = stabilize_tracking(raw, id_config, appearance)
    check_players(stable, id_report, fps)

    ball = ball_events.run_on_tracking(stable, fps) if use_ball else None
    progress("analysis", 1.0)
    ball3d = None
    if ball is not None and fit_3d:
        progress("ball3d", 0.0)
        fit_progress = partial(progress, "ball3d")
        ball3d = ball3d_for(
            cache, stable, ball, fps, id_report, refit=refit, workers=fit_workers, on_progress=fit_progress
        )
        progress("ball3d", 1.0)
    return MatchAnalysis(cache, raw, stable, id_report, fps, ball, ball3d)


def check_players(stable: TrackingResult, id_report: dict[str, object], fps: float) -> None:
    """Stop early, with a reason the user can act on, when the identity step
    found no frame with four positioned players, or when the players are on
    court for less than the movement analysis' rally window."""
    if "seed_frame" not in id_report:
        raise JobFailure(
            "Four players were never seen on court together. Check that the whole court is in view "
            "and that the court calibration matches the video."
        )
    frames = [pf.frame_id for pf in stable.player_frames if pf.player_id in SLOT_NAMES]
    seconds = (max(frames) - min(frames) + 1) / fps
    window = Settings().rally_window
    if seconds < window:
        raise JobFailure(f"The players are on court for only {seconds:.1f} s; at least {window:.0f} s are needed.")


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
    h = hashlib.sha1()
    with open(tracking_cache, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {
        "tracking_sha1": h.hexdigest(),
        "fps": round(fps, 4),
        "ball_settings": repr(ball_events.BallSettings()),
        "fit_settings": repr(ball_flight.FitSettings()),
    }


def stale_reason(fit: Path, key: dict) -> str | None:
    """Why the saved 3D fit can't be reused, or None if it can."""
    if not fit.exists():
        return "missing"
    try:
        saved = json.loads(key_path(fit).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "its key file is missing"
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
    workers: int | None = None,
    on_progress: Callable[[float], None] | None = None,
):
    """The 3D ball flights for a tracking cache: the saved fit when it was
    made from this tracking, these player identities and these settings,
    otherwise a new fit in ``workers`` processes, which is saved.

    Returns None when there is no court calibration or no ball track.
    """
    tracking_cache = Path(tracking_cache)
    fit = fit_path(tracking_cache)
    key = fit_key(tracking_cache, fps)
    # who hit each ball depends on the player identities
    key["player_ids"] = repr(StableIdConfig()) + " swaps " + json.dumps((id_report or {}).get("swaps_fixed", []))
    reason = "refit asked for" if refit else stale_reason(fit, key)
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
    r3 = ball_flight.run_on_tracking(stable, ball, fps, workers=workers, on_progress=on_progress)
    if r3 is None:
        return None
    save_pickle(r3, fit, key)
    logger.info("3D ball: %d/%d flights fit well; saved %s", sum(fl.ok for fl in r3.flights), len(r3.flights), fit)
    return r3
