"""
The players' pose keypoints, made after tracking and only where they are used.

The poses answer two questions, both about the moments when the ball changes
course near a player: did this player swing at it (ball_events), and was an
arm up (shots)? That is a fraction of a second around each such moment and one
or two players, a small part of the video. So instead of a pose for every
player in every frame during tracking, which was more than half of its work:

  1. ball_events.swings_asked() lists every (player, moment) the analysis can
     ask about;
  2. YOLO-Pose runs on the frames around each (the window swing.py looks
     at), in batches, the way tracking ran it on every frame: the full frame
     first, then a close-up of each wanted player it found no pose for.

The keypoints are saved next to the tracking cache as
<video>_tracking_cache_pose.npz, so the report and the 3D replay (which only
read the caches) see the same swings as the run that had the video.

    poses = poses_for("match_tracking_cache.pkl", stable, fps, id_report, video="match.mp4")
    tracks = PoseTracks(stable, poses=poses[0])

Tracking caches made when tracking did the poses carry the keypoints in their
player frames and need none of this.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from courtshot.analysis import ball_events
from courtshot.analysis.identities import SLOT_NAMES, StableIdConfig
from courtshot.analysis.swing import IDX, NAMES, PoseSettings, swing_window

logger = logging.getLogger(__name__)

SMOOTHING = 0.6  # weight of the new frame in the running average of a player's keypoints
SMOOTH_OVER_S = 0.2  # ...which starts again after a gap longer than this
BATCH = 8  # frames per pass of the pose model


def frames_wanted(stable, fps: float) -> dict[int, list[str]]:
    """frame -> the players whose pose is needed in it."""
    settings = PoseSettings()
    wanted: dict[int, set[str]] = {}
    for player, frame in ball_events.swings_asked(stable, fps):
        first, last = swing_window(frame, fps, settings)
        for f in range(first, last + 1):
            wanted.setdefault(f, set()).add(player)
    return {f: sorted(players) for f, players in sorted(wanted.items())}


def extract(
    video: str | Path, stable, wanted: dict[int, list[str]], fps: float, estimator=None, step: int | None = None
) -> dict:
    """The keypoints of the wanted players, the way tracking used to make
    them: one pass of the pose model over the full frame, and a close-up of
    every wanted player that pass found no pose for.

    In video faster than the people detector works at, the pose model sees
    every ``step``-th frame too (default: the same step) and the frames
    between get the keypoints half-way; the swing is then measured over the
    same stretch of time as in 25-30 fps video, which is what its thresholds
    were set on.

    Returns arrays: frames (N,), players (N,) as indices into SLOT_NAMES,
    xy (N, 17, 2) in image pixels and conf (N, 17).
    """
    from courtshot.io.video import FramePicker
    from courtshot.tracking.pipeline import player_step

    if estimator is None:
        from courtshot.tracking.device import detect_device
        from courtshot.tracking.player import YoloPoseEstimator

        estimator = YoloPoseEstimator(device=detect_device())
    step = step or player_step(fps)

    # the frames the model sees: the wanted ones on its grid and, for the ones between, those either side
    seen: dict[int, set[str]] = {}
    for frame_id, names in wanted.items():
        on_grid = frame_id - frame_id % step
        for g in (on_grid,) if on_grid == frame_id else (on_grid, on_grid + step):
            seen.setdefault(g, set()).update(names)
    # only players the detector saw: a position filled in across a gap has no picture behind it
    boxes_at: dict[int, list[tuple[str, object]]] = {}
    for pf in stable.player_frames:
        if pf.frame_id in seen and pf.confidence > 0 and pf.player_id in SLOT_NAMES:
            boxes_at.setdefault(pf.frame_id, []).append((pf.player_id, pf.bbox))

    made: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    previous: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}  # player -> (frame, xy, conf) as kept
    gap = max(step, int(round(SMOOTH_OVER_S * fps)))

    def keep(frame_id: int, player: str, keypoints) -> None:
        pts = np.full((len(NAMES), 2), np.nan, np.float32)
        cf = np.zeros(len(NAMES), np.float32)
        for k in keypoints:
            pts[IDX[k.name]] = (k.x, k.y)
            cf[IDX[k.name]] = k.confidence
        before = previous.get(player)
        if before is not None and frame_id - before[0] <= gap:
            both = (cf > 0) & (before[2] > 0)
            pts[both] = SMOOTHING * pts[both] + (1 - SMOOTHING) * before[1][both]
            cf[both] = SMOOTHING * cf[both] + (1 - SMOOTHING) * before[2][both]
        previous[player] = (frame_id, pts, cf)
        made[(player, frame_id)] = (pts, cf)

    def run(batch: list[tuple[int, np.ndarray]]) -> None:
        full = estimator.predict_full_frames([image for _, image in batch])
        found: dict[tuple[int, str], list] = {}
        closeups, whose = [], []
        for (frame_id, image), poses in zip(batch, full):
            here = boxes_at[frame_id]
            for (player, box), keypoints in zip(here, estimator.assign([box for _, box in here], poses)):
                if player not in seen[frame_id]:
                    continue
                found[(frame_id, player)] = keypoints
                if not keypoints:
                    closeups.append((image, box))
                    whose.append((frame_id, player))
        for at, keypoints in zip(whose, estimator.estimate_closeups(closeups)):
            found[at] = keypoints
        for (frame_id, player), keypoints in found.items():  # in frame order, as the smoothing needs
            if keypoints:
                keep(frame_id, player, keypoints)

    with FramePicker(video) as picker:
        batch: list[tuple[int, np.ndarray]] = []
        for frame_id in sorted(seen):
            if frame_id not in boxes_at:
                continue
            image = picker.read(frame_id)
            if image is None:
                break
            batch.append((frame_id, image))
            if len(batch) >= BATCH:
                run(batch)
                batch = []
        if batch:
            run(batch)

    frames: list[int] = []
    players: list[int] = []
    xy: list[np.ndarray] = []
    conf: list[np.ndarray] = []
    for frame_id, names in wanted.items():
        for player in names:
            on_grid = frame_id - frame_id % step
            pose = made.get((player, on_grid))
            if on_grid != frame_id:  # a frame between: part of the way from one pose to the next
                after = made.get((player, on_grid + step))
                if pose is None or after is None:
                    continue
                share = (frame_id - on_grid) / step
                both = (pose[1] > 0) & (after[1] > 0)
                pts = np.where(both[:, None], pose[0] + share * (after[0] - pose[0]), np.nan).astype(np.float32)
                pose = (pts, np.where(both, np.minimum(pose[1], after[1]), 0.0).astype(np.float32))
            if pose is None:
                continue
            frames.append(frame_id)
            players.append(SLOT_NAMES.index(player))
            xy.append(pose[0])
            conf.append(pose[1])
    return {
        "frames": np.array(frames, np.int32),
        "players": np.array(players, np.int8),
        "xy": np.stack(xy) if xy else np.empty((0, len(NAMES), 2), np.float32),
        "conf": np.stack(conf) if conf else np.empty((0, len(NAMES)), np.float32),
    }


def as_lookup(data: dict) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """The arrays of extract() as PoseTracks takes them: (player, frame) -> (xy, conf)."""
    return {
        (SLOT_NAMES[int(p)], int(f)): (data["xy"][i].astype(np.float64), data["conf"][i].astype(np.float64))
        for i, (f, p) in enumerate(zip(data["frames"], data["players"]))
    }


# ---------------------------------------------------------------------------
# saved next to the tracking cache
# ---------------------------------------------------------------------------


def sidecar(tracking_cache: str | Path) -> Path:
    p = Path(tracking_cache)
    return p.with_name(p.stem + "_pose.npz")


def player_ids_key(id_report: dict | None) -> str:
    """What says who is P_1..P_4: the identity settings and the swaps put right."""
    return repr(StableIdConfig()) + " swaps " + json.dumps((id_report or {}).get("swaps_fixed", []))


def _key(tracking_cache: Path, fps: float, id_report: dict | None) -> dict:
    from courtshot.tracking.cache import file_sha1
    from courtshot.tracking.pipeline import player_step
    from courtshot.tracking.player import YoloPoseEstimator

    return {
        "tracking_sha1": file_sha1(tracking_cache),
        "fps": round(fps, 4),
        "player_ids": player_ids_key(id_report),
        "ball_settings": repr(ball_events.BallSettings()),
        "window": [PoseSettings().before_s, PoseSettings().after_s],
        "model": YoloPoseEstimator.WEIGHTS,
        "closeup": [list(YoloPoseEstimator.CLOSEUP_SIZE), YoloPoseEstimator.CROP_PAD],
        "every": player_step(fps),
        "smoothing": [SMOOTHING, SMOOTH_OVER_S],
    }


def poses_for(
    tracking_cache: str | Path,
    stable,
    fps: float,
    id_report: dict | None = None,
    video: str | Path | None = None,
) -> tuple[dict | None, str | None]:
    """The saved poses that belong to this tracking cache and these player
    identities, as (lookup for PoseTracks, what identifies them).

    If there are none (or they were made for other tracking, identities or
    settings) and ``video`` is given, they are made now and saved. Without a
    video, returns (None, None): the analysis then goes by the ball alone.
    """
    tracking_cache = Path(tracking_cache)
    path = sidecar(tracking_cache)
    key = _key(tracking_cache, fps, id_report)
    mark = json.dumps(key, sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if json.loads(str(z["key"])) == key:
                return as_lookup({k: z[k] for k in ("frames", "players", "xy", "conf")}), mark
        logger.info("%s was made from other tracking or settings", path.name)
    if video is None or not Path(video).exists():
        logger.warning(
            "No pose data for %s (it is made by `courtshot run`, which has the video); "
            "hits and overheads are told by the ball alone.",
            tracking_cache.name,
        )
        return None, None
    wanted = frames_wanted(stable, fps)
    logger.info("Poses: %d frames around the ball's events, from %s ...", len(wanted), Path(video).name)
    data = extract(video, stable, wanted, fps)
    np.savez_compressed(path, key=json.dumps(key), **data)
    logger.info("Poses: %d made, saved %s", len(data["frames"]), path.name)
    return as_lookup(data), mark
