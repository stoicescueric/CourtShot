"""
The whole of CourtShot for one prepared video, as a job runs it:

    video ──track()───▶ raw tracking (cached in the work folder)
          ──analyse()─▶ stable player IDs, ball events, 3D ball flights
          ──render()──▶ tracked.mp4 (annotated) and plain.mp4, one pass

    shot = CourtShot("match.mp4", calibration, work_dir="work/", on_progress=print)
    analysis = shot.analyse()
    shot.render(analysis, "out/")

The JSON outputs come from courtshot.report (report_payload, scene_payload);
courtshot.job puts it all together.

Tracking is the slow, GPU-bound step. It is saved in the work folder as
<video>_tracking_cache.pkl and reused when it was made from this video,
calibration, ball model, settings and package version; the appearance vectors
and the 3D fit are cached next to it the same way. A retry of a job in the
same work folder therefore skips what already finished.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path

from courtshot.analysis import shots as shot_stats
from courtshot.analysis.identities import SLOT_NAMES, SLOT_TEAMS, StableIdConfig
from courtshot.analysis.match import MatchAnalysis, analyse_cache
from courtshot.schemas.tracking import CourtCalibration
from courtshot.tracking import cache as tracking_cache
from courtshot.tracking.device import detect_device
from courtshot.tracking.pipeline import TrackingPipeline, TrackingResult

logger = logging.getLogger(__name__)

# The fine-tuned ball model at the repository root (inference/src/courtshot/ -> repo).
DEFAULT_BALL_MODEL = Path(__file__).resolve().parents[3] / "checkpoints" / "padel_tracknetv3.pt"
# People kept per frame before the identity step picks the four players. More
# than 4 on purpose: an umpire or a reflection can score higher than a far
# player, and the identity step is the better judge of who is playing.
DEFAULT_MAX_PLAYERS = 8

Progress = Callable[[str, float], None]  # (stage, share of that stage done)


class CourtShot:
    """Tracking, analysis and videos for one match video.

    video_path:       the match video (fixed camera, constant frame rate)
    calibration:      the court calibration, in this video's pixels
    work_dir:         where the caches go; reused by a retry
    ball_model_path:  TrackNetV3 weights (default: DEFAULT_BALL_MODEL)
    device:           "mps", "cuda" or "cpu" for every model (None: detect)
    on_progress:      called with (stage, share done) for the stages
                      "tracking", "appearance", "analysis", "ball3d", "render"
    fit_workers:      processes for the 3D fit (None: ball_flight's default)
    enable_pose:      run pose estimation (needed to tell overheads and to
                      rule out hits by players who stood still)
    max_players:      people kept per frame for the identity step to choose from
    id_config:        settings of the identity step
    """

    def __init__(
        self,
        video_path: str | Path,
        calibration: CourtCalibration,
        work_dir: str | Path,
        ball_model_path: str | Path | None = None,
        device: str | None = None,
        on_progress: Progress | None = None,
        fit_workers: int | None = None,
        enable_pose: bool = True,
        max_players: int = DEFAULT_MAX_PLAYERS,
        id_config: StableIdConfig | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        self.ball_model_path = Path(ball_model_path or DEFAULT_BALL_MODEL)
        if not self.ball_model_path.exists():
            raise FileNotFoundError(f"Ball model not found: {self.ball_model_path}")
        self.calibration = calibration
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or detect_device()
        self.on_progress: Progress = on_progress or (lambda stage, done: None)
        self.fit_workers = fit_workers
        self.enable_pose = enable_pose
        self.max_players = max_players
        self.id_config = id_config
        self.cache_path = tracking_cache.cache_path_for(self.video_path, self.work_dir)

    # -- tracking ---------------------------------------------------------

    def tracking_key(self) -> dict[str, object]:
        """What the cached tracking of this video depends on."""
        return tracking_cache.tracking_key(
            self.video_path, self.calibration, self.ball_model_path, self.enable_pose, self.max_players
        )

    def track(self) -> TrackingResult:
        """Raw tracking of the video: from the cache when it can be reused,
        otherwise by running the detectors (and saving the result)."""
        key = self.tracking_key()
        self.on_progress("tracking", 0.0)
        tracking_cache.discard_if_stale(self.cache_path, key)
        if self.cache_path.exists():
            tracking = tracking_cache.load_tracking(self.cache_path)
            logger.info(
                "Tracking loaded from %s: %d player frames, %d ball frames",
                self.cache_path.name,
                len(tracking.player_frames),
                len(tracking.ball_frames),
            )
        else:
            logger.info("Tracking %s on %s ...", self.video_path.name, self.device)
            tracking = TrackingPipeline(
                video_path=self.video_path,
                ball_model_path=self.ball_model_path,
                calibration=self.calibration,
                enable_pose=self.enable_pose,
                max_players=self.max_players,
                device=self.device,
            ).run(on_progress=partial(self.on_progress, "tracking"))
            logger.info(
                "Tracking done: %d player frames, %d ball frames",
                len(tracking.player_frames),
                len(tracking.ball_frames),
            )
            tracking_cache.save_tracking(tracking, self.cache_path, key)
            logger.info("Tracking saved: %s", self.cache_path)
        self.on_progress("tracking", 1.0)
        return tracking

    # -- analysis ---------------------------------------------------------

    def analyse(self, fit_3d: bool = True) -> MatchAnalysis:
        """Track (or load the tracking), then stable IDs, ball events and,
        unless fit_3d is False, the 3D ball flights.

        Raises JobFailure when the players cannot be told apart
        (analysis.match.check_players).
        """
        raw = self.track()
        return analyse_cache(
            self.cache_path,
            video=self.video_path,
            raw=raw,
            id_config=self.id_config,
            fit_3d=fit_3d,
            device=self.device,
            fit_workers=self.fit_workers,
            on_progress=self.on_progress,
        )

    # -- videos -----------------------------------------------------------

    def render(
        self,
        analysis: MatchAnalysis,
        out_dir: str | Path,
        max_height: int = 720,
        shot_display_s: float = 1.5,
        bounce_display_s: float = 1.0,
        ffmpeg: str | None = None,
        ffprobe: str | None = None,
    ) -> list[str]:
        """tracked.mp4 (players, ball, ball speed, bounces and shot types drawn
        on the video) and plain.mp4 (the same frames as they were read), in one
        pass over the video. Both are H.264 at most max_height high, at the
        analysis frame rate, so frame k plays at k / fps like the 3D replay.
        ffmpeg / ffprobe are the commands to run (None: courtshot.io.video's
        defaults); ffmpeg's messages go to stderr, the job log.

        Returns the names of the two files.
        """
        from courtshot.io.video import VideoReader, VideoWriter
        from courtshot.viz.frame import FrameAnnotator

        out = Path(out_dir)
        tracking, ball, ball3d, fps = analysis.tracking, analysis.ball, analysis.ball3d, analysis.fps

        players_at: dict[int, list] = defaultdict(list)
        for pf in tracking.player_frames:
            players_at[pf.frame_id].append(pf)
        ball_at = {bf.frame_id: bf for bf in tracking.ball_frames}
        speeds = analysis.ball_speeds
        ball_court = ball3d.positions if ball3d is not None else {}

        shots = []
        if ball is not None:
            shots = shot_stats.build_shots(ball, tracking, dict(zip(SLOT_NAMES, SLOT_TEAMS)), ball3d, fps)
        shot_counts = Counter(s.type for s in shots)
        top_types = [(k.replace("_", "/"), n) for k, n in shot_counts.most_common(4)]

        # what to show in each frame, worked out once
        label_at: dict[int, str] = {}
        for s in sorted(shots, key=lambda s: s.time_s):
            first = round(s.time_s * fps)
            label = s.type.replace("_", " / ").upper()
            for f in range(first, first + int(shot_display_s * fps) + 1):
                label_at[f] = label  # a later shot takes over from an earlier one
        bounces_at: dict[int, list] = defaultdict(list)
        n_bounce = max(1, int(bounce_display_s * fps))
        for e in ball.events if ball is not None else []:
            if e.kind == "hit":
                continue
            first = round(e.time_s * fps)
            for k in range(n_bounce):
                bounces_at[first + k].append((e.px, e.kind, k / n_bounce))

        annotator = FrameAnnotator()
        logger.info("Writing tracked.mp4 and plain.mp4 to %s", out)
        self.on_progress("render", 0.0)
        with (
            VideoReader(self.video_path) as reader,
            VideoWriter(out / "tracked.mp4", fps, reader.frame_size, max_height, ffmpeg, ffprobe) as tracked,
            VideoWriter(out / "plain.mp4", fps, reader.frame_size, max_height, ffmpeg, ffprobe) as plain,
        ):
            total = max(1, reader.frame_count)
            for frame_id, _, frame in reader.frames():
                plain.write(frame)  # before the annotations are drawn into it
                stats: dict[str, str | int] = {"Frame": frame_id, "Shots": len(shots)}
                speed = speeds.get(frame_id) if speeds else None
                if speeds is not None:
                    stats["Ball"] = f"{speed:.0f} km/h" if speed is not None else "-"
                stats.update(top_types)
                xyz = ball_court.get(frame_id)
                annotator.annotate_frame(
                    frame=frame,
                    frame_id=frame_id,
                    player_frames=players_at.get(frame_id, []),
                    ball_frame=ball_at.get(frame_id),
                    calibration=tracking.calibration,
                    shot_label=label_at.get(frame_id),
                    stats=stats,
                    bounces=bounces_at.get(frame_id),
                    ball_speed_kmh=speed,
                    ball_court=None if xyz is None else (float(xyz[0]), float(xyz[1])),
                )
                tracked.write(frame)
                self.on_progress("render", min(1.0, (frame_id + 1) / total))
        self.on_progress("render", 1.0)
        logger.info("Videos saved: %d frames", plain.frame_count)
        return ["tracked.mp4", "plain.mp4"]
