"""
The whole of CourtShot for one video:

    video ──track()──▶ raw tracking (cached next to the video)
          ──analyse()─▶ stable player IDs, ball events, 3D ball flights
          ──▶ tracked video, match report, 3D replay

    from courtshot import CourtShot

    p = CourtShot("match.mp4")              # needs match_calibration.json next to it
    outputs = p.run("output/match")     # everything; or step by step:

    analysis = p.analyse()
    p.export_video(analysis, "output/match/match_tracked.mp4")

Tracking is the slow, GPU-bound step. It is saved as
<video>_tracking_cache.pkl and reused when it was made from this video,
calibration, ball model and settings; delete the cache to force it.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from courtshot.analysis import shots as shot_stats
from courtshot.analysis.identities import SLOT_NAMES, SLOT_TEAMS, StableIdConfig
from courtshot.analysis.match import MatchAnalysis, analyse_cache, fit_path
from courtshot.schemas.tracking import CourtCalibration
from courtshot.tracking import cache as tracking_cache
from courtshot.tracking.pipeline import TrackingPipeline, TrackingResult
from courtshot.tracking.player import PEOPLE_PRESETS

logger = logging.getLogger(__name__)

DEFAULT_BALL_MODEL = Path("checkpoints/padel_tracknetv3.pt")
BALL_DETECTORS = ("classical", "tracknet")
# People kept per frame before the identity step picks the four players. More
# than 4 on purpose: an umpire or a reflection can score higher than a far
# player, and the identity step is the better judge of who is playing.
DEFAULT_MAX_PLAYERS = 8


def load_calibration(video: Path, calibration: CourtCalibration | str | Path | None) -> CourtCalibration:
    """The calibration given, or <video>_calibration.json next to the video."""
    if isinstance(calibration, CourtCalibration):
        return calibration
    path = Path(calibration) if calibration is not None else video.with_name(video.stem + "_calibration.json")
    if not path.exists():
        if calibration is not None:
            raise FileNotFoundError(f"Calibration file not found: {path}")
        raise ValueError(
            f"No court calibration found for '{video.name}'. Calibrate the court first:\n"
            f'  python -m courtshot calibrate "{video}"'
        )
    with open(path) as f:
        return CourtCalibration(**json.load(f))


class CourtShot:
    """Tracking, analysis and outputs for one match video.

    video_path:       the match video (fixed camera)
    calibration:      a CourtCalibration, the path of a calibration JSON, or
                      None for <video>_calibration.json next to the video
    ball_detector:    "classical" (frame differencing on the CPU, no model;
                      the default) or "tracknet" (the TrackNetV3 network)
    ball_model_path:  TrackNetV3 weights, for ball_detector="tracknet"
                      (default: checkpoints/padel_tracknetv3.pt)
    people:           "fast" (nano person detector, pose on close-ups) or
                      "accurate" (medium detector, full-frame pose pass)
    enable_pose:      run pose estimation (needed to tell overheads and to
                      rule out hits by players who stood still)
    cache_tracking:   save / reuse the tracking next to the video
    cache_dir:        where the cache goes (default: next to the video)
    max_players:      people kept per frame for the identity step to choose from
    id_config:        settings of the identity step
    """

    def __init__(
        self,
        video_path: str | Path,
        calibration: CourtCalibration | str | Path | None = None,
        ball_model_path: str | Path | None = None,
        ball_detector: str = "classical",
        people: str = "fast",
        enable_pose: bool = True,
        cache_tracking: bool = True,
        cache_dir: str | Path | None = None,
        max_players: int = DEFAULT_MAX_PLAYERS,
        id_config: StableIdConfig | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if ball_detector not in BALL_DETECTORS:
            raise ValueError(f"Unknown ball detector {ball_detector!r}; use one of {', '.join(BALL_DETECTORS)}")
        self.ball_detector = ball_detector
        if people not in PEOPLE_PRESETS:
            raise ValueError(f"Unknown people preset {people!r}; use one of {', '.join(PEOPLE_PRESETS)}")
        self.people = people
        self.ball_model_path: Path | None = None
        if ball_detector == "tracknet":
            self.ball_model_path = Path(ball_model_path or DEFAULT_BALL_MODEL)
            if not self.ball_model_path.exists():
                raise FileNotFoundError(f"Ball model not found: {self.ball_model_path}")
        self.calibration = load_calibration(self.video_path, calibration)
        self.enable_pose = enable_pose
        self.cache_tracking = cache_tracking
        self.cache_path = tracking_cache.cache_path_for(self.video_path, cache_dir)
        self.max_players = max_players
        self.id_config = id_config
        self.tracking_from_cache = False

    # -- tracking ---------------------------------------------------------

    def tracking_key(self) -> dict[str, object]:
        """What the cached tracking of this video depends on."""
        return tracking_cache.tracking_key(
            self.video_path, self.calibration, self.ball_model_path, self.enable_pose, self.max_players, self.people
        )  # no ball model = the classical detector

    def track(self) -> TrackingResult:
        """Raw tracking of the video: from the cache when it can be reused,
        otherwise by running the detectors (and saving the result)."""
        key = self.tracking_key()
        if self.cache_tracking:
            tracking_cache.set_aside_if_stale(self.cache_path, key)
            if self.cache_path.exists():
                self.tracking_from_cache = True
                tracking = tracking_cache.load_tracking(self.cache_path)
                logger.info(
                    "Tracking loaded from %s: %d player frames, %d ball frames",
                    self.cache_path.name,
                    len(tracking.player_frames),
                    len(tracking.ball_frames),
                )
                return tracking

        logger.info("Tracking %s ...", self.video_path.name)
        tracking = TrackingPipeline(
            video_path=self.video_path,
            ball_model_path=self.ball_model_path,
            calibration=self.calibration,
            enable_pose=self.enable_pose,
            max_players=self.max_players,
            people=self.people,
        ).run()
        logger.info(
            "Tracking done: %d player frames, %d ball frames", len(tracking.player_frames), len(tracking.ball_frames)
        )
        if self.cache_tracking:
            tracking_cache.save_tracking(tracking, self.cache_path, key)
            logger.info("Tracking saved: %s", self.cache_path)
        return tracking

    # -- analysis ---------------------------------------------------------

    def analyse(self, fit_3d: bool = True, refit: bool = False) -> MatchAnalysis:
        """Track (or load the tracking), then stable IDs, ball events and,
        unless fit_3d is False, the 3D ball flights.

        The appearance vectors and the 3D fit are saved next to the tracking
        cache, so they need cache_tracking to be on; without it the
        identities go by motion alone and there is no 3D fit.
        """
        raw = self.track()
        if not self.cache_path.exists():
            logger.warning("Tracking is not cached: no appearance check and no 3D ball fit")
            from courtshot.analysis import ball_events
            from courtshot.analysis.identities import frame_rate, stabilize_tracking

            stable, id_report = stabilize_tracking(raw, self.id_config)
            fps = frame_rate(raw)
            return MatchAnalysis(
                self.cache_path, raw, stable, id_report, fps, ball_events.run_on_tracking(stable, fps), None
            )
        return analyse_cache(
            self.cache_path, video=self.video_path, raw=raw, id_config=self.id_config, fit_3d=fit_3d, refit=refit
        )

    # -- outputs ----------------------------------------------------------

    def export_video(
        self,
        analysis: MatchAnalysis,
        output_path: str | Path,
        shot_display_s: float = 1.5,
        bounce_display_s: float = 1.0,
    ) -> Path:
        """The video with players, ball, ball speed, bounces and shot types drawn on it."""
        from courtshot.io.video import VideoReader, VideoWriter
        from courtshot.viz.frame import FrameAnnotator

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tracking, ball, ball3d = analysis.tracking, analysis.ball, analysis.ball3d

        players_at: dict[int, list] = defaultdict(list)
        for pf in tracking.player_frames:
            players_at[pf.frame_id].append(pf)
        ball_at = {bf.frame_id: bf for bf in tracking.ball_frames}
        speeds = analysis.ball_speeds
        ball_court = ball3d.positions if ball3d is not None else {}

        shots = []
        if ball is not None:
            shots = shot_stats.build_shots(ball, tracking, dict(zip(SLOT_NAMES, SLOT_TEAMS)), ball3d, analysis.fps)
        shot_counts = Counter(s.type for s in shots)
        top_types = [(k.replace("_", "/"), n) for k, n in shot_counts.most_common(4)]

        annotator = FrameAnnotator()
        logger.info("Writing the tracked video to %s", output_path)
        with VideoReader(self.video_path) as reader:
            fps = reader.fps
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

            with VideoWriter(output_path, fps=fps, frame_size=reader.frame_size) as writer:
                for frame_id, _, frame in reader.frames():
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
                    writer.write(frame)
                    if frame_id % 500 == 0:
                        logger.info("  frame %d / %d", frame_id, reader.frame_count)
        logger.info("Tracked video saved: %s", output_path)
        return output_path

    def run(
        self,
        out_dir: str | Path | None = None,
        video: bool = True,
        report: bool = True,
        replay: bool = True,
        fit_3d: bool = True,
        redo_video: bool = False,
    ) -> dict[str, Path]:
        """Everything for this video; returns the paths of what was written.

        out_dir: default output/<video name>/
        video / report / replay: which outputs to make
        fit_3d: False skips the 3D ball fit (faster; no ball speed on the
                video, rougher shot data, no 3D replay)
        redo_video: write the tracked video again even when the one in
                out_dir is newer than the tracking and the 3D fit it shows
        """
        from courtshot.report.match_report import print_summary, write_match_report
        from courtshot.report.scene3d import write_scene

        name = self.video_path.stem
        out = Path(out_dir) if out_dir is not None else Path("output") / name
        out.mkdir(parents=True, exist_ok=True)
        analysis = self.analyse(fit_3d=fit_3d)
        written: dict[str, Path] = {}

        if video:
            out_video = out / f"{name}_tracked.mp4"
            fit = fit_path(self.cache_path)
            up_to_date = (
                not redo_video
                and self.tracking_from_cache
                and out_video.exists()
                and (not fit.exists() or out_video.stat().st_mtime >= fit.stat().st_mtime)
            )
            if up_to_date:
                logger.info("Tracking and 3D fit reused; keeping %s", out_video)
            else:
                self.export_video(analysis, out_video)
            written["video"] = out_video
        if report:
            print_summary(write_match_report(analysis, out))
            written["report"] = out / "report.html"
        if replay and analysis.ball3d is not None:
            written["replay"] = write_scene(analysis, out / "scene3d.html", video=self.video_path)
        return written
