"""
Video -> raw tracking: players (boxes, tracker IDs, pose) and the ball, frame
by frame. This is the slow, GPU-bound part of CourtShot; its result is cached
(courtshot.tracking.cache) and everything else works from that.

Original author: Xingnan Zhu.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from courtshot.schemas.tracking import BallFrame, CourtCalibration, PlayerFrame
from courtshot.tracking.ball import BallDetector
from courtshot.tracking.ball_classical import ClassicalBallDetector
from courtshot.tracking.device import detect_device, use_threads
from courtshot.tracking.player import PEOPLE_PRESETS, PlayerDetector, YoloPersonTracker, YoloPoseEstimator

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """What tracking a video gives."""

    player_frames: list[PlayerFrame] = field(default_factory=list)
    ball_frames: list[BallFrame] = field(default_factory=list)
    calibration: CourtCalibration | None = None


class TrackingPipeline:
    """Runs the player and ball detectors over a video.

    Without a calibration the players have no court position (and nobody is
    filtered out for standing outside the court). Without a ball model the
    ball is found by the classical detector (no network).
    """

    def __init__(
        self,
        video_path: str | Path,
        ball_model_path: str | Path | None = None,
        calibration: CourtCalibration | None = None,
        enable_pose: bool = True,
        max_players: int = 4,
        device: str | None = None,
        people: str = "fast",
        player_detector: PlayerDetector | None = None,
        ball_detector: BallDetector | ClassicalBallDetector | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.calibration = calibration
        self.device = device or detect_device()
        logger.info("Inference device: %s", self.device)

        if player_detector is None:
            preset = PEOPLE_PRESETS[people]
            player_detector = PlayerDetector(
                tracker=YoloPersonTracker(model_path=preset.detector, device=self.device, imgsz=preset.imgsz),
                pose=YoloPoseEstimator(device=self.device) if enable_pose else None,
                max_players=max_players,
                # $COURTSHOT_POSE_CLOSEUPS=1 still turns close-ups on for "accurate"
                pose_closeups_only=True if preset.pose_closeups_only else None,
            )
        self.player_detector = player_detector
        if ball_detector is None:
            ball_detector = (
                BallDetector(str(ball_model_path), device=self.device)
                if ball_model_path is not None
                else ClassicalBallDetector()
            )
        self.ball_detector = ball_detector

    def run(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
        step: int = 1,
    ) -> TrackingResult:
        """Track frames start_frame..end_frame (exclusive; None = to the end),
        every step-th one."""
        from courtshot.io.video import VideoReader

        player_frames: list[PlayerFrame] = []
        ball_frames: list[BallFrame] = []
        H = np.array(self.calibration.homography_matrix) if self.calibration is not None else None
        if H is None:
            logger.warning("No court calibration: player positions will be None")

        with VideoReader(self.video_path) as reader:
            total_frames = reader.frame_count

            # A fixed camera has a stable background: give the ball model the
            # median of frames spread over the video instead of letting it use
            # the first frame (which has players and maybe the ball in it).
            # (The classical detector needs none.)
            if isinstance(self.ball_detector, BallDetector):
                background = self._median_background(reader, start_frame, end_frame)
                if background is not None:
                    self.ball_detector.set_background(background)

            # TrackNet and the players are independent, so the ball model runs
            # on a worker thread while the main thread handles the players.
            # Each frame's ball result is collected before the next frame
            # starts, so the output is the same as running them in turn.
            # The classical detector takes a few ms and uses the player boxes,
            # so it runs after the players, on the main thread.
            classical = isinstance(self.ball_detector, ClassicalBallDetector)
            processed = 0
            ball_pool = (
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="courtshot-ball")
                if use_threads() and not classical
                else None
            )
            try:
                for frame_id, timestamp_ms, frame in reader.frames(
                    start_frame=start_frame, end_frame=end_frame, step=step
                ):
                    ball_future = None
                    if ball_pool is not None:
                        ball_future = ball_pool.submit(self.ball_detector.update, frame, frame_id, timestamp_ms)

                    try:
                        players = self.player_detector.detect_and_track(
                            frame, frame_id, timestamp_ms, homography_matrix=H
                        )
                    except BaseException:
                        if ball_future is not None:
                            ball_future.exception()  # wait: never leave the worker running
                        raise
                    player_frames.extend(players)

                    if ball_future is not None:
                        ball_frames.extend(ball_future.result())
                    elif classical:
                        ball_frames.extend(
                            self.ball_detector.update(frame, frame_id, timestamp_ms, players=[p.bbox for p in players])
                        )
                    else:
                        ball_frames.extend(self.ball_detector.update(frame, frame_id, timestamp_ms))

                    processed += 1
                    if processed % 500 == 0:
                        logger.info(
                            "Processed %d / %d frames (%.1f%%)",
                            processed,
                            total_frames,
                            100.0 * processed / total_frames,
                        )
            finally:
                if ball_pool is not None:
                    ball_pool.shutdown(wait=True)

            ball_frames.extend(self.ball_detector.flush())

        logger.info("Tracking complete: %d frames processed", processed)
        return TrackingResult(
            player_frames=player_frames,
            ball_frames=ball_frames,
            calibration=self.calibration,
        )

    @staticmethod
    def _median_background(
        reader, start_frame: int = 0, end_frame: int | None = None, samples: int = 30
    ) -> np.ndarray | None:
        """Per-pixel median of frames spread evenly over the processed span."""
        end = end_frame or reader.frame_count
        if end - start_frame < samples:
            return None
        picks = np.linspace(start_frame, end - 1, samples).astype(int)
        frames = [f for f in (reader.read_frame(int(i)) for i in picks) if f is not None]
        if len(frames) < samples // 2:
            return None
        return np.median(np.stack(frames), axis=0).astype(np.uint8)
