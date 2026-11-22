"""
Video -> raw tracking: the players (boxes, tracker IDs) and the ball, frame by
frame. This is the GPU-bound part of CourtShot; its result is cached
(courtshot.tracking.cache) and everything else works from that.

What makes it fast:

  * the people detector takes the frames in batches, and in video faster than
    PLAYER_FPS only every second (third, ...) frame; the players in the
    frames between are placed on the line from one detection to the next
  * the ball model takes its frames and windows in batches (and can run a
    window every few frames instead of every frame: ball_stride)
  * decoding, resizing, the ball model and the people detector each have
    their own thread, so none of them waits for another
  * no pose here: the players' keypoints are only needed around the ball's
    events and are made afterwards, for those moments (courtshot.analysis.poses)

Original author: Xingnan Zhu.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from courtshot.schemas.tracking import BallFrame, CourtCalibration, PlayerFrame
from courtshot.tracking.ball import BallDetector
from courtshot.tracking.device import detect_device, use_threads
from courtshot.tracking.player import PlayerDetector, YoloPersonTracker, players_between

logger = logging.getLogger(__name__)

# People are detected at about this many frames a second. The tracker and the
# identity step were set up on 25-30 fps video, and a player does not get far
# in a thirtieth of a second.
PLAYER_FPS = 30.0
# The ball model runs every this many frames (see BallDetector). At 1 every
# frame is in eight windows. 2 halves the model's work and finds the ball in as
# many frames, in the same place in 99.6% of them on the two check clips, but
# the hits and shot types read from the track are touchy: about one hit in ten
# came out differently. So 1 unless asked.
BALL_STRIDE = 1
BATCH = 8  # frames the people detector takes at once


def player_step(fps: float, player_fps: float = PLAYER_FPS) -> int:
    """People are detected in every this-many-th frame of a video at ``fps``."""
    return max(1, round(fps / player_fps)) if fps > 0 and player_fps > 0 else 1


@dataclass
class TrackingResult:
    """What tracking a video gives."""

    player_frames: list[PlayerFrame] = field(default_factory=list)
    ball_frames: list[BallFrame] = field(default_factory=list)
    calibration: CourtCalibration | None = None


class _Players:
    """The player side of the pipeline. It is given every frame in order (the
    image only for the frames the detector should see), runs the detector on
    batches of them and fills in the frames between.

    With ``threaded``, the tracker works on a thread of its own: it needs no
    GPU, so it follows the detections of one batch while the detector is
    already busy with the next.
    """

    def __init__(self, detector: PlayerDetector, H: np.ndarray | None, batch: int, threaded: bool = False) -> None:
        self.detector = detector
        self.H = H
        self.batch = batch
        self.out: list[PlayerFrame] = []
        self._waiting: list[tuple[int, float, np.ndarray | None]] = []  # frames not yet through the detector
        self._n_images = 0
        self._skipped: list[tuple[int, float]] = []  # frames after the last detected one, waiting for the next
        self._last: tuple[float, list[PlayerFrame]] | None = None  # the last detected frame: (timestamp, players)
        self._to_tracker: queue.Queue | None = queue.Queue(maxsize=4) if threaded else None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._closed = False

    def add(self, frame_id: int, timestamp_ms: float, image: np.ndarray | None) -> None:
        self._waiting.append((frame_id, timestamp_ms, image))
        if image is not None:
            self._n_images += 1
            if self._n_images >= self.batch:
                self._detect()

    def finish(self) -> list[PlayerFrame]:
        self._detect()
        if self._thread is not None:
            self._hand_over(None)
            self._thread.join()
            if self._error is not None:
                raise self._error
        # frames after the last one the detector saw: the players stay where they were
        for frame_id, timestamp_ms in self._skipped:
            if self._last is not None:
                self.out.extend(players_between(self._last[1], self._last[1], frame_id, timestamp_ms, 0.0))
        self._skipped = []
        return self.out

    def close(self) -> None:
        """Let the tracker thread go (after an error elsewhere)."""
        self._closed = True

    def _detect(self) -> None:
        """Run the detector on the waiting frames and pass them on to the tracker."""
        waiting, self._waiting, self._n_images = self._waiting, [], 0
        images = [img for _, _, img in waiting if img is not None]
        if not images:
            self._hand_over([(f, t, None) for f, t, _ in waiting])
            return
        shape = images[0].shape
        found = self.detector.detect_batch(images) if hasattr(self.detector, "detect_batch") else None
        if found is None:  # a detector that only does both steps at once
            seen = [(f, t) for f, t, img in waiting if img is not None]
            found = self.detector.detect_and_track_batch(images, [f for f, _ in seen], [t for _, t in seen], self.H)
            shape = None
        results = iter(found)
        self._hand_over([(f, t, None if img is None else (next(results), shape)) for f, t, img in waiting])

    def _hand_over(self, work: list | None) -> None:
        if self._to_tracker is None:
            if work is not None:
                self._track(work)
            return
        if self._thread is None:
            self._thread = threading.Thread(target=self._follow, name="courtshot-tracker", daemon=True)
            self._thread.start()
        while True:
            if self._error is not None:
                raise self._error
            try:
                self._to_tracker.put(work, timeout=0.2)
                return
            except queue.Full:
                pass

    def _follow(self) -> None:
        try:
            while not self._closed:
                try:
                    work = self._to_tracker.get(timeout=0.2)
                except queue.Empty:
                    continue
                if work is None:
                    return
                self._track(work)
        except BaseException as exc:  # raised by the thread that hands over
            self._error = exc

    def _track(self, work: list) -> None:
        """Tracker IDs for a batch of frames, and the frames between."""
        for frame_id, timestamp_ms, found in work:
            if found is None:
                self._skipped.append((frame_id, timestamp_ms))
                continue
            result, shape = found
            if shape is None:  # already tracked
                players = result
            else:
                players = self.detector.track_detected(result, shape, frame_id, timestamp_ms, self.H)
            if self._last is not None:
                t0, before = self._last
                span = timestamp_ms - t0
                for k, (fid, ts) in enumerate(self._skipped):
                    share = (ts - t0) / span if span > 0 else (k + 1) / (len(self._skipped) + 1)
                    self.out.extend(players_between(before, players, fid, ts, min(1.0, max(0.0, share))))
            self.out.extend(players)
            self._last = (timestamp_ms, players)
            self._skipped = []


class TrackingPipeline:
    """Runs the player and ball detectors over a video.

    Without a calibration the players have no court position (and nobody is
    filtered out for standing outside the court).

    player_fps:  people are detected at about this rate; in faster video the
                 frames between come from the detections around them
    ball_stride: the ball model runs every this many frames (1..8)
    """

    def __init__(
        self,
        video_path: str | Path,
        ball_model_path: str | Path,
        calibration: CourtCalibration | None = None,
        max_players: int = 4,
        device: str | None = None,
        player_detector: PlayerDetector | None = None,
        ball_detector: BallDetector | None = None,
        player_fps: float = PLAYER_FPS,
        ball_stride: int = BALL_STRIDE,
        batch: int = BATCH,
    ) -> None:
        self.video_path = Path(video_path)
        self.calibration = calibration
        self.device = device or detect_device()
        self.player_fps = player_fps
        self.batch = max(1, batch)
        logger.info("Inference device: %s", self.device)

        self.player_detector = player_detector or PlayerDetector(
            tracker=YoloPersonTracker(device=self.device), max_players=max_players
        )
        self.ball_detector = ball_detector or BallDetector(str(ball_model_path), device=self.device, stride=ball_stride)

    def run(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
        step: int = 1,
    ) -> TrackingResult:
        """Track frames start_frame..end_frame (exclusive; None = to the end),
        every step-th one."""
        from courtshot.io.video import VideoReader

        H = np.array(self.calibration.homography_matrix) if self.calibration is not None else None
        if H is None:
            logger.warning("No court calibration: player positions will be None")

        with VideoReader(self.video_path) as reader:
            total = len(range(start_frame, min(end_frame or reader.frame_count, reader.frame_count), step))
            every = player_step(reader.fps / step, self.player_fps)
            logger.info(
                "%d frames at %.2f fps; people detected in every %s frame, ball model every %d frames",
                total,
                reader.fps / step,
                {1: "", 2: "2nd", 3: "3rd"}.get(every, f"{every}th"),
                getattr(self.ball_detector, "stride", 1),
            )

            # A fixed camera has a stable background: give the ball model the
            # median of frames spread over the video instead of letting it use
            # the first frame (which has players and maybe the ball in it).
            background = self._median_background(reader, start_frame, end_frame)
            if background is not None:
                self.ball_detector.set_background(background)

            players = _Players(self.player_detector, H, self.batch, threaded=use_threads())
            ball_frames: list[BallFrame] = []
            frames = reader.frames(start_frame=start_frame, end_frame=end_frame, step=step)
            progress = _Progress(total)
            if use_threads():
                try:
                    self._run_threaded(frames, every, players, ball_frames, progress)
                except BaseException:
                    players.close()
                    raise
            else:
                for n, (frame_id, timestamp_ms, frame) in enumerate(frames):
                    ball_frames.extend(self.ball_detector.update(frame, frame_id, timestamp_ms))
                    players.add(frame_id, timestamp_ms, frame if n % every == 0 else None)
                    progress.step()
            ball_frames.extend(self.ball_detector.flush())
            player_frames = players.finish()

        logger.info("Tracking complete: %d frames in %s", progress.done, _hms(progress.elapsed))
        return TrackingResult(player_frames=player_frames, ball_frames=ball_frames, calibration=self.calibration)

    def _run_threaded(self, frames, every: int, players: _Players, ball_frames: list[BallFrame], progress) -> None:
        """One thread decodes, a few resize for the ball model, one runs the
        ball model, and this one runs the people detector. Each model still
        gets every frame in order, so the result is the same as in turn."""
        prepare = getattr(self.ball_detector, "prepare", None) if hasattr(self.ball_detector, "push_many") else None
        if prepare is not None:
            first = next(frames, None)
            if first is None:
                return
            prepare(first[2])  # loads the ball model here, before any other thread asks for it
            frames = _chain_first(first, frames)

        stop = threading.Event()
        errors: list[BaseException] = []
        to_ball: queue.Queue = queue.Queue(maxsize=4 * self.batch)
        to_players: queue.Queue = queue.Queue(maxsize=4 * self.batch)
        resizers = ThreadPoolExecutor(max_workers=3, thread_name_prefix="courtshot-resize")

        def put(q: queue.Queue, item) -> None:
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.2)
                    return
                except queue.Full:
                    pass

        def guarded(work):
            def run() -> None:
                try:
                    work()
                except BaseException as exc:  # reported by the main thread
                    errors.append(exc)
                    stop.set()

            return run

        def decode() -> None:
            try:
                for n, (frame_id, timestamp_ms, frame) in enumerate(frames):
                    if stop.is_set():
                        return
                    small = resizers.submit(prepare, frame) if prepare is not None else frame
                    put(to_ball, (frame_id, timestamp_ms, frame.shape, small))
                    put(to_players, (frame_id, timestamp_ms, frame if n % every == 0 else None))
            finally:
                put(to_ball, None)
                put(to_players, None)

        def ball() -> None:
            ended = False
            while not stop.is_set() and not ended:
                try:
                    item = to_ball.get(timeout=0.2)
                except queue.Empty:
                    continue
                # whatever else is waiting goes along, up to a batch: the
                # model takes several frames at once much better than one by one
                items = []
                while True:
                    if item is None:
                        ended = True
                        break
                    items.append(item)
                    if len(items) >= self.batch:
                        break
                    try:
                        item = to_ball.get_nowait()
                    except queue.Empty:
                        break
                if not items:
                    continue
                if prepare is not None:
                    ball_frames.extend(
                        self.ball_detector.push_many(
                            [small.result() for _, _, _, small in items],
                            items[0][2],
                            [frame_id for frame_id, _, _, _ in items],
                            [timestamp_ms for _, timestamp_ms, _, _ in items],
                        )
                    )
                else:
                    for frame_id, timestamp_ms, _, frame in items:
                        ball_frames.extend(self.ball_detector.update(frame, frame_id, timestamp_ms))

        threads = [
            threading.Thread(target=guarded(decode), name="courtshot-decode", daemon=True),
            threading.Thread(target=guarded(ball), name="courtshot-ball", daemon=True),
        ]
        for t in threads:
            t.start()
        try:
            while not stop.is_set():
                try:
                    item = to_players.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                players.add(*item)
                progress.step()
        except BaseException:
            stop.set()
            raise
        finally:
            for t in threads:
                t.join()
            resizers.shutdown(wait=True)
        if errors:
            raise errors[0]

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


def _chain_first(first, rest):
    yield first
    yield from rest


def _hms(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class _Progress:
    """A log line every 20 seconds: how far, how fast, how long still."""

    EVERY_S = 20.0

    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self._start = self._last = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def step(self) -> None:
        self.done += 1
        now = time.perf_counter()
        if now - self._last < self.EVERY_S:
            return
        self._last = now
        rate = self.done / max(now - self._start, 1e-9)
        left = max(0, self.total - self.done) / max(rate, 1e-9)
        logger.info(
            "Processed %d / %d frames (%.1f%%), %.0f frames/s, about %s left",
            self.done,
            self.total,
            100.0 * self.done / max(1, self.total),
            rate,
            _hms(left),
        )
