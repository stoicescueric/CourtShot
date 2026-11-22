"""
Tests for what makes tracking fast without changing its result: frames taken
in batches, people detected in every second frame of fast video with the
frames between filled in, and the poses made afterwards for the moments the
ball analysis asks about. The models are replaced by stand-ins.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from courtshot.analysis import poses
from courtshot.io.video import FramePicker
from courtshot.schemas.tracking import BoundingBox, PlayerFrame, PoseKeypoint, Position2D
from courtshot.tracking.pipeline import TrackingPipeline, _Players, player_step
from courtshot.tracking.player import letterbox, players_between

FPS = 60.0


def _player(frame_id: int, x: float, player_id: str = "P_001", conf: float = 0.8) -> PlayerFrame:
    return PlayerFrame(
        frame_id=frame_id,
        timestamp_ms=1000.0 * frame_id / FPS,
        player_id=player_id,
        bbox=BoundingBox(x1=x, y1=10.0, x2=x + 20.0, y2=70.0),
        position=Position2D(x=x / 100.0, y=5.0),
        confidence=conf,
    )


class _MovingDetector:
    """Stands in for PlayerDetector: one tracked player whose box is at
    x = 10 * frame, and one untracked detection. Notes the frames it is shown."""

    def __init__(self):
        self.batches: list[list[int]] = []

    def detect_and_track_batch(self, frames, frame_ids, timestamps_ms, homography_matrix=None):
        self.batches.append(list(frame_ids))
        return [[_player(f, 10.0 * f), _player(f, 500.0, "P_det_001")] for f in frame_ids]


class _NoBall:
    def set_background(self, frame):
        pass

    def update(self, frame, frame_id, timestamp_ms):
        return []

    def flush(self):
        return []


class TestPlayerStep:
    def test_every_frame_up_to_30_fps_and_every_second_one_at_50_and_60(self):
        assert [player_step(fps) for fps in (23.976, 25, 30, 50, 59.94, 60, 120)] == [1, 1, 1, 2, 2, 2, 4]


class TestLetterbox:
    def test_1080p_becomes_what_the_detector_takes(self):
        frame = np.zeros((1080, 1920, 3), np.uint8)
        image, scale, left, top = letterbox(frame, 1280)
        assert image.shape == (736, 1280, 3) and (left, top) == (0, 8)
        assert scale == pytest.approx(1280 / 1920)
        assert image[0, 0, 0] == 114 and image[top, 0, 0] == 0  # grey bar above the picture


class TestFramesBetween:
    def test_a_tracked_player_is_placed_on_the_line_between_two_detections(self):
        (mid,) = players_between([_player(10, 100.0)], [_player(12, 120.0, conf=0.6)], 11, 183.3, 0.5)
        assert (mid.frame_id, mid.timestamp_ms, mid.player_id) == (11, 183.3, "P_001")
        assert (mid.bbox.x1, mid.bbox.x2) == (110.0, 130.0)
        assert mid.position.x == pytest.approx(1.1) and mid.confidence == pytest.approx(0.7)

    def test_only_players_in_both_frames_and_none_without_a_tracker_id(self):
        before = [_player(10, 100.0), _player(10, 300.0, "P_002"), _player(10, 500.0, "P_det_000")]
        after = [_player(12, 120.0), _player(12, 500.0, "P_det_000")]
        assert [p.player_id for p in players_between(before, after, 11, 0.0, 0.5)] == ["P_001"]

    @pytest.mark.parametrize("threaded", [False, True])
    def test_detector_sees_every_second_frame_and_every_frame_gets_its_players(self, threaded):
        detector = _MovingDetector()
        players = _Players(detector, None, batch=4, threaded=threaded)
        for f in range(11):  # frames 0..10: the detector's are 0, 2, .., 10
            players.add(f, 1000.0 * f / FPS, np.zeros((4, 4, 3), np.uint8) if f % 2 == 0 else None)
        players.add(11, 1000.0 * 11 / FPS, None)  # nothing after it to draw the line to
        out = players.finish()
        assert [f for batch in detector.batches for f in batch] == [0, 2, 4, 6, 8, 10]
        tracked = [p for p in out if p.player_id == "P_001"]
        assert [p.frame_id for p in tracked] == list(range(12))
        assert [p.bbox.x1 for p in tracked[:11]] == pytest.approx([10.0 * f for f in range(11)])
        assert tracked[11].bbox.x1 == 100.0  # after the last detection: where the player was
        assert [p.frame_id for p in out if p.player_id == "P_det_001"] == [0, 2, 4, 6, 8, 10]

    def test_an_error_in_the_tracker_thread_comes_out(self):
        class Broken(_MovingDetector):
            def detect_batch(self, frames):
                return [object() for _ in frames]

            def track_detected(self, *args):
                raise RuntimeError("tracker broke")

        players = _Players(Broken(), None, batch=2, threaded=True)
        with pytest.raises(RuntimeError, match="tracker broke"):
            for f in range(40):
                players.add(f, 0.0, np.zeros((4, 4, 3), np.uint8))
            players.finish()


def _write_video(path, n_frames: int, fps: float = FPS) -> None:
    """Frames whose brightness says which frame they are."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n_frames):
        writer.write(np.full((48, 64, 3), 10 + 4 * i, np.uint8))
    writer.release()


class TestPipelineOnAVideo:
    @pytest.mark.parametrize("threads", ["1", "0"])
    def test_60_fps_video_goes_to_the_detector_every_second_frame(self, tmp_path, monkeypatch, threads):
        monkeypatch.setenv("COURTSHOT_THREADS", threads)
        video = tmp_path / "clip.mp4"
        _write_video(video, 20)
        detector = _MovingDetector()
        result = TrackingPipeline(
            video, "unused.pt", device="cpu", player_detector=detector, ball_detector=_NoBall(), batch=4
        ).run()
        assert [f for batch in detector.batches for f in batch] == list(range(0, 20, 2))
        assert all(len(batch) <= 4 for batch in detector.batches)
        assert [p.frame_id for p in result.player_frames if p.player_id == "P_001"] == list(range(20))


def _brightness(path) -> list[float]:
    """Mean brightness of every frame, read one after the other."""
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(float(frame.mean()))
    cap.release()
    return out


class TestFramePicker:
    def test_gives_the_frames_asked_for(self, tmp_path):
        video = tmp_path / "clip.mp4"
        _write_video(video, 60)
        every = _brightness(video)
        with FramePicker(video) as picker:
            got = [float(picker.read(f).mean()) for f in (0, 3, 4, 30, 59)]
            assert picker.read(60) is None
        assert got == [every[f] for f in (0, 3, 4, 30, 59)]

    def test_frames_must_be_asked_for_in_order(self, tmp_path):
        video = tmp_path / "clip.mp4"
        _write_video(video, 10)
        with FramePicker(video) as picker:
            picker.read(5)
            with pytest.raises(ValueError):
                picker.read(2)


class _PoseByFrame:
    """Stands in for YoloPoseEstimator: the full-frame pass finds nobody, the
    close-up puts every keypoint at x = the frame's brightness."""

    def __init__(self):
        self.closeups = 0

    def predict_full_frames(self, frames):
        return [None for _ in frames]

    def assign(self, bboxes, full_frame):
        return [[] for _ in bboxes]

    def estimate_closeups(self, views):
        self.closeups += len(views)
        return [[PoseKeypoint(name="left_wrist", x=float(frame.mean()), y=1.0, confidence=0.9)] for frame, _ in views]


class TestPosesAfterTracking:
    def _stable(self, n: int):
        from types import SimpleNamespace

        frames = [_player(f, 10.0, "P_1") for f in range(n)] + [_player(f, 30.0, "P_2") for f in range(n)]
        return SimpleNamespace(player_frames=frames)

    def test_only_the_wanted_players_and_frames_get_a_pose(self, tmp_path):
        video = tmp_path / "clip.mp4"
        _write_video(video, 30, fps=25.0)
        estimator = _PoseByFrame()
        wanted = {f: ["P_1"] for f in range(10, 14)}
        data = poses.extract(video, self._stable(30), wanted, 25.0, estimator=estimator)
        assert list(data["frames"]) == [10, 11, 12, 13] and set(data["players"]) == {0}
        assert estimator.closeups == 4
        lookup = poses.as_lookup(data)
        xy, conf = lookup[("P_1", 10)]
        assert conf[poses.IDX["left_wrist"]] == pytest.approx(0.9) and conf.sum() == pytest.approx(0.9)
        assert xy[poses.IDX["left_wrist"], 0] == pytest.approx(_brightness(video)[10], abs=1e-3)

    def test_in_fast_video_the_model_sees_every_second_frame_and_the_rest_lie_between(self, tmp_path):
        video = tmp_path / "clip.mp4"
        _write_video(video, 30)
        estimator = _PoseByFrame()
        wanted = {f: ["P_2"] for f in range(11, 16)}  # 11..15: needs 10, 12, 14 and 16 from the model
        data = poses.extract(video, self._stable(30), wanted, FPS, estimator=estimator)
        assert estimator.closeups == 4
        assert list(data["frames"]) == [11, 12, 13, 14, 15]
        x = data["xy"][:, poses.IDX["left_wrist"], 0]
        assert x[1] > x[0] and x[4] > x[3]  # 11 lies before 12, 15 after 14
        assert x[2] == pytest.approx((x[1] + x[3]) / 2, abs=1e-3)  # 13 is half-way from 12 to 14
