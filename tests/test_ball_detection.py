"""
Tests for the ball detector around the network: turning heatmaps into
detections, and averaging each frame over the windows it appears in. The
network itself is replaced by a stand-in.
"""

from __future__ import annotations

import numpy as np

from padex.tracking.ball import BALL_BOX_RADIUS, BallDetector

SHAPE = (576, 1024, 3)  # video frame: twice the network's 512 x 288


def _heatmap(*blobs: tuple[int, int, int, float]) -> np.ndarray:
    """blobs: (x, y, half-size, value) squares on an empty 512 x 288 heatmap."""
    hm = np.zeros((BallDetector.INFER_H, BallDetector.INFER_W), np.float32)
    for x, y, r, value in blobs:
        hm[y - r : y + r + 1, x - r : x + r + 1] = value
    return hm


class _ScriptedDetector(BallDetector):
    """BallDetector whose network output is given: one value per frame, and a
    window's heatmaps are filled with the values of its SEQ_LEN frames."""

    def __init__(self, values: list[float]):
        super().__init__("unused.pt")
        self.values = values
        self.seen = 0

    def _forward(self, frame):
        self.seen += 1
        if self.seen < self.SEQ_LEN:
            return None
        window = self.values[self.seen - self.SEQ_LEN : self.seen]
        return np.stack([_heatmap((100, 100, 2, v)) for v in window])


class TestCandidates:
    def setup_method(self):
        self.det = BallDetector("unused.pt")

    def test_nothing_above_the_threshold(self):
        assert self.det.candidates(_heatmap((100, 100, 2, 0.4)), SHAPE) == []

    def test_position_is_scaled_to_the_video_frame(self):
        ((x, y, conf),) = self.det.candidates(_heatmap((100, 50, 2, 0.9)), SHAPE)
        assert (x, y) == (200.0, 100.0)
        assert abs(conf - 0.9) < 1e-6

    def test_largest_blob_first_and_at_most_three(self):
        hm = _heatmap((50, 50, 1, 0.9), (150, 50, 3, 0.8), (250, 50, 2, 0.7), (350, 50, 0, 0.95))
        out = self.det.candidates(hm, SHAPE)
        assert [round(x / 2) for x, _, _ in out] == [150, 250, 50]  # by size; the single pixel is left out

    def test_blobs_too_large_to_be_a_ball_are_ignored(self):
        hm = _heatmap((200, 150, 10, 0.9), (50, 50, 2, 0.8))  # 21 x 21 = 441 px > MAX_BLOB_AREA
        ((x, y, _),) = self.det.candidates(hm, SHAPE)
        assert (x, y) == (100.0, 100.0)


class TestStreaming:
    def _run(self, values):
        det = _ScriptedDetector(values)
        frame = np.zeros(SHAPE, np.uint8)
        out = []
        for i in range(len(values)):
            out.extend(det.update(frame, i, i * 40.0))
        out.extend(det.flush())
        return out

    def test_every_frame_comes_back_once_and_in_order(self):
        out = self._run([0.9] * 20)
        assert [bf.frame_id for bf in out] == list(range(20))
        assert [bf.timestamp_ms for bf in out] == [i * 40.0 for i in range(20)]

    def test_results_arrive_seq_len_minus_one_frames_late(self):
        det = _ScriptedDetector([0.9] * 20)
        frame = np.zeros(SHAPE, np.uint8)
        counts = [len(det.update(frame, i, 0.0)) for i in range(20)]
        late = BallDetector.SEQ_LEN - 1
        assert counts == [0] * late + [1] * (20 - late)
        assert len(det.flush()) == late

    def test_detection_is_a_small_box_around_the_ball(self):
        bf = self._run([0.9] * 20)[10]
        assert bf.bbox is not None and bf.confidence > 0.5
        assert (bf.bbox.x1 + bf.bbox.x2) / 2 == 200.0 and (bf.bbox.y1 + bf.bbox.y2) / 2 == 200.0
        assert bf.bbox.x2 - bf.bbox.x1 == 2 * BALL_BOX_RADIUS
        assert bf.candidates[0][:2] == (200.0, 200.0)

    def test_a_frame_is_judged_by_all_windows_it_is_in(self):
        # Every window sees the same value for a given frame here, so the
        # average is that value: frames below the threshold have no ball.
        values = [0.9] * 10 + [0.2] * 3 + [0.9] * 10
        found = [bf.bbox is not None for bf in self._run(values)]
        assert found == [v > 0.5 for v in values]

    def test_fewer_frames_than_one_window(self):
        out = self._run([0.9] * 5)
        assert [bf.frame_id for bf in out] == list(range(5))
        assert all(bf.bbox is None for bf in out)

    def test_reset_starts_a_new_video(self):
        det = _ScriptedDetector([0.9] * 20)
        frame = np.zeros(SHAPE, np.uint8)
        for i in range(10):
            det.update(frame, i, 0.0)
        det.reset()
        assert det.flush() == []
