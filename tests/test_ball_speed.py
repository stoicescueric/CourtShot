"""
Tests for the ball speed shown on the tracked video: the per-frame speeds
from the fitted 3D flights, and the label drawn next to the ball.
"""

from __future__ import annotations

import numpy as np

from courtshot.analysis import ball_flight as ball3d
from courtshot.schemas.tracking import BallFrame, BoundingBox
from courtshot.viz.frame import FrameAnnotator

FPS = 25.0


def _flight(t0: float, t1: float, speed_ms: float, ok: bool = True) -> ball3d.Flight:
    # launched level, 2 m up, along the court
    params = np.r_[5.0, 4.0, 2.0, 0.0, speed_ms, 0.0, ball3d.DEFAULTS]
    return ball3d.Flight(t0, params, np.array([]), np.empty((0, 2)), 1.0, 0.1, t1, FPS, ok=ok)


def _speeds(*flights) -> dict[int, float]:
    return ball3d.frame_speeds(ball3d.Ball3DResult(None, list(flights)), FPS)


class TestFrameSpeeds:
    def test_starts_at_the_launch_speed_and_slows_down(self):
        sp = _speeds(_flight(1.0, 1.6, 20.0))  # 72 km/h
        assert sorted(sp) == list(range(25, 41))
        assert 66.0 < sp[25] <= 72.5
        assert sp[25] > sp[30] > sp[35]  # air drag
        assert all(40.0 < v < 73.0 for v in sp.values())

    def test_poor_fits_give_no_speed(self):
        assert _speeds(_flight(1.0, 1.6, 20.0, ok=False)) == {}

    def test_no_speed_between_flights(self):
        sp = _speeds(_flight(1.0, 1.4, 20.0), _flight(2.0, 2.4, 10.0))
        assert 40 not in sp and 45 not in sp
        assert sp[50] < 40.0 < sp[25]  # each frame takes its own flight's speed

    def test_a_hit_does_not_leak_into_the_next_flight(self):
        # 108 km/h, then a soft 18 km/h shot straight after: no frame may average the two
        sp = _speeds(_flight(1.0, 1.4, 30.0), _flight(1.4, 2.0, 5.0))
        assert sp[36] < 30.0 and sp[34] > 60.0

    def test_empty(self):
        assert _speeds() == {}


class TestSpeedLabel:
    def _ball(self, x: float, y: float) -> BallFrame:
        return BallFrame(
            frame_id=0, timestamp_ms=0.0, bbox=BoundingBox(x1=x - 4, y1=y - 4, x2=x + 4, y2=y + 4), confidence=0.9
        )

    def test_label_is_drawn_next_to_the_ball(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        FrameAnnotator().draw_ball_speed(frame, self._ball(600, 300), 87.4)
        ys, xs = np.nonzero(frame.any(axis=2))
        assert len(xs) > 50
        assert xs.min() >= 600 and xs.max() < 800 and ys.min() > 290 and ys.max() < 345

    def test_label_stays_inside_the_frame_at_the_edges(self):
        for x, y in ((1275, 300), (5, 5), (1275, 715), (640, 716)):
            frame = np.zeros((720, 1280, 3), np.uint8)
            FrameAnnotator().draw_ball_speed(frame, self._ball(x, y), 123.0)
            assert frame.any()  # drawn, and cv2 did not have to clip it away entirely

    def test_nothing_drawn_without_a_speed_or_without_the_ball(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        ann = FrameAnnotator()
        ann.draw_ball_speed(frame, self._ball(600, 300), None)
        ann.draw_ball_speed(frame, None, 80.0)
        ann.draw_ball_speed(frame, BallFrame(frame_id=0, timestamp_ms=0.0), 80.0)
        assert not frame.any()

    def test_annotate_frame_takes_the_speed(self):
        a = np.zeros((720, 1280, 3), np.uint8)
        b = np.zeros((720, 1280, 3), np.uint8)
        FrameAnnotator().annotate_frame(a, 0, [], self._ball(600, 300))
        FrameAnnotator().annotate_frame(b, 0, [], self._ball(600, 300), ball_speed_kmh=95.0)
        assert (a != b).any()
