"""
Tests for the static-ball filter: it counts occupancy near in time, not over
the video. (Moved out of the reference test_reuse_and_candidates.py.)
"""

from __future__ import annotations

from courtshot.analysis import ball_events as bt


class TestDropStatic:
    FPS = 30.0
    H = 720.0

    def test_spare_ball_on_the_floor_is_dropped(self):
        det = {f: [(400.0, 500.0)] for f in range(90)}  # 3 s on one spot
        assert bt.drop_static(det, self.H, self.FPS, bt.BallSettings()) == {}

    def test_spot_crossed_in_many_rallies_is_kept(self):
        # The ball crosses the same spot for 10 frames in each of 20 rallies a
        # minute apart: 200 frames in total, far more than static_s, but never
        # more than 10 frames within the window.
        det = {}
        for rally in range(20):
            start = int(rally * 60 * self.FPS)
            for i in range(10):
                det[start + i] = [(400.0, 500.0)]
        out = bt.drop_static(det, self.H, self.FPS, bt.BallSettings())
        assert out == det

    def test_moving_ball_is_kept_next_to_a_static_one(self):
        det = {f: [(400.0, 500.0), (100.0 + 12.0 * f, 300.0)] for f in range(90)}
        out = bt.drop_static(det, self.H, self.FPS, bt.BallSettings())
        assert len(out) == 90
        assert all(ps == [(100.0 + 12.0 * f, 300.0)] for f, ps in out.items())
