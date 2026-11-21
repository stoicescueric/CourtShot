"""
Tests for the classical (no network) ball detector on synthetic frames: a
blue court with a small bright ball flying over it.
"""

from __future__ import annotations

import numpy as np

from courtshot.tracking.ball_classical import ClassicalBallDetector

H, W = 720, 1280
COURT = (200, 60, 40)  # BGR: blue


def _frame(balls=(), squares=()) -> np.ndarray:
    """balls: (x, y) yellow dots of radius 3; squares: (x, y, half-size) white squares."""
    import cv2

    f = np.full((H, W, 3), COURT, np.uint8)
    for x, y, r in squares:
        f[y - r : y + r + 1, x - r : x + r + 1] = 255
    for x, y in balls:
        cv2.circle(f, (int(x), int(y)), 3, (60, 230, 230), -1)
    return f


def _run(frames, height=H):
    det = ClassicalBallDetector()
    out = []
    for i, f in enumerate(frames):
        if height != H:
            import cv2

            f = cv2.resize(f, (round(W * height / H), height))
        out.extend(det.update(f, i, i * 40.0))
    out.extend(det.flush())
    return out


def _flight(n=20):
    return [(200 + 15 * i, 500 - 20 * i + i * i) for i in range(n)]  # a parabola, 15-25 px per frame


class TestFlyingBall:
    def test_found_on_its_path(self):
        path = _flight()
        out = _run([_frame(balls=[p]) for p in path])
        # every frame with a neighbour on both sides is found, within 2 px
        for bf in out[1:-1]:
            x, y = path[bf.frame_id]
            assert bf.bbox is not None, bf.frame_id
            cx, cy, conf = bf.candidates[0]
            assert abs(cx - x) < 2 and abs(cy - y) < 2
            assert 0 < conf <= 1

    def test_positions_are_in_original_pixels_at_1080p(self):
        path = _flight()
        out = _run([_frame(balls=[p]) for p in path], height=1080)
        x, y = path[10]
        cx, cy, _ = out[10].candidates[0]
        assert abs(cx - 1.5 * x) < 3 and abs(cy - 1.5 * y) < 3

    def test_static_bright_spot_is_ignored(self):
        out = _run([_frame(squares=[(640, 360, 2)]) for _ in range(10)])
        assert all(bf.bbox is None for bf in out)

    def test_large_moving_bright_thing_is_ignored(self):
        # a white 41 x 41 "shirt" moving across the court
        out = _run([_frame(squares=[(300 + 20 * i, 360, 20)]) for i in range(10)])
        assert all(bf.bbox is None for bf in out)

    def test_ball_is_ranked_above_an_isolated_flicker(self):
        # one-frame flicker elsewhere: no temporal support, so the ball stays first
        path = _flight()
        frames = [_frame(balls=[p]) for p in path]
        frames[10] = _frame(balls=[path[10]], squares=[(1000, 150, 2)])
        out = _run(frames)
        cx, cy, _ = out[10].candidates[0]
        assert abs(cx - path[10][0]) < 2 and abs(cy - path[10][1]) < 2


class TestPlayers:
    def test_a_shoe_inside_a_player_box_loses_to_the_ball(self):
        from courtshot.schemas.tracking import BoundingBox

        # a bright "shoe" wobbling in place at the bottom of a player's box, and the ball in flight
        path = _flight()
        shoe = [(900 + (6 if i % 2 else -6), 600) for i in range(len(path))]
        box = BoundingBox(x1=850, y1=420, x2=950, y2=620)
        det = ClassicalBallDetector()
        for i, (p, q) in enumerate(zip(path, shoe)):
            det.update(_frame(balls=[p, q]), i, 0.0, players=[box])
        out = det.flush()
        for bf in out[2:-2]:
            cx, cy, _ = bf.candidates[0]
            x, y = path[bf.frame_id]
            assert abs(cx - x) < 2 and abs(cy - y) < 2, bf.frame_id


class TestPath:
    def test_the_path_does_not_detour_to_a_brighter_blob_far_away(self):
        # a brighter flicker 150 px off the ball's path for two frames
        path = _flight()
        frames = [_frame(balls=[p]) for p in path]
        for i in (9, 10):
            x, y = path[i]
            frames[i] = _frame(balls=[path[i]], squares=[(int(x) + 150, int(y), 2)])
        out = _run(frames)
        for i in (9, 10):
            cx, _, _ = out[i].candidates[0]
            assert abs(cx - path[i][0]) < 2


class TestStreaming:
    def test_every_frame_comes_back_once_and_in_order(self):
        out = _run([_frame(balls=[p]) for p in _flight(15)])
        assert [bf.frame_id for bf in out] == list(range(15))
        assert [bf.timestamp_ms for bf in out] == [i * 40.0 for i in range(15)]

    def test_results_come_from_flush(self):
        det = ClassicalBallDetector()
        assert all(det.update(_frame(balls=[p]), i, 0.0) == [] for i, p in enumerate(_flight(10)))
        assert len(det.flush()) == 10

    def test_short_videos(self):
        for n in (0, 1, 2, 3):
            out = _run([_frame() for _ in range(n)])
            assert [bf.frame_id for bf in out] == list(range(n))

    def test_reusable_after_flush(self):
        det = ClassicalBallDetector()
        for rnd in range(2):
            out = []
            for i, p in enumerate(_flight(10)):
                out.extend(det.update(_frame(balls=[p]), i, 0.0))
            out.extend(det.flush())
            assert [bf.frame_id for bf in out] == list(range(10)), rnd
