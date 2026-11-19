"""Tests for the mini court overlay."""

import numpy as np

from padex.schemas.tracking import BoundingBox, PlayerFrame, Position2D
from padex.viz.mini_court import MiniCourt


def _frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def _make_pf(player_id="P_1", team_id="T_1", x=5.0, y=10.0):
    return PlayerFrame(
        frame_id=0,
        timestamp_ms=0.0,
        player_id=player_id,
        team_id=team_id,
        bbox=BoundingBox(x1=100, y1=200, x2=200, y2=400),
        position=Position2D(x=x, y=y),
        confidence=0.9,
    )


class TestMiniCourt:
    def setup_method(self):
        self.mc = MiniCourt(width_px=150, height_px=300, margin=10)

    def test_draw_modifies_frame(self):
        frame = _frame()
        self.mc.draw(frame, [_make_pf()])
        assert frame.sum() > 0

    def test_draw_no_players(self):
        frame = _frame()
        self.mc.draw(frame, [])
        assert frame.sum() > 0  # the court itself

    def test_ball_is_drawn_only_when_its_position_is_known(self):
        with_ball, without = _frame(), _frame()
        self.mc.draw(with_ball, [], ball_court=(2.5, 5.0))
        self.mc.draw(without, [], ball_court=None)
        assert not np.array_equal(with_ball, without)

    def test_ball_outside_the_court_is_drawn_at_its_edge(self):
        frame = _frame()
        self.mc.draw(frame, [], ball_court=(-3.0, 25.0))  # must not draw outside the overlay
        assert frame[:, :1100].sum() == 0

    def test_draw_multiple_players(self):
        frame = _frame()
        pfs = [
            _make_pf("P_1", "T_1", 2.0, 5.0),
            _make_pf("P_2", "T_1", 8.0, 5.0),
            _make_pf("P_3", "T_2", 2.0, 15.0),
            _make_pf("P_4", "T_2", 8.0, 15.0),
        ]
        self.mc.draw(frame, pfs)
        assert frame.sum() > 0

    def test_player_without_a_position_is_skipped(self):
        pf = _make_pf().model_copy(update={"position": None})
        a, b = _frame(), _frame()
        self.mc.draw(a, [pf])
        self.mc.draw(b, [])
        assert np.array_equal(a, b)

    def test_top_right_position(self):
        frame = _frame()
        self.mc.draw(frame, [_make_pf()], position="top_right")
        assert frame[:320, 1100:, :].sum() > 0
        assert frame[400:, :, :].sum() == 0

    def test_court_to_mini_center(self):
        assert self.mc._court_to_mini(5.0, 10.0) == (75, 150)

    def test_court_to_mini_origin(self):
        assert self.mc._court_to_mini(0.0, 0.0) == (0, 300)  # bottom: y is flipped
