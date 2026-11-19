"""Tests for the tracking data models."""

import pytest
from pydantic import ValidationError

from padex.schemas.tracking import BallFrame, PlayerFrame, Position2D


def test_position2d_valid():
    pos = Position2D(x=5.0, y=10.0)
    assert pos.x == 5.0
    assert pos.y == 10.0


def test_position2d_must_be_on_the_court():
    with pytest.raises(ValidationError):
        Position2D(x=11.0, y=10.0)


def test_player_frame():
    pf = PlayerFrame(
        frame_id=0,
        timestamp_ms=0.0,
        player_id="P1",
        bbox={"x1": 0, "y1": 0, "x2": 100, "y2": 200},
        confidence=0.95,
    )
    assert pf.player_id == "P1"
    assert pf.position is None and pf.team_id is None and pf.keypoints == []


def test_ball_frame_defaults():
    bf = BallFrame(frame_id=0, timestamp_ms=0.0)
    assert bf.bbox is None
    assert bf.confidence == 0.0
    assert bf.candidates == []
