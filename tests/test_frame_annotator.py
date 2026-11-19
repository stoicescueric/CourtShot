"""Tests for drawing on video frames."""

import numpy as np

from padex.schemas.tracking import (
    BallFrame,
    BoundingBox,
    CourtCalibration,
    PlayerFrame,
    PoseKeypoint,
    Position2D,
)
from padex.viz.frame import FrameAnnotator


def _frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def _make_pf(player_id="P_1", team_id="T_1", keypoints=()):
    return PlayerFrame(
        frame_id=0,
        timestamp_ms=0.0,
        player_id=player_id,
        team_id=team_id,
        bbox=BoundingBox(x1=100, y1=200, x2=200, y2=400),
        position=Position2D(x=5.0, y=10.0),
        confidence=0.9,
        keypoints=list(keypoints),
    )


def _make_bf():
    return BallFrame(
        frame_id=0,
        timestamp_ms=0.0,
        bbox=BoundingBox(x1=300, y1=300, x2=320, y2=320),
        confidence=0.8,
    )


def _make_calibration(scale: float = 0.02):
    # pixels -> metres by a plain scale: the court is 500 x 1000 px at 0.02
    return CourtCalibration(
        frame_width=1280,
        frame_height=720,
        homography_matrix=[[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        court_keypoints_px=[(100, 100), (500, 100), (100, 650), (500, 650)],
        court_keypoints_m=[(0, 0), (10, 0), (0, 20), (10, 20)],
        reprojection_error=1.5,
    )


class TestFrameAnnotator:
    def setup_method(self):
        self.annotator = FrameAnnotator()

    def test_draw_player_bboxes(self):
        frame = _frame()
        self.annotator.draw_player_bboxes(frame, [_make_pf()])
        assert frame[200:400, 100:200, :].sum() > 0

    def test_teams_get_different_colours(self):
        f1, f2 = _frame(), _frame()
        self.annotator.draw_player_bboxes(f1, [_make_pf("P_1", "T_1")])
        self.annotator.draw_player_bboxes(f2, [_make_pf("P_3", "T_2")])
        assert not np.array_equal(f1[200, 150], f2[200, 150])  # on the top edge of the box

    def test_draw_pose_keypoints(self):
        kps = [
            PoseKeypoint(name="left_shoulder", x=120, y=240, confidence=0.9),
            PoseKeypoint(name="left_elbow", x=120, y=300, confidence=0.9),
            PoseKeypoint(name="left_wrist", x=120, y=360, confidence=0.1),
        ]  # too unsure to draw
        frame = _frame()
        self.annotator.draw_pose_keypoints(frame, [_make_pf(keypoints=kps)])
        assert frame[260:280, 115:125].sum() > 0  # the upper arm
        assert frame[320:370, 110:130].sum() == 0  # nothing towards the unsure wrist

    def test_draw_ball(self):
        frame = _frame()
        self.annotator.draw_ball(frame, _make_bf())
        assert frame[300:320, 300:320, :].sum() > 0

    def test_draw_ball_none_or_not_found(self):
        frame = _frame()
        self.annotator.draw_ball(frame, None)
        self.annotator.draw_ball(frame, BallFrame(frame_id=0, timestamp_ms=0.0))
        assert frame.sum() == 0

    def test_draw_court_lines(self):
        frame = _frame()
        self.annotator.draw_court_lines(frame, _make_calibration())
        assert frame.sum() > 0
        assert frame[500, 100:400].sum() > 0  # the net, at y = 10 m = 500 px

    def test_court_lines_follow_the_calibration_given(self):
        small, large = _frame(), _frame()
        self.annotator.draw_court_lines(small, _make_calibration(0.02))
        self.annotator.draw_court_lines(large, _make_calibration(0.04))  # half the size in pixels
        assert not np.array_equal(small, large)

    def test_draw_court_lines_none(self):
        frame = _frame()
        self.annotator.draw_court_lines(frame, None)
        assert frame.sum() == 0

    def test_draw_court_keypoints(self):
        frame = _frame()
        self.annotator.draw_court_keypoints(frame, _make_calibration())
        assert frame[95:105, 95:105].sum() > 0

    def test_draw_mini_court(self):
        frame = _frame()
        self.annotator.draw_mini_court(frame, [_make_pf()], ball_court=(5.0, 10.0))
        assert frame.sum() > 0

    def test_draw_stats_panel(self):
        frame = _frame()
        self.annotator.draw_stats_panel(frame, {"Shots": 5, "Ball": "80 km/h"})
        assert frame[:100, :260, :].sum() > 0

    def test_draw_stats_panel_empty(self):
        frame = _frame()
        self.annotator.draw_stats_panel(frame, {})
        assert frame.sum() == 0

    def test_draw_shot_label(self):
        frame = _frame()
        self.annotator.draw_shot_label(frame, "VOLLEY", _make_bf())
        assert frame.sum() > 0

    def test_no_shot_label_without_a_label_or_a_ball(self):
        frame = _frame()
        self.annotator.draw_shot_label(frame, None, _make_bf())
        self.annotator.draw_shot_label(frame, "VOLLEY", None)
        assert frame.sum() == 0

    def test_draw_bounces(self):
        frame = _frame()
        self.annotator.draw_bounces(frame, [((400.0, 300.0), "ground", 0.0), ((600.0, 200.0), "wall", 0.8)])
        assert frame[280:320, 380:420].sum() > 0
        assert frame[180:220, 580:620].sum() > 0
        # ground marks are green, wall marks red (BGR)
        ground, wall = frame[280:320, 380:420], frame[180:220, 580:620]
        assert ground[..., 1].sum() > ground[..., 2].sum()
        assert wall[..., 2].sum() > wall[..., 1].sum()

    def test_draw_frame_number(self):
        frame = _frame()
        self.annotator.draw_frame_number(frame, 42)
        assert frame[:40, :200, :].sum() > 0

    def test_annotate_frame_all(self):
        frame = _frame()
        result = self.annotator.annotate_frame(
            frame=frame,
            frame_id=42,
            player_frames=[_make_pf("P_1", "T_1"), _make_pf("P_3", "T_2")],
            ball_frame=_make_bf(),
            calibration=_make_calibration(),
            shot_label="BANDEJA / VIBORA",
            stats={"Shots": 5},
            bounces=[((400.0, 300.0), "ground", 0.2)],
            ball_speed_kmh=80.0,
            ball_court=(5.0, 12.0),
        )
        assert result is frame
        assert frame.sum() > 0

    def test_annotate_frame_minimal(self):
        frame = _frame()
        result = self.annotator.annotate_frame(frame=frame, frame_id=0, player_frames=[])
        assert result is frame
        assert frame.sum() > 0  # frame number and mini court
