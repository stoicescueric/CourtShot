"""
Tests for player detection: who is kept in a frame, where they stand on the
court, and how pose is attached. The YOLO models are replaced by stand-ins.
"""

from __future__ import annotations

import numpy as np

from courtshot.schemas.tracking import BoundingBox, PoseKeypoint
from courtshot.tracking.player import (
    COCO_KEYPOINT_NAMES,
    PlayerDetector,
    RawDetection,
    YoloPoseEstimator,
    foot_position,
    iou_matrix,
)

FRAME = np.zeros((8, 8, 3), np.uint8)


def _det(x: float, conf: float, track_id: int | None = None) -> RawDetection:
    """A person standing at court x (with an identity homography), feet at y = 5."""
    return RawDetection(bbox=BoundingBox(x1=x, y1=0.0, x2=x + 0.5, y2=5.0), confidence=conf, track_id=track_id)


class _FixedTracker:
    """Stands in for YoloPersonTracker: the same detections every frame."""

    def __init__(self, dets):
        self.dets = dets
        self.resets = 0

    def track(self, frame):
        return list(self.dets)

    def reset(self):
        self.resets += 1


class _RecordingPose:
    """Stands in for YoloPoseEstimator; records how it was called."""

    def __init__(self, keypoints=()):
        self.keypoints = list(keypoints)
        self.full_frame_runs = 0
        self.estimate_kwargs = []

    def predict_full_frame(self, frame):
        self.full_frame_runs += 1
        return None

    def estimate(self, frame, bboxes, **kwargs):
        self.estimate_kwargs.append(kwargs)
        return [list(self.keypoints) for _ in bboxes]

    def reset(self):
        pass


def _detector(dets, pose=None, **kwargs) -> PlayerDetector:
    return PlayerDetector(tracker=_FixedTracker(dets), pose=pose, **kwargs)


class TestFootPosition:
    def test_bottom_centre_of_the_box(self):
        assert foot_position(BoundingBox(x1=100, y1=50, x2=200, y2=300)) == (150.0, 300.0)
        assert foot_position(BoundingBox(x1=0, y1=0, x2=80, y2=200)) == (40.0, 200.0)


class TestCourtFilter:
    """With an identity homography, pixel coordinates are court metres."""

    H = np.eye(3)

    def setup_method(self):
        self.detector = _detector([])

    def test_inside_the_court(self):
        assert self.detector._pixel_to_court(BoundingBox(x1=4.0, y1=0.0, x2=6.0, y2=10.0), self.H) == (5.0, 10.0)

    def test_far_outside_is_dropped(self):
        assert self.detector._pixel_to_court(BoundingBox(x1=50.0, y1=0.0, x2=60.0, y2=100.0), self.H) is None

    def test_within_the_margin_is_kept_and_clamped_onto_the_court(self):
        # feet at (5.0, 20.5): half a metre behind the back wall
        assert self.detector._pixel_to_court(BoundingBox(x1=4.0, y1=0.0, x2=6.0, y2=20.5), self.H) == (5.0, 20.0)

    def test_people_outside_the_court_never_become_players(self):
        dets = [_det(5.0, 0.9), _det(40.0, 0.95)]
        out = _detector(dets).detect_and_track(FRAME, 0, 0.0, self.H)
        assert [pf.confidence for pf in out] == [0.9]

    def test_without_a_calibration_everyone_is_kept_without_a_position(self):
        out = _detector([_det(5.0, 0.9), _det(40.0, 0.95)]).detect_and_track(FRAME, 0, 0.0, None)
        assert len(out) == 2 and all(pf.position is None for pf in out)


class TestPlayerCap:
    DETS = [_det(float(i), 0.9 - 0.1 * i) for i in range(6)]

    def test_default_keeps_the_four_most_confident(self):
        out = _detector(self.DETS).detect_and_track(FRAME, 0, 0.0, np.eye(3))
        assert [round(pf.confidence, 1) for pf in out] == [0.9, 0.8, 0.7, 0.6]

    def test_higher_cap_keeps_the_extra_candidates(self):
        out = _detector(self.DETS, max_players=8).detect_and_track(FRAME, 0, 0.0, np.eye(3))
        assert len(out) == 6

    def test_cap_does_not_leak_between_detectors(self):
        _detector(self.DETS, max_players=8)
        assert len(_detector(self.DETS).detect_and_track(FRAME, 0, 0.0, np.eye(3))) == 4


class TestPlayerIds:
    def test_tracked_and_untracked_ids(self):
        out = _detector([_det(1.0, 0.9, track_id=5), _det(3.0, 0.8)]).detect_and_track(FRAME, 7, 280.0, np.eye(3))
        assert [pf.player_id for pf in out] == ["P_005", "P_det_001"]
        assert all(pf.frame_id == 7 and pf.timestamp_ms == 280.0 and pf.team_id is None for pf in out)

    def test_reset_reaches_the_tracker(self):
        d = _detector([])
        d.reset()
        assert d.tracker.resets == 1


class TestPose:
    DETS = [_det(1.0, 0.9, track_id=1)]

    def test_default_runs_the_full_frame_pass_once(self):
        pose = _RecordingPose()
        _detector(self.DETS, pose, pose_closeups_only=False).detect_and_track(FRAME, 0, 0.0, np.eye(3))
        # either ahead of time on the worker thread, or inside estimate()
        ran_ahead = pose.full_frame_runs == 1 and "full_frame" in pose.estimate_kwargs[0]
        left_to_estimate = pose.full_frame_runs == 0 and pose.estimate_kwargs == [{}]
        assert ran_ahead or left_to_estimate

    def test_closeups_only_skips_the_full_frame_pass(self):
        pose = _RecordingPose()
        _detector(self.DETS, pose, pose_closeups_only=True).detect_and_track(FRAME, 0, 0.0, np.eye(3))
        assert pose.full_frame_runs == 0
        assert pose.estimate_kwargs == [{"full_frame": None}]

    def test_keypoints_of_a_tracked_player_are_smoothed_over_time(self):
        pose = _RecordingPose([PoseKeypoint(name="nose", x=100.0, y=50.0, confidence=1.0)])
        d = _detector(self.DETS, pose, pose_closeups_only=True)
        first = d.detect_and_track(FRAME, 0, 0.0, np.eye(3))[0].keypoints[0]
        pose.keypoints = [PoseKeypoint(name="nose", x=200.0, y=50.0, confidence=1.0)]
        second = d.detect_and_track(FRAME, 1, 40.0, np.eye(3))[0].keypoints[0]
        assert first.x == 100.0
        assert second.x == 100.0 + PlayerDetector.POSE_SMOOTHING * 100.0  # part of the way to the new place


class TestPoseMatching:
    def test_iou_matrix(self):
        boxes = [BoundingBox(x1=0, y1=0, x2=10, y2=10), BoundingBox(x1=100, y1=100, x2=110, y2=110)]
        dets = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [200, 200, 210, 210]], dtype=np.float32)
        iou = iou_matrix(boxes, dets)
        assert iou.shape == (2, 3)
        np.testing.assert_allclose(iou[0], [1.0, 1 / 3, 0.0])
        np.testing.assert_allclose(iou[1], 0.0)
        assert iou_matrix([], dets).shape == (0, 3)

    def test_each_pose_goes_to_the_box_it_overlaps(self):
        est = YoloPoseEstimator(model_path="unused.pt", crop_fallback=False)
        est._model = object()  # never called: the full-frame result is handed in
        boxes = [
            BoundingBox(x1=0, y1=0, x2=10, y2=20),
            BoundingBox(x1=50, y1=0, x2=60, y2=20),
            BoundingBox(x1=200, y1=0, x2=210, y2=20),
        ]
        det_xyxy = np.array([[51, 0, 61, 20], [0, 1, 10, 20]], dtype=np.float32)  # in the other order
        xy = np.zeros((2, 17, 2), np.float32)
        xy[0, :, 0], xy[1, :, 0] = 55.0, 5.0
        conf = np.full((2, 17), 0.9, np.float32)
        conf[1, 0] = 0.0  # a keypoint that was not found is left out
        out = est.estimate(FRAME, boxes, full_frame=(det_xyxy, xy, conf))
        assert [kp.x for kp in out[0]] == [5.0] * 16 and out[0][0].name == COCO_KEYPOINT_NAMES[1]
        assert [kp.x for kp in out[1]] == [55.0] * 17
        assert out[2] == []  # nobody near the third box
