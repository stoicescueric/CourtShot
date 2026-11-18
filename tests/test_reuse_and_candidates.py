"""
Tests for the guards around reused results and player candidates:
  - a tracking cache / 3D fit is only reused for the inputs it was made from
  - the static-ball filter counts occupancy near in time, not over the video
  - the player cap is configurable; pose can come from close-ups only
"""

from __future__ import annotations

import json
import os
import pickle

import ball_track as bt
import export_scene3d as scene
import numpy as np

from padex.pipeline import Padex
from padex.schemas.tracking import BoundingBox, CourtCalibration
from padex.tracking.pipeline import TrackingResult
from padex.tracking.player import PlayerDetector, RawDetection


def _calibration(scale: float = 1.0) -> CourtCalibration:
    return CourtCalibration(
        frame_width=1280,
        frame_height=720,
        homography_matrix=[[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
    )


# ---------------------------------------------------------------------------
# tracking cache
# ---------------------------------------------------------------------------


class TestTrackingCacheKey:
    def _padex(self, tmp_path, **kwargs) -> Padex:
        video = tmp_path / "clip.mp4"
        if not video.exists():
            video.write_bytes(b"not really a video")
        kwargs.setdefault("calibration", _calibration())
        return Padex(video, **kwargs)

    def _save_cache(self, p: Padex, tmp_path) -> None:
        cache = tmp_path / "clip_tracking_cache.pkl"
        with open(cache, "wb") as f:
            pickle.dump(TrackingResult(), f)
        cache.with_suffix(".key.json").write_text(json.dumps(p._tracking_key()))

    def test_same_inputs_reuse_the_cache(self, tmp_path):
        p = self._padex(tmp_path)
        self._save_cache(p, tmp_path)
        again = self._padex(tmp_path)
        assert isinstance(again._run_tracking(), TrackingResult)
        assert again.tracking_from_cache

    def test_key_changes_with_each_input(self, tmp_path):
        base = self._padex(tmp_path)._tracking_key()
        model = tmp_path / "ball.pt"
        model.write_bytes(b"weights")
        assert (
            self._padex(tmp_path, calibration=_calibration(2.0))._tracking_key() != base
        )
        assert self._padex(tmp_path, ball_model_path=model)._tracking_key() != base
        assert self._padex(tmp_path, use_tracknet_v3=True)._tracking_key() != base
        assert self._padex(tmp_path, max_players=8)._tracking_key() != base

    def test_key_changes_with_pose_mode(self, tmp_path, monkeypatch):
        base = self._padex(tmp_path)._tracking_key()
        monkeypatch.setenv("PADEX_POSE_CLOSEUPS", "1")
        assert self._padex(tmp_path)._tracking_key() != base

    def test_retrained_model_under_the_same_name_changes_the_key(self, tmp_path):
        model = tmp_path / "ball.pt"
        model.write_bytes(b"weights")
        before = self._padex(tmp_path, ball_model_path=model)._tracking_key()
        model.write_bytes(b"other weights")
        assert self._padex(tmp_path, ball_model_path=model)._tracking_key() != before

    def test_stale_reason_names_what_changed(self, tmp_path):
        p = self._padex(tmp_path)
        self._save_cache(p, tmp_path)
        other = self._padex(tmp_path, calibration=_calibration(2.0))
        key_path = tmp_path / "clip_tracking_cache.key.json"
        reason = Padex._stale_reason(key_path, other._tracking_key())
        assert reason is not None and "calibration_sha1" in reason
        assert Padex._stale_reason(key_path, p._tracking_key()) is None

    def test_cache_without_a_key_is_still_used(self, tmp_path):
        p = self._padex(tmp_path)
        with open(tmp_path / "clip_tracking_cache.pkl", "wb") as f:
            pickle.dump(TrackingResult(), f)
        assert isinstance(p._run_tracking(), TrackingResult)
        assert p.tracking_from_cache


# ---------------------------------------------------------------------------
# 3D fit cache
# ---------------------------------------------------------------------------


class TestBall3dCacheKey:
    def _files(self, tmp_path):
        tracking = tmp_path / "clip_tracking_cache.pkl"
        tracking.write_bytes(b"tracking v1")
        fit = tmp_path / "clip_tracking_cache_ball3d.pkl"
        fit.write_bytes(b"fit")
        return tracking, fit

    def test_fit_is_reused_for_the_tracking_it_was_made_from(self, tmp_path):
        tracking, fit = self._files(tmp_path)
        key = scene.fit_key(tracking, 25.0)
        scene.key_path(fit).write_text(json.dumps(key))
        assert scene.stale_reason(fit, tracking, scene.fit_key(tracking, 25.0)) is None

    def test_fit_is_not_reused_after_retracking(self, tmp_path):
        tracking, fit = self._files(tmp_path)
        scene.key_path(fit).write_text(json.dumps(scene.fit_key(tracking, 25.0)))
        tracking.write_bytes(b"tracking v2")
        reason = scene.stale_reason(fit, tracking, scene.fit_key(tracking, 25.0))
        assert reason is not None and "tracking_sha1" in reason

    def test_old_fit_without_key_goes_by_file_age(self, tmp_path):
        tracking, fit = self._files(tmp_path)
        key = scene.fit_key(tracking, 25.0)
        os.utime(fit, (1_000_000, 1_000_000))
        os.utime(tracking, (2_000_000, 2_000_000))
        assert scene.stale_reason(fit, tracking, key) is not None
        os.utime(fit, (3_000_000, 3_000_000))
        assert scene.stale_reason(fit, tracking, key) is None

    def test_missing_fit(self, tmp_path):
        tracking, fit = self._files(tmp_path)
        fit.unlink()
        assert scene.stale_reason(fit, tracking, {}) == "missing"


# ---------------------------------------------------------------------------
# static-ball filter
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# player candidates and pose mode
# ---------------------------------------------------------------------------


def _det(x: float, conf: float) -> RawDetection:
    return RawDetection(
        bbox=BoundingBox(x1=x, y1=0.0, x2=x + 0.5, y2=5.0), confidence=conf
    )


class _FixedDetections:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, frame, frame_id, timestamp_ms):
        return list(self.dets)


class _NoTeams:
    def classify(self, detections, frame):
        return {}


class _RecordingPose:
    """Stands in for YoloPoseEstimationStrategy; records how it was called."""

    def __init__(self):
        self.full_frame_runs = 0
        self.estimate_kwargs = []

    def predict_full_frame(self, frame):
        self.full_frame_runs += 1
        return None

    def estimate(self, frame, bboxes, **kwargs):
        self.estimate_kwargs.append(kwargs)
        return [[] for _ in bboxes]


def _detector(dets, pose=None, max_players=None, **kwargs) -> PlayerDetector:
    strategy = _FixedDetections(dets)
    strategy.detect_with_tracking = strategy.detect
    return PlayerDetector(
        detection_strategy=strategy,
        team_classifier=_NoTeams(),
        pose_strategy=pose,
        max_players=max_players,
        **kwargs,
    )


class TestPlayerCap:
    FRAME = np.zeros((8, 8, 3), np.uint8)
    DETS = [_det(float(i), 0.9 - 0.1 * i) for i in range(6)]

    def test_default_keeps_the_four_most_confident(self):
        out = _detector(self.DETS).detect(self.FRAME, 0, 0.0, np.eye(3))
        assert [round(pf.confidence, 1) for pf in out] == [0.9, 0.8, 0.7, 0.6]

    def test_higher_cap_keeps_the_extra_candidates(self):
        out = _detector(self.DETS, max_players=8).detect(self.FRAME, 0, 0.0, np.eye(3))
        assert len(out) == 6

    def test_cap_does_not_leak_between_detectors(self):
        _detector(self.DETS, max_players=8)
        assert PlayerDetector.MAX_PLAYERS == 4
        assert len(_detector(self.DETS).detect(self.FRAME, 0, 0.0, np.eye(3))) == 4


class TestPoseCloseupsOnly:
    FRAME = np.zeros((8, 8, 3), np.uint8)
    DETS = [_det(1.0, 0.9)]

    def test_default_runs_the_full_frame_pass_once(self):
        pose = _RecordingPose()
        _detector(self.DETS, pose, pose_closeups_only=False).detect_and_track(
            self.FRAME, 0, 0.0, np.eye(3)
        )
        # either ahead of time on the worker thread, or inside estimate()
        ran_ahead = (
            pose.full_frame_runs == 1 and "full_frame" in pose.estimate_kwargs[0]
        )
        left_to_estimate = pose.full_frame_runs == 0 and pose.estimate_kwargs == [{}]
        assert ran_ahead or left_to_estimate

    def test_closeups_only_skips_the_full_frame_pass(self):
        pose = _RecordingPose()
        _detector(self.DETS, pose, pose_closeups_only=True).detect_and_track(
            self.FRAME, 0, 0.0, np.eye(3)
        )
        assert pose.full_frame_runs == 0
        assert pose.estimate_kwargs == [{"full_frame": None}]
