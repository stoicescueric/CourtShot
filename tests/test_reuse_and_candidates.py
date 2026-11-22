"""
Tests for the guards around reused results:
  - a tracking cache / 3D fit is only reused for the inputs it was made from
  - caches made by earlier versions still load
  - the static-ball filter counts occupancy near in time, not over the video
"""

from __future__ import annotations

import enum
import json
import os
import pickle
import sys
import types

import pytest

from courtshot.analysis import ball_events as bt
from courtshot.analysis import match
from courtshot.pipeline import CourtShot
from courtshot.schemas.tracking import BallFrame, BoundingBox, CourtCalibration
from courtshot.tracking import cache as tracking_cache
from courtshot.tracking.pipeline import TrackingResult


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
    def _courtshot(self, tmp_path, **kwargs) -> CourtShot:
        video = tmp_path / "clip.mp4"
        if not video.exists():
            video.write_bytes(b"not really a video")
        model = tmp_path / "ball.pt"
        if not model.exists():
            model.write_bytes(b"weights")
        kwargs.setdefault("calibration", _calibration())
        kwargs.setdefault("ball_model_path", model)
        return CourtShot(video, **kwargs)

    def _save_cache(self, p: CourtShot) -> None:
        tracking_cache.save_tracking(TrackingResult(), p.cache_path, p.tracking_key())

    def test_same_inputs_reuse_the_cache(self, tmp_path):
        self._save_cache(self._courtshot(tmp_path))
        again = self._courtshot(tmp_path)
        assert isinstance(again.track(), TrackingResult)
        assert again.tracking_from_cache

    def test_key_changes_with_each_input(self, tmp_path):
        base = self._courtshot(tmp_path).tracking_key()
        other_model = tmp_path / "other_ball.pt"
        other_model.write_bytes(b"weights")
        assert self._courtshot(tmp_path, calibration=_calibration(2.0)).tracking_key() != base
        assert self._courtshot(tmp_path, ball_model_path=other_model).tracking_key() != base
        assert self._courtshot(tmp_path, enable_pose=False).tracking_key() != base
        assert self._courtshot(tmp_path, max_players=4).tracking_key() != base

    def test_tracking_made_at_another_ball_stride_is_not_reused(self, tmp_path):
        p = self._courtshot(tmp_path)
        self._save_cache(p)
        other = self._courtshot(tmp_path, ball_stride=p.ball_stride + 1)
        reason = tracking_cache.stale_reason(p.cache_path, other.tracking_key())
        assert reason is not None and "ball_stride" in reason

    def test_cache_from_before_tracking_was_made_faster_is_reused(self, tmp_path):
        """Its key has no player_fps and no ball_stride; it is tracking of
        this video all the same."""
        p = self._courtshot(tmp_path)
        self._save_cache(p)
        key_file = tracking_cache.key_path_for(p.cache_path)
        old = {k: v for k, v in json.loads(key_file.read_text()).items() if k not in ("player_fps", "ball_stride")}
        key_file.write_text(json.dumps(old))
        assert tracking_cache.stale_reason(p.cache_path, p.tracking_key()) is None
        assert self._courtshot(tmp_path).track() is not None and p.cache_path.exists()

    def test_retrained_model_under_the_same_name_changes_the_key(self, tmp_path):
        before = self._courtshot(tmp_path).tracking_key()
        (tmp_path / "ball.pt").write_bytes(b"other weights")
        assert self._courtshot(tmp_path).tracking_key() != before

    def test_stale_reason_names_what_changed(self, tmp_path):
        p = self._courtshot(tmp_path)
        self._save_cache(p)
        other = self._courtshot(tmp_path, calibration=_calibration(2.0))
        reason = tracking_cache.stale_reason(p.cache_path, other.tracking_key())
        assert reason is not None and "calibration_sha1" in reason
        assert tracking_cache.stale_reason(p.cache_path, p.tracking_key()) is None

    def test_stale_cache_is_set_aside_not_deleted(self, tmp_path):
        p = self._courtshot(tmp_path)
        self._save_cache(p)
        other = self._courtshot(tmp_path, calibration=_calibration(2.0))
        tracking_cache.set_aside_if_stale(other.cache_path, other.tracking_key())
        assert not p.cache_path.exists()
        assert (tmp_path / "clip_tracking_cache.old.pkl").exists()

    def test_cache_without_a_key_is_still_used(self, tmp_path):
        p = self._courtshot(tmp_path)
        with open(p.cache_path, "wb") as f:
            pickle.dump(TrackingResult(), f)
        assert isinstance(p.track(), TrackingResult)
        assert p.tracking_from_cache

    def test_missing_calibration_or_ball_model_is_reported(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"not really a video")
        with pytest.raises(ValueError, match="calibrat"):
            CourtShot(video, ball_model_path=video)  # no clip_calibration.json next to it
        with pytest.raises(FileNotFoundError, match="Ball model"):
            CourtShot(video, calibration=_calibration(), ball_model_path=tmp_path / "nope.pt")


class TestOlderCaches:
    def test_cache_made_when_the_package_was_called_padex_still_loads(self, tmp_path):
        frame = BallFrame(frame_id=3, timestamp_ms=120.0, bbox=BoundingBox(x1=1, y1=2, x2=9, y2=10), confidence=0.7)
        data = pickle.dumps(TrackingResult(ball_frames=[frame], calibration=_calibration()), protocol=0)
        assert b"courtshot." in data
        path = tmp_path / "old_tracking_cache.pkl"
        path.write_bytes(data.replace(b"courtshot.", b"padex."))  # protocol 0 names classes as plain text
        with pytest.raises(ModuleNotFoundError):
            pickle.loads(path.read_bytes())
        loaded = tracking_cache.load_tracking(path)
        assert isinstance(loaded, TrackingResult) and loaded.ball_frames == [frame]
        assert loaded.calibration == _calibration()

    def test_key_naming_the_tracker_config_by_its_old_name_still_matches(self, tmp_path):
        cache = tmp_path / "clip_tracking_cache.pkl"
        cache.write_bytes(b"tracking")
        tracking_cache.key_path_for(cache).write_text(json.dumps({"tracker": "padex_bytetrack.yaml"}))
        assert tracking_cache.stale_reason(cache, {"tracker": "courtshot_bytetrack.yaml"}) is None
        assert tracking_cache.stale_reason(cache, {"tracker": "courtshot_botsort.yaml"}) is not None

    def test_cache_with_the_ball_fields_that_are_gone_still_loads(self, tmp_path, monkeypatch):
        """Tracking caches made before BallFrame lost its floor position and
        visibility state hold instances of two classes that no longer exist."""
        import courtshot.schemas.tracking as schema

        class BallVisibility(enum.StrEnum):
            VISIBLE = "visible"

        class Position3D(schema.BaseModel):
            x: float
            y: float
            z: float = 0.0

        class OldBallFrame(schema.BallFrame):
            position: Position3D | None = None
            visibility: BallVisibility = BallVisibility.VISIBLE

        for cls in (BallVisibility, Position3D):
            cls.__module__, cls.__qualname__ = "courtshot.schemas.tracking", cls.__name__
            monkeypatch.setattr(schema, cls.__name__, cls, raising=False)
        OldBallFrame.__module__, OldBallFrame.__qualname__ = "courtshot.schemas.tracking", "BallFrame"
        new_ball_frame = schema.BallFrame
        monkeypatch.setattr(schema, "BallFrame", OldBallFrame)

        old = TrackingResult(
            ball_frames=[
                OldBallFrame(
                    frame_id=3,
                    timestamp_ms=120.0,
                    bbox=BoundingBox(x1=1, y1=2, x2=9, y2=10),
                    confidence=0.7,
                    position=Position3D(x=5.0, y=10.0),
                    candidates=[(5.0, 6.0, 0.7)],
                )
            ]
        )
        path = tmp_path / "old_tracking_cache.pkl"
        with open(path, "wb") as f:
            pickle.dump(old, f)

        # back to today's schema: the two classes are gone
        monkeypatch.setattr(schema, "BallFrame", new_ball_frame)
        monkeypatch.delattr(schema, "BallVisibility")
        monkeypatch.delattr(schema, "Position3D")
        with pytest.raises(AttributeError):
            pickle.loads(path.read_bytes())

        bf = tracking_cache.load_tracking(path).ball_frames[0]
        assert isinstance(bf, BallFrame)
        assert (bf.frame_id, bf.confidence, bf.bbox.x2, bf.candidates) == (3, 0.7, 9.0, [(5.0, 6.0, 0.7)])


# ---------------------------------------------------------------------------
# 3D fit cache
# ---------------------------------------------------------------------------


class TestBall3dCacheKey:
    def _files(self, tmp_path):
        tracking = tmp_path / "clip_tracking_cache.pkl"
        tracking.write_bytes(b"tracking v1")
        fit = match.fit_path(tracking)
        fit.write_bytes(b"fit")
        return tracking, fit

    def test_fit_sits_next_to_the_tracking_cache(self, tmp_path):
        tracking, fit = self._files(tmp_path)
        assert fit == tmp_path / "clip_tracking_cache_ball3d.pkl"

    def test_fit_is_reused_for_the_tracking_it_was_made_from(self, tmp_path):
        tracking, fit = self._files(tmp_path)
        key = match.fit_key(tracking, 25.0)
        match.key_path(fit).write_text(json.dumps(key))
        assert match.stale_reason(fit, tracking, match.fit_key(tracking, 25.0)) is None

    def test_fit_is_not_reused_after_retracking(self, tmp_path):
        tracking, fit = self._files(tmp_path)
        match.key_path(fit).write_text(json.dumps(match.fit_key(tracking, 25.0)))
        tracking.write_bytes(b"tracking v2")
        reason = match.stale_reason(fit, tracking, match.fit_key(tracking, 25.0))
        assert reason is not None and "tracking_sha1" in reason

    def test_old_fit_without_key_goes_by_file_age(self, tmp_path):
        tracking, fit = self._files(tmp_path)
        key = match.fit_key(tracking, 25.0)
        os.utime(fit, (1_000_000, 1_000_000))
        os.utime(tracking, (2_000_000, 2_000_000))
        assert match.stale_reason(fit, tracking, key) is not None
        os.utime(fit, (3_000_000, 3_000_000))
        assert match.stale_reason(fit, tracking, key) is None

    def test_missing_fit(self, tmp_path):
        tracking, fit = self._files(tmp_path)
        fit.unlink()
        assert match.stale_reason(fit, tracking, {}) == "missing"

    def test_fit_saved_by_an_earlier_version_is_fitted_again(self, tmp_path, monkeypatch):
        """A fit pickled when the 3D code was a top-level script names a
        module that is gone; it must be redone, not crash the run."""
        tracking, fit = self._files(tmp_path)
        gone = types.ModuleType("ball3d_of_an_earlier_version")

        class Ball3DResult:
            flights = ()

        Ball3DResult.__module__, Ball3DResult.__qualname__ = gone.__name__, "Ball3DResult"
        gone.Ball3DResult = Ball3DResult
        monkeypatch.setitem(sys.modules, gone.__name__, gone)
        with open(fit, "wb") as f:
            pickle.dump(Ball3DResult(), f)
        monkeypatch.delitem(sys.modules, gone.__name__)

        key = match.fit_key(tracking, 25.0)
        key["player_ids"] = repr(match.StableIdConfig()) + " swaps []"
        match.key_path(fit).write_text(json.dumps(key))
        refits = []
        monkeypatch.setattr(
            match.ball_flight,
            "run_on_tracking",
            lambda stable, ball, fps: refits.append(1) or types.SimpleNamespace(flights=[]),
        )
        result = match.ball3d_for(tracking, stable=None, ball=None, fps=25.0, id_report={})
        assert refits == [1] and result.flights == []


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
