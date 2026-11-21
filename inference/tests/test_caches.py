"""
Tests for the caches in a job's work folder: reused only for the inputs (and
package version) they were made from, never trusted without their key, and
never left half-written.
"""

from __future__ import annotations

import json
import pickle
import types

import pytest

from courtshot.analysis import match
from courtshot.pipeline import CourtShot
from courtshot.schemas.tracking import CourtCalibration
from courtshot.tracking import cache as tracking_cache
from courtshot.tracking.pipeline import TrackingResult


def _calibration(scale: float = 1.0) -> CourtCalibration:
    return CourtCalibration(
        frame_width=1280,
        frame_height=720,
        homography_matrix=[[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
    )


class TestTrackingCache:
    def _courtshot(self, tmp_path, **kwargs) -> CourtShot:
        video = tmp_path / "clip.mp4"
        if not video.exists():
            video.write_bytes(b"not really a video")
        model = tmp_path / "ball.pt"
        if not model.exists():
            model.write_bytes(b"weights")
        kwargs.setdefault("calibration", _calibration())
        kwargs.setdefault("ball_model_path", model)
        kwargs.setdefault("device", "cpu")
        return CourtShot(video, work_dir=tmp_path / "work", **kwargs)

    def _save_cache(self, p: CourtShot) -> None:
        tracking_cache.save_tracking(TrackingResult(), p.cache_path, p.tracking_key())

    def test_cache_lives_in_the_work_folder(self, tmp_path):
        assert self._courtshot(tmp_path).cache_path == tmp_path / "work" / "clip_tracking_cache.pkl"

    def test_same_inputs_reuse_the_cache(self, tmp_path):
        self._save_cache(self._courtshot(tmp_path))
        progress = []
        again = self._courtshot(tmp_path, on_progress=lambda stage, done: progress.append((stage, done)))
        assert isinstance(again.track(), TrackingResult)  # no detector was built: it would fail on this "video"
        assert progress == [("tracking", 0.0), ("tracking", 1.0)]

    def test_key_changes_with_each_input(self, tmp_path):
        base = self._courtshot(tmp_path).tracking_key()
        other_model = tmp_path / "other_ball.pt"
        other_model.write_bytes(b"weights")
        assert self._courtshot(tmp_path, calibration=_calibration(2.0)).tracking_key() != base
        assert self._courtshot(tmp_path, ball_model_path=other_model).tracking_key() != base
        assert self._courtshot(tmp_path, enable_pose=False).tracking_key() != base
        assert self._courtshot(tmp_path, max_players=4).tracking_key() != base

    def test_key_changes_with_pose_mode(self, tmp_path, monkeypatch):
        base = self._courtshot(tmp_path).tracking_key()
        monkeypatch.setenv("COURTSHOT_POSE_CLOSEUPS", "1")
        assert self._courtshot(tmp_path).tracking_key() != base

    def test_key_changes_with_the_package_version(self, tmp_path, monkeypatch):
        base = self._courtshot(tmp_path).tracking_key()
        monkeypatch.setattr("courtshot.__version__", "9.9.9")
        changed = self._courtshot(tmp_path).tracking_key()
        assert changed != base and changed["package_version"] == "9.9.9"

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

    def test_stale_cache_is_discarded_with_its_key(self, tmp_path):
        p = self._courtshot(tmp_path)
        self._save_cache(p)
        other = self._courtshot(tmp_path, calibration=_calibration(2.0))
        tracking_cache.discard_if_stale(other.cache_path, other.tracking_key())
        assert list((tmp_path / "work").iterdir()) == []

    def test_cache_without_a_key_is_discarded(self, tmp_path):
        p = self._courtshot(tmp_path)
        p.cache_path.write_bytes(pickle.dumps(TrackingResult()))
        assert tracking_cache.stale_reason(p.cache_path, p.tracking_key()) == "its key file is missing"
        tracking_cache.discard_if_stale(p.cache_path, p.tracking_key())
        assert not p.cache_path.exists()

    def test_unreadable_key_is_not_trusted(self, tmp_path):
        p = self._courtshot(tmp_path)
        self._save_cache(p)
        tracking_cache.key_path_for(p.cache_path).write_text("{not json")
        assert tracking_cache.stale_reason(p.cache_path, p.tracking_key()) == "its key file is unreadable"

    def test_missing_ball_model_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Ball model"):
            self._courtshot(tmp_path, ball_model_path=tmp_path / "nope.pt")


class TestAtomicWrites:
    def test_a_failed_write_leaves_neither_file_nor_temporary(self, tmp_path):
        path = tmp_path / "x_tracking_cache.pkl"
        with pytest.raises(RuntimeError), tracking_cache.atomic_file(path) as f:
            f.write(b"half a pickle")
            raise RuntimeError("killed mid-write")
        assert list(tmp_path.iterdir()) == []

    def test_an_earlier_file_survives_a_failed_rewrite(self, tmp_path):
        path = tmp_path / "x_tracking_cache.pkl"
        tracking_cache.save_tracking(TrackingResult(), path, {"v": 1})
        before = path.read_bytes()
        with pytest.raises(RuntimeError), tracking_cache.atomic_file(path) as f:
            f.write(b"garbage")
            raise RuntimeError("killed mid-write")
        assert path.read_bytes() == before

    def test_pickle_and_key_are_both_written(self, tmp_path):
        path = tmp_path / "x_tracking_cache.pkl"
        tracking_cache.save_tracking(TrackingResult(), path, {"v": 1})
        assert isinstance(tracking_cache.load_tracking(path), TrackingResult)
        assert json.loads(tracking_cache.key_path_for(path).read_text()) == {"v": 1}
        assert sorted(p.name for p in tmp_path.iterdir()) == ["x_tracking_cache.key.json", "x_tracking_cache.pkl"]


class TestBall3dCache:
    def _files(self, tmp_path):
        tracking = tmp_path / "clip_tracking_cache.pkl"
        tracking.write_bytes(b"tracking v1")
        return tracking, match.fit_path(tracking)

    def _key(self, tracking):
        key = match.fit_key(tracking, 25.0)
        key["player_ids"] = repr(match.StableIdConfig()) + " swaps []"
        return key

    def _fake_fit(self, monkeypatch, calls: list):
        def run_on_tracking(stable, ball, fps, workers=None, on_progress=None):
            calls.append(workers)
            return types.SimpleNamespace(flights=[])

        monkeypatch.setattr(match.ball_flight, "run_on_tracking", run_on_tracking)

    def test_fit_is_saved_with_its_key_and_reused(self, tmp_path, monkeypatch):
        tracking, fit = self._files(tmp_path)
        calls: list = []
        self._fake_fit(monkeypatch, calls)
        match.ball3d_for(tracking, stable=None, ball=None, fps=25.0, id_report={}, workers=2)
        assert calls == [2] and fit.exists() and match.key_path(fit).exists()
        match.ball3d_for(tracking, stable=None, ball=None, fps=25.0, id_report={})
        assert calls == [2]  # loaded, not fitted again

    def test_fit_without_its_key_is_fitted_again(self, tmp_path, monkeypatch):
        tracking, fit = self._files(tmp_path)
        fit.write_bytes(pickle.dumps(types.SimpleNamespace(flights=["stale"])))
        assert match.stale_reason(fit, self._key(tracking)) == "its key file is missing"
        calls: list = []
        self._fake_fit(monkeypatch, calls)
        result = match.ball3d_for(tracking, stable=None, ball=None, fps=25.0, id_report={})
        assert calls == [None] and result.flights == []

    def test_fit_is_not_reused_after_retracking(self, tmp_path):
        tracking, fit = self._files(tmp_path)
        tracking_cache.save_pickle(types.SimpleNamespace(flights=[]), fit, self._key(tracking))
        assert match.stale_reason(fit, self._key(tracking)) is None
        tracking.write_bytes(b"tracking v2")
        reason = match.stale_reason(fit, self._key(tracking))
        assert reason is not None and "tracking_sha1" in reason

    def test_missing_fit(self, tmp_path):
        tracking, fit = self._files(tmp_path)
        assert match.stale_reason(fit, {}) == "missing"
