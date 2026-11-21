"""
Tests for courtshot.job.run_job, without the detectors: the tracking cache of
a synthetic match is put in the work folder first, so the run starts from it
(as a retry would) and the appearance model is switched off.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from courtshot.errors import JobFailure
from courtshot.io.video import probe_video
from courtshot.job import STAGES, run_job
from courtshot.pipeline import DEFAULT_MAX_PLAYERS
from courtshot.tracking import cache as tracking_cache
from tests import synthetic


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return synthetic.write_video(tmp_path_factory.mktemp("video") / "clip.mp4")


@pytest.fixture
def no_appearance(monkeypatch):
    monkeypatch.setattr("courtshot.analysis.match.cache_appearance", lambda *a, **k: None)


def _prepare(tmp_path, video, tracking=None):
    """A work folder holding this tracking (default: the synthetic match) as a valid cache."""
    model = tmp_path / "ball.pt"
    model.write_bytes(b"weights")
    work = tmp_path / "work"
    key = tracking_cache.tracking_key(video, synthetic.calibration(), model, True, DEFAULT_MAX_PLAYERS)
    cache = tracking_cache.cache_path_for(video, work)
    tracking_cache.save_tracking(tracking or synthetic.raw_tracking(), cache, key)
    return work, model


def _run(tmp_path, video, out_name="out", calibration=None, tracking=None, progress=None, **tools):
    work, model = _prepare(tmp_path, video, tracking)
    return run_job(
        video,
        calibration or synthetic.job_calibration(),
        work,
        tmp_path / out_name,
        progress or (lambda stage, done: None),
        pipeline_version="0.2.0+test",
        device="cpu",
        fit_workers=1,
        ball_model=model,
        **tools,
    )


class TestRun:
    def test_files_and_progress(self, tmp_path, video, no_appearance):
        calls: list[tuple[str, float]] = []
        names = _run(tmp_path, video, progress=lambda stage, done: calls.append((stage, done)))
        assert sorted(names) == ["plain.mp4", "positions.csv", "report.json", "summary.json", "tracked.mp4"]
        out = tmp_path / "out"
        assert sorted(p.name for p in out.iterdir()) == sorted(names)

        report = json.loads((out / "report.json").read_text())
        assert report["schema_version"] == 1 and report["pipeline_version"] == "0.2.0+test"
        assert report["match"] is None  # no ball in this match, so no scene.json either

        for name in ("tracked.mp4", "plain.mp4"):
            info = probe_video(out / name)
            assert (info["codec_name"], info["pix_fmt"]) == ("h264", "yuv420p")
            assert (info["width"], info["height"]) == (synthetic.W, synthetic.H)
            assert info["nb_read_packets"] == 250  # frame k of the video is frame k of the pipeline

        assert {stage for stage, _ in calls} <= set(STAGES)
        assert all(0.0 <= done <= 1.0 for _, done in calls)
        for stage in ("tracking", "appearance", "analysis", "render"):
            assert (stage, 1.0) in calls
        assert sum(stage == "render" for stage, _ in calls) > 100  # called often

    def test_the_given_ffmpeg_and_ffprobe_are_used(self, tmp_path, video, no_appearance, monkeypatch):
        """A LaunchAgent sees neither Homebrew's folder nor COURTSHOT_FFMPEG: the worker's paths reach the videos."""
        tools = {"ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")}
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.delenv("COURTSHOT_FFMPEG", raising=False)
        monkeypatch.delenv("COURTSHOT_FFPROBE", raising=False)
        assert shutil.which("ffmpeg") is None and shutil.which("ffprobe") is None
        names = _run(tmp_path, video, **tools)
        assert {"tracked.mp4", "plain.mp4"} <= set(names)

    def test_same_input_twice_gives_identical_json(self, tmp_path, video, no_appearance):
        _run(tmp_path, video, "first")
        _run(tmp_path, video, "second")
        for name in ("report.json", "summary.json"):
            assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()

    def test_files_of_an_earlier_attempt_are_removed(self, tmp_path, video, no_appearance):
        out = tmp_path / "out"
        out.mkdir()
        (out / "scene.json").write_text("{}")  # left over: this run has no 3D result
        (out / "tracked.mp4.tmp").write_bytes(b"half a video")  # left by a killed attempt
        names = _run(tmp_path, video)
        assert sorted(p.name for p in out.iterdir()) == sorted(names) and "scene.json" not in names


def _refused(match: str, **kwargs) -> None:
    with pytest.raises(JobFailure, match=match) as info:
        _run(**kwargs)
    assert info.value.retryable is False


class TestEarlyFailures:
    def test_invalid_calibration(self, tmp_path, video):
        five = dict(list(synthetic.points().items())[:5])
        bad = {"points": five, "frame_width": synthetic.W, "frame_height": synthetic.H}
        _refused("at least 6", tmp_path=tmp_path, video=video, calibration=bad)

    def test_frame_size_other_than_the_calibration(self, tmp_path, video):
        scale = 2
        cal = {
            "points": {n: [x * scale, y * scale] for n, (x, y) in synthetic.points().items()},
            "frame_width": synthetic.W * scale,
            "frame_height": synthetic.H * scale,
        }
        _refused("480x270.*960x540", tmp_path=tmp_path, video=video, calibration=cal)

    def test_four_players_never_together(self, tmp_path, video, no_appearance):
        three = synthetic.raw_tracking(players=3)
        _refused("Four players were never", tmp_path=tmp_path, video=video, tracking=three)
        assert not (tmp_path / "out" / "tracked.mp4").exists()  # stopped before the slow stages

    def test_players_on_court_too_briefly(self, tmp_path, video, no_appearance):
        short = synthetic.raw_tracking(n_frames=100)  # 4 s
        _refused(r"only 4\.0 s", tmp_path=tmp_path, video=video, tracking=short)


class TestImports:
    def test_job_and_3d_fit_modules_do_not_load_torch(self):
        """Every process of the 3D-fit pool imports these; loading PyTorch in
        each would exhaust a small Mac."""
        code = (
            "import sys, courtshot.job, courtshot.analysis.ball_flight, courtshot.pipeline;"
            "print(sorted(m for m in ('torch', 'ultralytics', 'transformers') if m in sys.modules))"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
        assert out.strip() == "[]"

    def test_mps_fallback_is_on_before_torch_loads(self):
        code = (
            "import os, sys, courtshot.tracking.device;"
            "assert 'torch' not in sys.modules;"
            "print(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'])"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTORCH_ENABLE_MPS_FALLBACK"}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
        assert out.stdout.strip() == "1"
