"""Progress weights, settings, the client's refusals and the job module's import discipline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import CALIBRATION, LIMITS
from stub_server import StubServer

from courtshot.worker.client import JobGone, JobRef, ServerClient
from courtshot.worker.config import add_arguments, parse_weights, settings_from
from courtshot.worker.progress import DEFAULT_WEIGHTS, overall


def test_overall_progress_is_a_weighted_sum() -> None:
    assert overall(DEFAULT_WEIGHTS, "download", 0) == 0
    assert overall(DEFAULT_WEIGHTS, "tracking", 0.5) == pytest.approx((5 + 3 + 22.5) / 100)
    assert overall(DEFAULT_WEIGHTS, "upload", 1) == 1
    assert overall(DEFAULT_WEIGHTS, "render", 7) == overall(DEFAULT_WEIGHTS, "upload", 0)  # clamped


def test_stage_weights_setting() -> None:
    weights = parse_weights("tracking=50, ball3d=15")
    assert weights == {**DEFAULT_WEIGHTS, "tracking": 50, "ball3d": 15}
    assert parse_weights("") == DEFAULT_WEIGHTS
    with pytest.raises(ValueError):
        parse_weights("painting=3")


def test_relative_paths_are_made_absolute() -> None:
    # The job runs in its match folder: a relative path would point elsewhere there.
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = ["--work-root", "work", "--ball-model", "m.pt", "--ffmpeg", "bin/ffmpeg", "--ffprobe", "ffprobe"]
    settings = settings_from(parser.parse_args(args))
    assert settings.work_root == Path.cwd() / "work"
    assert settings.ball_model == Path.cwd() / "m.pt"
    assert settings.ffmpeg == str(Path.cwd() / "bin" / "ffmpeg")
    assert settings.ffprobe == "ffprobe"  # a bare name is looked up on PATH


def test_client_retries_5xx_and_raises_job_gone(stub: StubServer, video_bytes: bytes) -> None:
    job = stub.add_job(video_bytes, CALIBRATION, LIMITS, attempt=2)
    client = ServerClient(stub.url, stub.token, tries=3, backoff=0.01)
    assert client.claim({}) is not None
    assert client.claim({}) is None  # 204: nothing to do

    stub.inject("heartbeat", "500", times=2)
    client.heartbeat(JobRef(job.id, 2), "download", 0.1)
    assert job.heartbeats == [("download", 0.1)]

    with pytest.raises(JobGone) as stale:
        client.heartbeat(JobRef(job.id, 1), "download", 0.2)  # an attempt the server gave away
    assert stale.value.status == 409
    with pytest.raises(JobGone) as unknown:
        client.complete(JobRef(99, 1))
    assert unknown.value.status == 404


def test_job_module_imports_only_the_standard_library() -> None:
    # Every process of the 3D-fit pool imports it again under spawn.
    code = ("import sys, courtshot.worker.job_main; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('courtshot', 'requests', 'numpy', 'torch')))")
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert loaded.strip() == "['courtshot', 'courtshot.worker', 'courtshot.worker.job_main']"
