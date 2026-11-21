from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"

# The tests import the stub and the fakes; the job subprocess imports courtshot and the fakes.
sys.path.insert(0, str(HERE))
os.environ["PYTHONPATH"] = os.pathsep.join([str(SRC), str(HERE)])

from courtshot.worker.client import ServerClient  # noqa: E402
from courtshot.worker.config import Settings, find_tool  # noqa: E402
from courtshot.worker.runner import Runner  # noqa: E402
from stub_server import StubServer  # noqa: E402

FFMPEG, FFPROBE = find_tool("ffmpeg"), find_tool("ffprobe")
LIMITS = {"min_video_seconds": 1, "max_video_seconds": 60, "calibration_max_error_m": 0.25}
CALIBRATION = {"points": {"bottom_left": [10, 110]}, "frame_width": 160, "frame_height": 120}


def make_video(path: Path, *, seconds: float = 2, size: str = "160x120", rate: int = 25,
               filters: str | None = None, extra: tuple[str, ...] = ()) -> Path:
    command = [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}:duration={seconds}"]
    command += ["-vf", filters] if filters else []
    command += [*extra, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True)
    return path


def remux(source: Path, dest: Path, *before_input: str, after_input: tuple[str, ...] = ()) -> Path:
    command = [FFMPEG, "-v", "error", "-y", *before_input, "-i", str(source), "-c", "copy", *after_input, str(dest)]
    subprocess.run(command, check=True)
    return dest


@pytest.fixture(scope="session")
def video_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    # Noise makes it a few hundred kilobytes: several reads, so a broken download has something to resume.
    return make_video(tmp_path_factory.mktemp("video") / "clip.mp4", filters="noise=alls=60:allf=t").read_bytes()


@pytest.fixture
def stub() -> Iterator[StubServer]:
    with StubServer() as server:
        yield server


@pytest.fixture
def make_runner(stub: StubServer, tmp_path: Path) -> Callable[..., Runner]:
    def make(runner: str = "fakes:run_ok", **overrides: Any) -> Runner:
        settings = Settings(server_url=stub.url, token=stub.token, work_root=tmp_path / "work", ffmpeg=FFMPEG,
                            ffprobe=FFPROBE, runner=runner, poll_seconds=0.05, **overrides)
        client = ServerClient(stub.url, stub.token, timeout=(2, 5), tries=2, backoff=0.01)
        return Runner(settings, client, kill_grace=1.0, retry_pause=0.05, max_retry_pause=0.2)

    return make
