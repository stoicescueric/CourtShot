"""This Mac: the checks the worker must pass, the setup downloads, and what it tells the server."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import courtshot
from courtshot.worker.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BALL_MODEL = REPO_ROOT / "checkpoints" / "padel_tracknetv3.pt"
YOLO_WEIGHTS = ("yolo26m.pt", "yolo26m-pose.pt")
SIGLIP_MODEL = "google/siglip-base-patch16-224"
# Run in a child process so the long-running worker never holds PyTorch in memory.
TORCH_PROBE = "import torch; print(torch.__version__, torch.backends.mps.is_available())"


class Check(NamedTuple):
    name: str
    ok: bool
    detail: str


def run_checks(settings: Settings) -> list[Check]:
    return [_ffmpeg(settings.ffmpeg), _ffprobe(settings.ffprobe), _torch(settings.device), _ball_model(settings)]


def download_models() -> list[Check]:
    """Fetch the YOLO weights and SigLIP now, so the first job does not depend on the network."""
    from courtshot.weights import get_weight_path

    def siglip() -> str:
        from transformers import SiglipVisionModel

        SiglipVisionModel.from_pretrained(SIGLIP_MODEL)
        return "in the Hugging Face cache"

    actions: list[tuple[str, Callable[[], Any]]] = [(name, partial(get_weight_path, name)) for name in YOLO_WEIGHTS]
    actions.append((SIGLIP_MODEL, siglip))
    return [_download(name, action) for name, action in actions]


def pipeline_version() -> str:
    """The package version plus the repository's short commit, e.g. 0.2.0+1a2b3c4."""
    commit = _output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"]) or "unknown"
    return f"{courtshot.__version__}+{commit}"


def worker_info(settings: Settings, version: str) -> dict[str, Any]:
    memory = _output(["sysctl", "-n", "hw.memsize"])
    return {
        "hostname": socket.gethostname(),
        "chip": _output(["sysctl", "-n", "machdep.cpu.brand_string"]) or "unknown",
        "memory_gb": round(int(memory) / 2**30, 1) if memory.isdigit() else None,
        "device": settings.device or "mps",
        "pipeline_version": version,
    }


def _ffmpeg(path: str) -> Check:
    version = _output([path, "-hide_banner", "-version"])
    if not version:
        return Check("ffmpeg", False, f"not found at {path} (set COURTSHOT_FFMPEG)")
    if "libx264" not in _output([path, "-hide_banner", "-encoders"]):
        return Check("ffmpeg", False, f"{path} has no libx264 encoder")
    return Check("ffmpeg", True, f"{path}: {version.splitlines()[0]}")


def _ffprobe(path: str) -> Check:
    version = _output([path, "-hide_banner", "-version"])
    if not version:
        return Check("ffprobe", False, f"not found at {path} (set COURTSHOT_FFPROBE)")
    return Check("ffprobe", True, f"{path}: {version.splitlines()[0]}")


def _torch(device: str | None) -> Check:
    found = (_output([sys.executable, "-c", TORCH_PROBE], timeout=300).splitlines() or [""])[-1].split()
    if len(found) != 2:
        return Check("torch", False, "PyTorch cannot be imported (run uv sync in inference/)")
    version, mps = found[0], found[1] == "True"
    if mps:
        return Check("torch", True, f"PyTorch {version}, MPS available")
    if device == "cpu":
        return Check("torch", True, f"PyTorch {version}, no MPS; running on the CPU as configured")
    return Check("torch", False, f"PyTorch {version} sees no MPS (set COURTSHOT_DEVICE=cpu to run on the CPU)")


def _ball_model(settings: Settings) -> Check:
    path = settings.ball_model or DEFAULT_BALL_MODEL
    if path.is_file():
        return Check("ball model", True, str(path))
    return Check("ball model", False, f"missing: {path} (set COURTSHOT_BALL_MODEL)")


def _download(name: str, action: Callable[[], Any], tries: int = 3) -> Check:
    error = ""
    for attempt in range(tries):
        if attempt:
            time.sleep(5 * attempt)
        try:
            return Check(name, True, str(action()))
        except Exception as exc:  # network errors come in many types; report the last one
            error = f"{type(exc).__name__}: {exc}"
    return Check(name, False, error)


def _output(command: list[str], timeout: float = 30) -> str:
    """The command's standard output, or "" when it cannot run or fails."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""
