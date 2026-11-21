"""Worker settings: command-line flags, falling back to COURTSHOT_* environment variables."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from courtshot.worker.progress import DEFAULT_WEIGHTS, STAGES

DEFAULT_WORK_ROOT = Path.home() / "Library" / "Application Support" / "CourtShot" / "work"
DEFAULT_RUNNER = "courtshot.job:run_job"


def find_tool(name: str) -> str:
    """Absolute path of a command, falling back to Homebrew's folder (a LaunchAgent may not see it)."""
    return shutil.which(name) or f"/opt/homebrew/bin/{name}"


@dataclass(frozen=True)
class Settings:
    server_url: str = ""
    token: str = field(default="", repr=False)  # never in logs
    work_root: Path = DEFAULT_WORK_ROOT
    ffmpeg: str = field(default_factory=lambda: find_tool("ffmpeg"))
    ffprobe: str = field(default_factory=lambda: find_tool("ffprobe"))
    fit_workers: int | None = None
    device: str | None = None
    ball_model: Path | None = None
    poll_seconds: float = 5.0
    stage_weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    runner: str = DEFAULT_RUNNER  # "module:function"; tests point it at a fake


def parse_weights(text: str) -> dict[str, float]:
    """"tracking=50,ball3d=15" over the defaults."""
    weights = dict(DEFAULT_WEIGHTS)
    for item in filter(None, (part.strip() for part in text.split(","))):
        stage, _, value = item.partition("=")
        if stage not in STAGES or float(value) < 0:
            raise ValueError(f"bad stage weight {item!r}; stages are {', '.join(STAGES)}")
        weights[stage] = float(value)
    if sum(weights.values()) <= 0:
        raise ValueError("the stage weights sum to zero")
    return weights


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Each flag defaults to its environment variable. argparse applies `type` to a string default too,
    so a bad value from the environment is reported like a bad flag."""
    env = os.environ.get
    parser.add_argument("--server-url", default=env("COURTSHOT_SERVER_URL", ""))
    parser.add_argument("--token", default=env("COURTSHOT_WORKER_TOKEN", ""),
                        help="worker token; prefer COURTSHOT_WORKER_TOKEN, flags show in ps")
    parser.add_argument("--work-root", type=Path, default=env("COURTSHOT_WORK_ROOT", str(DEFAULT_WORK_ROOT)))
    parser.add_argument("--ffmpeg", default=env("COURTSHOT_FFMPEG") or find_tool("ffmpeg"))
    parser.add_argument("--ffprobe", default=env("COURTSHOT_FFPROBE") or find_tool("ffprobe"))
    parser.add_argument("--fit-workers", type=_positive_int, default=env("COURTSHOT_FIT_WORKERS") or None)
    parser.add_argument("--device", type=_device, default=env("COURTSHOT_DEVICE") or None, help="mps or cpu")
    parser.add_argument("--ball-model", type=Path, default=env("COURTSHOT_BALL_MODEL") or None)
    parser.add_argument("--poll-seconds", type=float, default=env("COURTSHOT_POLL_SECONDS", "5"))
    parser.add_argument("--stage-weights", type=parse_weights, default=env("COURTSHOT_STAGE_WEIGHTS", ""),
                        help="e.g. tracking=50,ball3d=15 (the rest keep their defaults)")


def settings_from(args: argparse.Namespace) -> Settings:
    return Settings(
        server_url=args.server_url.rstrip("/"),
        token=args.token,
        work_root=_absolute(args.work_root),
        ffmpeg=_absolute_command(args.ffmpeg),
        ffprobe=_absolute_command(args.ffprobe),
        fit_workers=args.fit_workers,
        device=args.device,
        ball_model=_absolute(args.ball_model) if args.ball_model else None,
        poll_seconds=args.poll_seconds,
        stage_weights=args.stage_weights,
    )


def _absolute(path: Path) -> Path:
    """The job runs in its match folder, so a path relative to the worker's folder would point elsewhere."""
    return path.expanduser().absolute()


def _absolute_command(command: str) -> str:
    """A bare name is looked up on PATH; a path is made absolute (symlinks kept: Homebrew upgrades move targets)."""
    return str(_absolute(Path(command))) if "/" in command else command


def _positive_int(value: str) -> int:
    if not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got {value!r}")
    return int(value)


def _device(value: str) -> str:
    if value not in ("mps", "cpu"):
        raise argparse.ArgumentTypeError(f"expected mps or cpu, got {value!r}")
    return value
