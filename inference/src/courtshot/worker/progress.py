"""Job stages and their weights, and the small JSON files the worker and its job share.

Standard library only: the job subprocess imports it.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

STAGES = ("download", "prepare", "tracking", "appearance", "analysis", "ball3d", "render", "upload")

# ponytail: guesses until the first real timings on Apple Silicon; a setting (COURTSHOT_STAGE_WEIGHTS).
DEFAULT_WEIGHTS: dict[str, float] = {
    "download": 5,
    "prepare": 3,
    "tracking": 45,
    "appearance": 8,
    "analysis": 4,
    "ball3d": 20,
    "render": 10,
    "upload": 5,
}


def overall(weights: Mapping[str, float], stage: str, fraction: float) -> float:
    """Overall progress 0..1: every stage before `stage` done, `stage` itself at `fraction`."""
    index = STAGES.index(stage)
    done = sum(weights[s] for s in STAGES[:index]) + weights[stage] * min(max(fraction, 0.0), 1.0)
    return round(done / sum(weights.values()), 4)


def write_json(path: Path, data: Any) -> None:
    """Write JSON under a temporary name, then rename: a reader never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False))
    os.replace(tmp, path)


class ProgressFile:
    """progress.json: {"stage", "progress"}, progress overall. Cheap to call often: writes are throttled."""

    def __init__(self, path: Path, weights: Mapping[str, float], min_interval: float = 0.5) -> None:
        self.path = path
        self.weights = weights
        self.min_interval = min_interval
        self._stage: str | None = None
        self._written_at = 0.0

    def update(self, stage: str, fraction: float) -> None:
        if stage not in STAGES:
            return
        now = time.monotonic()
        if stage == self._stage and fraction < 1.0 and now - self._written_at < self.min_interval:
            return
        write_json(self.path, {"stage": stage, "progress": overall(self.weights, stage, fraction)})
        self._stage, self._written_at = stage, now


def read_progress(path: Path) -> tuple[str, float] | None:
    try:
        data = json.loads(path.read_text())
        return str(data["stage"]), float(data["progress"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
