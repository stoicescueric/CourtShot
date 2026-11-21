"""The job subprocess: python -m courtshot.worker.job_main <job.json>.

Only the standard library at module level: under the spawn start method every process of the 3D-fit pool
imports this module again, and PyTorch loaded once per process would exhaust a small Mac.

Exit codes, with failure.json {error, retryable} beside them:
0 results written; 3 JobFailure, a reason for the user; 1 crash, traceback in the job log.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

EXIT_OK, EXIT_CRASH, EXIT_JOB_FAILURE = 0, 1, 3
DEFAULT_RUNNER = "courtshot.job:run_job"


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    job = json.loads(Path(argv[0]).read_text())

    from courtshot.errors import JobFailure
    from courtshot.worker.prepare import prepare_video
    from courtshot.worker.progress import ProgressFile, write_json

    progress = ProgressFile(Path(job["progress"]), job["stage_weights"])
    try:
        video = prepare_video(
            Path(job["video"]), Path(job["prepared"]), job["calibration"], job["limits"],
            ffmpeg=job["ffmpeg"], ffprobe=job["ffprobe"], on_progress=lambda f: progress.update("prepare", f),
        )
        run = _load(job.get("runner", DEFAULT_RUNNER))
        names = run(
            video,
            job["calibration"],
            Path(job["work_dir"]),
            Path(job["out_dir"]),
            progress.update,
            pipeline_version=job["pipeline_version"],
            device=job["device"],
            fit_workers=job["fit_workers"],
            max_error_m=job["limits"]["calibration_max_error_m"],
            ball_model=Path(job["ball_model"]) if job["ball_model"] else None,
        )
        logging.info("job done: %s", ", ".join(names))
    except JobFailure as exc:
        logging.error("job failed: %s", exc.message)
        write_json(Path(job["failure"]), {"error": exc.message, "retryable": exc.retryable})
        return EXIT_JOB_FAILURE
    except Exception as exc:
        traceback.print_exc()
        write_json(Path(job["failure"]), {"error": f"{type(exc).__name__}: {exc}"[:500], "retryable": True})
        return EXIT_CRASH
    return EXIT_OK


def _load(target: str) -> Callable[..., Any]:
    """"package.module:function" -> the function."""
    module, _, name = target.partition(":")
    function: Callable[..., Any] = getattr(importlib.import_module(module), name)
    return function


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
