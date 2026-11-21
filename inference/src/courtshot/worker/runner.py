"""The worker's parent process: claim a job, download its video, run it in a subprocess, deliver, repeat.

The parent talks to the server; the job runs in `python -m courtshot.worker.job_main` in its own process
group, so a cancel kills the whole group (3D-fit pool and ffmpeg included) and every job starts with
fresh memory.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, TypeVar

from courtshot.worker import procgroup
from courtshot.worker.client import JobGone, JobRef, ServerClient, ServerUnavailable
from courtshot.worker.config import Settings
from courtshot.worker.host import pipeline_version, worker_info
from courtshot.worker.progress import ProgressFile, read_progress, write_json
from courtshot.worker.transfer import download_video, upload_file

logger = logging.getLogger(__name__)

T = TypeVar("T")

VIDEO_EXTENSIONS = ("mp4", "mov", "m4v", "webm")
REQUIRED_FILES = ("report.json", "tracked.mp4", "plain.mp4", "summary.json", "positions.csv")
RESULT_FILES = (*REQUIRED_FILES, "scene.json", "shots.csv", "ball_events.csv")
EXIT_JOB_FAILURE = 3
LOG_TAIL_CHARS = 8000
WORK_FOLDER_MAX_AGE_S = 7 * 24 * 3600
MAX_POLL_SECONDS = 60.0
GROUP_RECORD = "job-group.json"
LOCK_FILE = "worker.lock"
# A heartbeat whose answer never comes gives up within 2 x 25 s + 1 s; with 15 s between beats the next one
# still lands inside the 90 s lease.
HEARTBEAT_TIMEOUT = (5.0, 20.0)
HEARTBEAT_TRIES = 2


class Stopping(Exception):
    """The worker was asked to stop (SIGTERM or SIGINT)."""


class WorkRootBusy(Exception):
    """Another worker holds the work root's lock."""


@dataclass(frozen=True)
class JobPaths:
    """<work_root>/match-<id>/ survives failed attempts (video, caches); one attempt-<job>-<n>/ inside."""

    match: Path
    attempt: Path
    source: Path

    @classmethod
    def for_claim(cls, work_root: Path, claim: dict[str, Any]) -> JobPaths:
        ext = str(claim["video"]["ext"]).lower()
        if ext not in VIDEO_EXTENSIONS:
            raise ValueError(f"unexpected video extension {ext!r}")
        match = work_root / f"match-{int(claim['match_id'])}"
        attempt = match / f"attempt-{int(claim['job_id'])}-{int(claim['attempt'])}"
        return cls(match, attempt, match / f"source.{ext}")

    @property
    def prepared(self) -> Path:
        return self.match / "prepared.mp4"

    @property
    def cache(self) -> Path:
        return self.match / "cache"

    @property
    def group_record(self) -> Path:
        return self.match / GROUP_RECORD

    @property
    def job_file(self) -> Path:
        return self.attempt / "job.json"

    @property
    def progress(self) -> Path:
        return self.attempt / "progress.json"

    @property
    def failure(self) -> Path:
        return self.attempt / "failure.json"

    @property
    def log(self) -> Path:
        return self.attempt / "job.log"

    @property
    def out(self) -> Path:
        return self.attempt / "out"

    def create(self) -> None:
        """A fresh attempt folder; earlier attempts' outputs go, the video and caches stay."""
        for earlier in self.match.glob("attempt-*"):
            shutil.rmtree(earlier, ignore_errors=True)
        self.out.mkdir(parents=True)
        self.cache.mkdir(exist_ok=True)
        os.utime(self.match)  # the age sweep reads this


class Runner:
    def __init__(
        self,
        settings: Settings,
        client: ServerClient | None = None,
        *,
        kill_grace: float = 10.0,
        retry_pause: float = 5.0,
        max_retry_pause: float = 300.0,
    ) -> None:
        self.settings = settings
        self.client = client or ServerClient(settings.server_url, settings.token)
        self.kill_grace = kill_grace
        self.retry_pause = retry_pause
        self.max_retry_pause = max_retry_pause
        self.version = pipeline_version()
        self.stop_requested = threading.Event()

    def stop(self) -> None:
        """Stop claiming; a running job is killed and reported as retryable. Safe from a signal handler."""
        self.stop_requested.set()

    def serve(self) -> None:
        """Claim and run jobs, one at a time, until stop(). Raises WorkRootBusy if another worker uses the work root."""
        with self._lock_work_root():
            self.clean_work_root()
            awake = _keep_awake()
            info = worker_info(self.settings, self.version)
            failures = 0
            try:
                while not self.stop_requested.is_set():
                    try:
                        claim = self.client.claim(info)
                        failures = 0
                    except Exception as exc:  # the server is down or refuses us: keep polling, slower
                        logger.warning("claim failed: %s", exc)
                        claim, failures = None, failures + 1
                    if claim is not None:
                        self.run_job(claim)
                        self.sweep_old_folders()
                        continue
                    delay = min(self.settings.poll_seconds * 2 ** min(failures, 16),
                                max(self.settings.poll_seconds, MAX_POLL_SECONDS))
                    self.stop_requested.wait(delay * random.uniform(0.8, 1.2))
            finally:
                if awake is not None:
                    awake.terminate()

    def _lock_work_root(self) -> IO[str]:
        """One worker per work root: a second one would kill the first one's job as a leftover at startup."""
        self.settings.work_root.mkdir(parents=True, exist_ok=True)
        lock = (self.settings.work_root / LOCK_FILE).open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise WorkRootBusy(f"another worker is using {self.settings.work_root}") from None
        return lock

    def run_job(self, claim: dict[str, Any]) -> None:
        """One attempt, start to end. Never raises: every outcome is reported to the server or logged."""
        try:
            run = JobRun(self, claim)
        except Exception as exc:  # a malformed claim or an unwritable work folder; the lease requeues the job
            logger.error("cannot start a job for claim %r: %s", claim, exc)
            return
        run.execute()

    def clean_work_root(self) -> None:
        """At startup: kill job process groups a previous run left behind, then drop old work folders."""
        self.settings.work_root.mkdir(parents=True, exist_ok=True)
        for record in self.settings.work_root.glob(f"match-*/{GROUP_RECORD}"):
            procgroup.kill_recorded(record, self.kill_grace)
        self.sweep_old_folders()

    def sweep_old_folders(self) -> None:
        cutoff = time.time() - WORK_FOLDER_MAX_AGE_S
        for folder in self.settings.work_root.glob("match-*"):
            if folder.is_dir() and folder.stat().st_mtime < cutoff:
                logger.info("removing old work folder %s", folder)
                shutil.rmtree(folder, ignore_errors=True)


class JobRun:
    """One attempt of one job on this Mac."""

    def __init__(self, runner: Runner, claim: dict[str, Any]) -> None:
        self.runner = runner
        self.settings = runner.settings
        self.client = runner.client
        self.claim = claim
        self.job = JobRef(int(claim["job_id"]), int(claim["attempt"]))
        self.paths = JobPaths.for_claim(self.settings.work_root, claim)
        self.paths.create()
        self.progress = ProgressFile(self.paths.progress, self.settings.stage_weights)
        self.progress.update("download", 0.0)
        self.beat = Heartbeat(self.client.copy(timeout=HEARTBEAT_TIMEOUT, tries=HEARTBEAT_TRIES), self.job,
                              self.paths.progress, float(claim["heartbeat_seconds"]))

    def execute(self) -> None:
        logger.info("job %d attempt %d: match %s", self.job.id, self.job.attempt, self.claim["match_id"])
        self.beat.start()
        try:
            self._download()
            code = self._process()
            if code == 0:
                self._deliver()
                shutil.rmtree(self.paths.match, ignore_errors=True)
                logger.info("job %d delivered", self.job.id)
            else:
                self._report_failure(*self._outcome(code))
        except JobGone as gone:
            self._drop(gone)
        except Stopping:
            self._report_failure("the worker was stopped", retryable=True, persist=False)
        except Exception as exc:
            logger.exception("job %d failed in the worker", self.job.id)
            self._report_failure(f"worker error: {type(exc).__name__}: {exc}", retryable=True)
        finally:
            self.beat.stop()

    def _drop(self, gone: JobGone) -> None:
        logger.warning("job %d: %s; dropping it", self.job.id, gone)
        if gone.status in (404, 410):
            shutil.rmtree(self.paths.match, ignore_errors=True)

    def _checkpoint(self) -> None:
        """Raise when the server dropped the job or the worker is stopping."""
        if self.beat.gone is not None:
            raise self.beat.gone
        self._check_stop()

    def _check_stop(self) -> None:
        if self.runner.stop_requested.is_set():
            raise Stopping()

    def _download(self) -> None:
        size = int(self.claim["video"]["size_bytes"])
        self._persist(lambda: download_video(
            self.client, self.job, self.paths.source, size, self._checkpoint,
            lambda fraction: self.progress.update("download", fraction),
        ))

    def _process(self) -> int:
        """Run job_main in its own process group; returns its exit code (negative: killed by a signal)."""
        limits = self.claim["limits"]
        write_json(self.paths.job_file, {
            "video": str(self.paths.source),
            "prepared": str(self.paths.prepared),
            "calibration": self.claim["calibration"],
            "limits": limits,
            "work_dir": str(self.paths.cache),
            "out_dir": str(self.paths.out),
            "progress": str(self.paths.progress),
            "failure": str(self.paths.failure),
            "ffmpeg": self.settings.ffmpeg,
            "ffprobe": self.settings.ffprobe,
            "stage_weights": dict(self.settings.stage_weights),
            "runner": self.settings.runner,
            "pipeline_version": self.runner.version,
            "device": self.settings.device,
            "fit_workers": self.settings.fit_workers,
            "ball_model": str(self.settings.ball_model) if self.settings.ball_model else None,
        })
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before PyTorch is imported
        with self.paths.log.open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", procgroup.JOB_MODULE, str(self.paths.job_file)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                cwd=self.paths.match, env=env, start_new_session=True,
            )
        procgroup.record(self.paths.group_record, process.pid, self.paths.job_file)
        try:
            while process.poll() is None:
                self._checkpoint()
                self.runner.stop_requested.wait(0.2)
        finally:
            # Also after job_main ended by itself: a SIGKILLed leader leaves its 3D-fit pool and ffmpeg behind.
            procgroup.kill_group(process.pid, self.runner.kill_grace, leader=process)
            self.paths.group_record.unlink(missing_ok=True)
        return process.returncode

    def _outcome(self, code: int) -> tuple[str, bool]:
        """(error, retryable) for a job that did not succeed."""
        failure = _read_json(self.paths.failure)
        if code == EXIT_JOB_FAILURE:  # a reason for the user; retryable only if the job says so
            return str(failure.get("error") or "the job failed"), bool(failure.get("retryable", False))
        if code < 0:
            return f"the job was killed ({signal.strsignal(-code) or -code})", True
        return str(failure.get("error") or f"the job crashed (exit code {code})"), True

    def _deliver(self) -> None:
        out = self.paths.out
        missing = [name for name in REQUIRED_FILES if not (out / name).is_file()]
        if missing:
            raise RuntimeError(f"the job did not write {', '.join(missing)}")
        unexpected = sorted(p.name for p in out.iterdir() if p.name not in RESULT_FILES)
        if unexpected:
            logger.warning("job %d: not uploading unexpected files %s", self.job.id, unexpected)
        files = [out / name for name in RESULT_FILES if (out / name).is_file()]
        total = sum(f.stat().st_size for f in files) or 1
        held: dict[str, int] = {}

        def tracker(name: str) -> Callable[[int], None]:
            def update(offset: int) -> None:
                held[name] = offset
                self.progress.update("upload", sum(held.values()) / total)
            return update

        def send_all() -> None:
            chunk_bytes = int(self.claim["chunk_bytes"])
            for file in files:
                upload_file(self.client, self.job, file, chunk_bytes, self._checkpoint, tracker(file.name))

        self._persist(send_all)
        # If an earlier `complete` got through, heartbeats and file routes answer 409 while `complete` answers
        # 200 again: only its own answer tells a lost answer from a lost job.
        self._persist(lambda: self.client.complete(self.job), checkpoint=self._check_stop)

    def _persist(self, action: Callable[[], T], checkpoint: Callable[[], None] | None = None) -> T:
        """Run `action` until it gets through, pausing longer after each network failure.

        Heartbeats keep the lease meanwhile; if the server gives the job away, `checkpoint` (by default
        _checkpoint) or the next call says so.
        """
        pause = self.runner.retry_pause
        while True:
            try:
                return action()
            except ServerUnavailable as exc:
                logger.warning("job %d: %s; trying again in %.0f s", self.job.id, exc, pause)
                self.runner.stop_requested.wait(pause)
                (checkpoint or self._checkpoint)()
                pause = min(2 * pause, self.runner.max_retry_pause)

    def _report_failure(self, error: str, retryable: bool, *, persist: bool = True) -> None:
        logger.warning("job %d failed (%s): %s", self.job.id, "retryable" if retryable else "final", error)
        tail = _log_tail(self.paths.log)
        try:
            if persist:
                self._persist(lambda: self.client.fail(self.job, error, retryable, tail))
            else:
                self.client.fail(self.job, error, retryable, tail)
        except JobGone as gone:  # no longer ours to fail
            self._drop(gone)
        except Exception as exc:  # the lease expires on the server and the job is queued again
            logger.error("job %d: could not report the failure: %s", self.job.id, exc)


class Heartbeat(threading.Thread):
    """Posts the job's stage and progress (read from progress.json) every `interval` seconds."""

    def __init__(self, client: ServerClient, job: JobRef, progress_path: Path, interval: float) -> None:
        super().__init__(name=f"heartbeat-{job.id}", daemon=True)
        self.client = client
        self.job = job
        self.progress_path = progress_path
        self.interval = interval
        self.gone: JobGone | None = None
        self._halt = threading.Event()

    def run(self) -> None:
        shown: tuple[str, int] | None = None
        while self.gone is None:
            stage, progress = read_progress(self.progress_path) or ("download", 0.0)
            # a live status line in the worker log, only when the stage or the whole percent changes
            if (stage, int(progress * 100)) != shown:
                shown = (stage, int(progress * 100))
                logger.info("job %d: %s, %d%% overall", self.job.id, *shown)
            try:
                self.client.heartbeat(self.job, stage, progress)
            except JobGone as exc:
                self.gone = exc
            except Exception as exc:  # keep beating; the lease tolerates a few misses
                logger.warning("job %d: heartbeat failed: %s", self.job.id, exc)
            if self._halt.wait(self.interval):
                return

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5)


def _keep_awake() -> subprocess.Popen[bytes] | None:
    """caffeinate -i -w <our pid>: no idle sleep while the worker runs, and it ends with us."""
    try:
        return subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    except OSError:
        logger.warning("caffeinate is not available; the Mac may go to sleep during a job")
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _log_tail(path: Path) -> str:
    try:
        with path.open("rb") as log:
            log.seek(max(0, log.seek(0, os.SEEK_END) - 4 * LOG_TAIL_CHARS))
            return log.read().decode("utf-8", "replace")[-LOG_TAIL_CHARS:]
    except OSError:
        return ""
