"""The job queue (design 9). Every state change checks its precondition in its own write transaction."""

from datetime import timedelta
from typing import Any

from fastapi import HTTPException
from sqlalchemy import case, func, or_, select, update
from sqlalchemy.orm import Session

from .db import Database, utcnow
from .models import ACTIVE_JOB_STATUSES, Job, Match, Worker
from .settings import Settings
from .storage import Storage

ERROR_MAX = 500
LOG_TAIL_MAX = 8000
LEASE_LOST = "the worker stopped responding"


def active_job(s: Session, match_id: int) -> Job | None:
    return s.scalar(select(Job).where(Job.match_id == match_id, Job.status.in_(ACTIVE_JOB_STATUSES)))


def newest_jobs(s: Session, match_id: int | None = None, succeeded: bool = False) -> dict[int, Job]:
    """The newest job of each match (or of one match), optionally among succeeded jobs only."""
    newest = select(func.max(Job.id)).group_by(Job.match_id)
    if match_id is not None:
        newest = newest.where(Job.match_id == match_id)
    if succeeded:
        newest = newest.where(Job.status == "succeeded")
    return {job.match_id: job for job in s.scalars(select(Job).where(Job.id.in_(newest)))}


def enqueue(s: Session, match: Match) -> None:
    """Queues a job for `match` with a copy of its calibration, inside the caller's write transaction."""
    if active_job(s, match.id) is not None:
        raise HTTPException(409, "A job is already active for this match")
    job = Job(match_id=match.id, status="queued", attempt=0, calibration=match.calibration, created_at=utcnow())
    s.add(job)
    s.flush()  # the partial unique index is the backstop; its IntegrityError answers 409


def claim(db: Database, worker_id: int, info: dict[str, Any]) -> tuple[Job, Match] | None:
    """Hands the oldest claimable job to a worker, or None."""
    now = utcnow()
    with db.write() as s:
        s.execute(update(Worker).where(Worker.id == worker_id).values(last_seen_at=now, info=info))
        job = s.scalar(
            select(Job)
            .where(Job.status == "queued", or_(Job.not_before.is_(None), Job.not_before <= now))
            .order_by(Job.created_at, Job.id)
            .limit(1)
        )
        if job is None:
            return None
        job.status = "running"
        job.attempt += 1
        job.worker_id = worker_id
        job.heartbeat_at = job.started_at = now
        job.stage = job.progress = job.finished_at = None
        return job, s.get_one(Match, job.match_id)


def check_running(job: Job | None, attempt: int, worker_id: int) -> Job:
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "running" or job.attempt != attempt or job.worker_id != worker_id:
        raise HTTPException(409, "This attempt is no longer current")
    return job


def running_job(db: Database, job_id: int, attempt: int, worker_id: int) -> tuple[Job, Match]:
    """Read-only check for routes that change nothing in the database (video, result chunks)."""
    with db.read() as s:
        job = check_running(s.get(Job, job_id), attempt, worker_id)
        return job, s.get_one(Match, job.match_id)


def heartbeat(db: Database, job_id: int, attempt: int, worker_id: int, stage: str, progress: float) -> None:
    now = utcnow()
    with db.write() as s:
        job = check_running(s.get(Job, job_id), attempt, worker_id)
        job.heartbeat_at = now
        job.stage = stage
        job.progress = progress
        s.execute(update(Worker).where(Worker.id == worker_id).values(last_seen_at=now))


def complete(db: Database, storage: Storage, job_id: int, attempt: int, worker_id: int) -> None:
    with db.write() as s:
        job = s.get(Job, job_id)
        if job is not None and job.status == "succeeded" and job.attempt == attempt and job.worker_id == worker_id:
            return  # a repeated complete: the first response was lost
        job = check_running(job, attempt, worker_id)
        missing = storage.missing_required(job.match_id, job.id, attempt)
        if missing:
            raise HTTPException(422, f"Missing result files: {', '.join(missing)}")
        try:
            report = storage.read_result_json(job.match_id, job.id, attempt, "report.json")
        except ValueError:
            report = None
        if not isinstance(report, dict):
            raise HTTPException(422, "report.json is not a JSON object")
        job.status = "succeeded"
        job.progress = 1.0
        job.error = None
        job.finished_at = utcnow()
        # This commit is the switch to the new results (design 8.3).
        s.get_one(Match, job.match_id).summary = summary_from_report(report)


def fail(
    db: Database,
    settings: Settings,
    job_id: int,
    attempt: int,
    worker_id: int,
    error: str,
    retryable: bool,
    log_tail: str | None,
) -> None:
    now = utcnow()
    with db.write() as s:
        job = check_running(s.get(Job, job_id), attempt, worker_id)
        job.error = error[:ERROR_MAX]
        job.log_tail = log_tail[-LOG_TAIL_MAX:] if log_tail else None
        if retryable and job.attempt < settings.max_attempts:
            job.status = "queued"
            job.not_before = now + timedelta(seconds=settings.retry_delay_seconds)
            job.worker_id = job.heartbeat_at = job.stage = job.progress = None
        else:
            job.status = "failed"
            job.finished_at = now


def cancel(db: Database, match_id: int) -> None:
    with db.write() as s:
        if s.get(Match, match_id) is None:
            raise HTTPException(404, "Match not found")
        job = active_job(s, match_id)
        if job is None:
            raise HTTPException(409, "No job is active for this match")
        job.status = "cancelled"
        job.finished_at = utcnow()


def expire_leases(db: Database, settings: Settings) -> int:
    """One UPDATE: running jobs without a recent heartbeat go back to the queue, or fail at the limit."""
    now = utcnow()
    give_up = Job.attempt >= settings.max_attempts

    def keep_if_failed(column: Any) -> Any:
        return case((give_up, column), else_=None)

    with db.write() as s:
        result = s.execute(
            update(Job)
            .where(Job.status == "running", Job.heartbeat_at < now - timedelta(seconds=settings.lease_seconds))
            .values(
                status=case((give_up, "failed"), else_="queued"),
                error=case((give_up, LEASE_LOST), else_=Job.error),
                finished_at=case((give_up, now), else_=None),
                worker_id=keep_if_failed(Job.worker_id),
                heartbeat_at=keep_if_failed(Job.heartbeat_at),
                stage=keep_if_failed(Job.stage),
                progress=keep_if_failed(Job.progress),
            )
            .execution_options(synchronize_session=False)
        )
        return result.rowcount


def summary_from_report(report: dict[str, Any]) -> dict[str, float | int | None]:
    """The list-page numbers of report.json (contract: reference-maps/report-data.md)."""
    totals = report.get("match")
    if isinstance(totals, dict):
        rallies, shots, in_play = totals.get("rallies"), totals.get("shots"), totals.get("time_in_play_s")
    else:
        movement = report.get("movement_rallies")
        rallies = len(movement) if isinstance(movement, list) else None
        shots = in_play = None
    return {
        "duration_s": _number(report.get("duration_s")),
        "rallies": _number(rallies),
        "shots": _number(shots),
        "time_in_play_s": _number(in_play),
    }


def _number(value: Any) -> float | int | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None
