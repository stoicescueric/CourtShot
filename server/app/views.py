"""Response shapes of server/API.md, built from the stored facts."""

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Database, iso, utcnow
from .jobs import newest_jobs
from .matches import get_match
from .models import Job, Match, Worker
from .settings import Settings
from .storage import Storage

MATCH_STATUS_OF_JOB = {
    "queued": "queued",
    "running": "processing",
    "succeeded": "ready",
    "failed": "failed",
    "cancelled": "cancelled",
}


def match_status(upload_complete: bool, has_calibration: bool, newest_job_status: str | None) -> str:
    """The one place a match's status comes from (design 7)."""
    if not upload_complete:
        return "uploading"
    if not has_calibration or newest_job_status is None:
        # With a calibration and a complete upload a job always exists; this is the fallback.
        return "needs_calibration"
    return MATCH_STATUS_OF_JOB[newest_job_status]


def job_json(job: Job, worker_names: dict[int, str], with_log: bool) -> dict[str, Any]:
    shape = {
        "id": job.id,
        "status": job.status,
        "attempt": job.attempt,
        "stage": job.stage,
        "progress": job.progress,
        "error": job.error,
        "worker": worker_names.get(job.worker_id) if job.worker_id is not None else None,
        "created_at": iso(job.created_at),
        "started_at": iso(job.started_at),
        "finished_at": iso(job.finished_at),
    }
    if with_log:
        shape["log_tail"] = job.log_tail
    return shape


def match_json(
    match: Match,
    newest: Job | None,
    current: Job | None,
    worker_names: dict[int, str],
    storage: Storage,
    detail: bool,
) -> dict[str, Any]:
    offset = match.size_bytes if match.upload_complete else storage.upload_offset(match.id, match.source_ext)
    shape = {
        "id": match.id,
        "title": match.title,
        "original_filename": match.original_filename,
        "created_at": iso(match.created_at),
        "size_bytes": match.size_bytes,
        "duration_s": match.duration_s,
        "frame_width": match.frame_width,
        "frame_height": match.frame_height,
        "status": match_status(match.upload_complete, match.calibration is not None, newest.status if newest else None),
        "upload": {"offset": offset, "complete": match.upload_complete},
        "has_calibration": match.calibration is not None,
        "job": job_json(newest, worker_names, with_log=detail) if newest else None,
        "results": (
            {
                "job_id": current.id,
                "files": storage.result_names(match.id, current.id, current.attempt),
                "finished_at": iso(current.finished_at),
            }
            if current
            else None
        ),
        "summary": match.summary,
    }
    if detail:
        shape["fingerprint"] = match.fingerprint
        shape["calibration"] = match.calibration
    return shape


def _worker_names(s: Session) -> dict[int, str]:
    return {worker_id: name for worker_id, name in s.execute(select(Worker.id, Worker.name))}


def match_detail(db: Database, storage: Storage, match_id: int) -> dict[str, Any]:
    with db.read() as s:
        match = get_match(s, match_id)
        newest = newest_jobs(s, match_id).get(match_id)
        current = newest_jobs(s, match_id, succeeded=True).get(match_id)
        return match_json(match, newest, current, _worker_names(s), storage, detail=True)


def match_list(db: Database, storage: Storage, settings: Settings) -> dict[str, Any]:
    with db.read() as s:
        newest = newest_jobs(s)
        current = newest_jobs(s, succeeded=True)
        names = _worker_names(s)
        matches = [
            match_json(m, newest.get(m.id), current.get(m.id), names, storage, detail=False)
            for m in s.scalars(select(Match).order_by(Match.id.desc()))
        ]
        online = s.scalar(
            select(Worker.id)
            .where(Worker.revoked_at.is_(None), Worker.last_seen_at >= _online_cutoff(settings))
            .limit(1)
        )
    return {"matches": matches, "workers_online": online is not None}


def worker_list(db: Database, settings: Settings) -> dict[str, Any]:
    cutoff = _online_cutoff(settings)
    with db.read() as s:
        workers = s.scalars(select(Worker).where(Worker.revoked_at.is_(None)).order_by(Worker.name)).all()
        running = {
            job.worker_id: (job, title)
            for job, title in s.execute(
                select(Job, Match.title).join(Match, Match.id == Job.match_id).where(Job.status == "running")
            )
        }
    shapes = []
    for worker in workers:
        job, title = running.get(worker.id, (None, None))
        shapes.append(
            {
                "name": worker.name,
                "last_seen_at": iso(worker.last_seen_at),
                "online": worker.last_seen_at is not None and worker.last_seen_at >= cutoff,
                "job": (
                    {
                        "id": job.id,
                        "match_id": job.match_id,
                        "match_title": title,
                        "stage": job.stage,
                        "progress": job.progress,
                    }
                    if job
                    else None
                ),
            }
        )
    return {"workers": shapes}


def _online_cutoff(settings: Settings) -> datetime:
    return utcnow() - timedelta(seconds=settings.worker_online_seconds)
