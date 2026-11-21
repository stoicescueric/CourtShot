"""Background sweeps of the server process: lease expiry every 5 s, cleanup every 10 min."""

import asyncio
import logging
from collections.abc import Callable
from datetime import timedelta

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import delete, or_, select

from .auth import SESSION_IDLE
from .db import Database, utcnow
from .jobs import expire_leases, newest_jobs
from .models import Job, Match, Session
from .settings import Settings
from .storage import Storage

LEASE_SWEEP_SECONDS = 5.0
CLEANUP_SECONDS = 600.0

log = logging.getLogger(__name__)


async def run(db: Database, storage: Storage, settings: Settings) -> None:
    await asyncio.gather(
        _every(LEASE_SWEEP_SECONDS, lambda: expire_leases(db, settings)),
        _every(CLEANUP_SECONDS, lambda: cleanup(db, storage, settings)),
    )


async def _every(seconds: float, sweep: Callable[[], object]) -> None:
    while True:
        try:
            await run_in_threadpool(sweep)
        except Exception:
            log.exception("sweep failed")
        await asyncio.sleep(seconds)


def cleanup(db: Database, storage: Storage, settings: Settings) -> None:
    now = utcnow()
    with db.write() as s:
        abandoned = s.scalars(
            delete(Match)
            .where(
                Match.upload_complete.is_(False),
                Match.created_at < now - timedelta(days=settings.incomplete_upload_days),
            )
            .returning(Match.id)
        ).all()
        s.execute(delete(Session).where(or_(Session.expires_at <= now, Session.last_seen_at <= now - SESSION_IDLE)))
    for match_id in abandoned:
        storage.delete_match_dir(match_id)

    # List the folders before reading the rows: a folder is only made after its row commits,
    # and a run folder only after its attempt started running.
    match_folders = storage.match_ids_on_disk()
    run_folders = storage.run_folders()
    with db.read() as s:
        known = set(s.scalars(select(Match.id)))
        keep = {(job.match_id, f"{job.id}-{job.attempt}") for job in newest_jobs(s, succeeded=True).values()}
        keep |= {
            (match_id, f"{job_id}-{attempt}")
            for match_id, job_id, attempt in s.execute(
                select(Job.match_id, Job.id, Job.attempt).where(Job.status == "running")
            )
        }
    for match_id in match_folders:
        if match_id not in known:
            storage.delete_match_dir(match_id)
    for match_id, folders in run_folders.items():
        for folder in folders:
            if match_id in known and (match_id, folder) not in keep:
                storage.delete_run_folder(match_id, folder)
