"""Worker routes: /api/worker/... Bearer token only; cookies and Origin are ignored."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from starlette.responses import FileResponse

from .. import jobs
from ..auth import current_worker
from ..db import Database
from ..deps import get_db, get_settings, get_storage, read_body
from ..models import Worker
from ..schemas import ClaimBody, FailBody, HeartbeatBody
from ..settings import Settings
from ..storage import RESULT_FILES, Storage

router = APIRouter(prefix="/api/worker")

Db = Depends(get_db)
Files = Depends(get_storage)
Config = Depends(get_settings)
Me = Depends(current_worker)
Attempt = Annotated[int, Header(alias="X-Job-Attempt", ge=1)]


@router.post("/claim", response_model=None)
def claim(body: ClaimBody, worker: Worker = Me, db: Database = Db, settings: Settings = Config) -> Any:
    claimed = jobs.claim(db, worker.id, body.info.model_dump())
    if claimed is None:
        return Response(status_code=204)
    job, match = claimed
    return {
        "job_id": job.id,
        "attempt": job.attempt,
        "match_id": match.id,
        "video": {"size_bytes": match.size_bytes, "ext": match.source_ext},
        "calibration": job.calibration,
        "limits": {
            "min_video_seconds": settings.min_video_seconds,
            "max_video_seconds": settings.max_video_seconds,
            "calibration_max_error_m": settings.calibration_max_error_m,
        },
        "heartbeat_seconds": settings.heartbeat_seconds,
        "chunk_bytes": settings.chunk_bytes,
    }


@router.get("/jobs/{job_id}/video")
def video(
    job_id: int, attempt: Attempt, worker: Worker = Me, db: Database = Db, storage: Storage = Files
) -> FileResponse:
    _, match = jobs.running_job(db, job_id, attempt, worker.id)
    return storage.video_response(match.id, match.source_ext)


@router.post("/jobs/{job_id}/heartbeat")
def heartbeat(job_id: int, attempt: Attempt, body: HeartbeatBody, worker: Worker = Me, db: Database = Db) -> dict:
    jobs.heartbeat(db, job_id, attempt, worker.id, body.stage, body.progress)
    return {}


def _result_name(name: str) -> str:
    if name not in RESULT_FILES:
        raise HTTPException(422, f"Unknown result file {name!r}")
    return name


@router.get("/jobs/{job_id}/files/{name}")
def file_offset(
    job_id: int, name: str, attempt: Attempt, worker: Worker = Me, db: Database = Db, storage: Storage = Files
) -> dict[str, int]:
    _result_name(name)
    job, match = jobs.running_job(db, job_id, attempt, worker.id)
    return {"offset": storage.result_offset(match.id, job.id, attempt, name)}


@router.patch("/jobs/{job_id}/files/{name}")
async def file_chunk(
    job_id: int,
    name: str,
    attempt: Attempt,
    request: Request,
    offset: int = Query(ge=0),
    worker: Worker = Me,
    db: Database = Db,
    storage: Storage = Files,
    settings: Settings = Config,
) -> dict[str, int]:
    _result_name(name)
    job, match = await run_in_threadpool(jobs.running_job, db, job_id, attempt, worker.id)
    data = await read_body(request, settings.max_chunk_bytes)  # before any lock is taken

    def write() -> int:
        storage.ensure_free_space(len(data))
        return storage.write_result_chunk(match.id, job.id, attempt, name, offset, data)

    return {"offset": await run_in_threadpool(write)}


@router.post("/jobs/{job_id}/complete")
def complete(job_id: int, attempt: Attempt, worker: Worker = Me, db: Database = Db, storage: Storage = Files) -> dict:
    jobs.complete(db, storage, job_id, attempt, worker.id)
    return {}


@router.post("/jobs/{job_id}/fail")
def fail(
    job_id: int, attempt: Attempt, body: FailBody, worker: Worker = Me, db: Database = Db, settings: Settings = Config
) -> dict:
    jobs.fail(db, settings, job_id, attempt, worker.id, body.error, body.retryable, body.log_tail)
    return {}
