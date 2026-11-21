"""Browser routes about matches: /api/matches/..."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from starlette.responses import FileResponse

from .. import jobs, matches
from ..auth import check_origin, current_user
from ..db import Database
from ..deps import get_db, get_settings, get_storage, read_body
from ..models import Job, Match
from ..schemas import CalibrationBody, NewMatch, TitleBody
from ..settings import Settings
from ..storage import RESULT_FILES, Storage
from ..views import match_detail, match_list

router = APIRouter(prefix="/api/matches", dependencies=[Depends(check_origin), Depends(current_user)])

Db = Depends(get_db)
Files = Depends(get_storage)
Config = Depends(get_settings)


@router.get("")
def list_matches(db: Database = Db, storage: Storage = Files, settings: Settings = Config) -> dict[str, Any]:
    return match_list(db, storage, settings)


@router.post("", status_code=201)
def create_match(
    body: NewMatch, db: Database = Db, storage: Storage = Files, settings: Settings = Config
) -> dict[str, Any]:
    match_id = matches.create(db, storage, settings, body)
    return match_detail(db, storage, match_id)


@router.get("/{match_id}")
def get_match(match_id: int, db: Database = Db, storage: Storage = Files) -> dict[str, Any]:
    return match_detail(db, storage, match_id)


@router.patch("/{match_id}")
def rename_match(match_id: int, body: TitleBody, db: Database = Db, storage: Storage = Files) -> dict[str, Any]:
    matches.rename(db, match_id, body.title)
    return match_detail(db, storage, match_id)


@router.delete("/{match_id}", status_code=204)
def delete_match(match_id: int, db: Database = Db, storage: Storage = Files) -> Response:
    matches.delete_match(db, storage, match_id)
    return Response(status_code=204)


def _match(db: Database, match_id: int) -> Match:
    with db.read() as s:
        return matches.get_match(s, match_id)


@router.get("/{match_id}/upload")
def upload_state(match_id: int, db: Database = Db, storage: Storage = Files) -> dict[str, Any]:
    match = _match(db, match_id)
    offset = match.size_bytes if match.upload_complete else storage.upload_offset(match.id, match.source_ext)
    return {
        "offset": offset,
        "size_bytes": match.size_bytes,
        "complete": match.upload_complete,
        "fingerprint": match.fingerprint,
    }


@router.patch("/{match_id}/upload")
async def upload_chunk(
    match_id: int,
    request: Request,
    offset: int = Query(ge=0),
    db: Database = Db,
    storage: Storage = Files,
    settings: Settings = Config,
) -> dict[str, Any]:
    match = await run_in_threadpool(_match, db, match_id)
    if match.upload_complete:
        return {"offset": match.size_bytes, "complete": True}
    data = await read_body(request, settings.max_chunk_bytes)  # before any lock is taken
    size = await run_in_threadpool(matches.write_upload, db, storage, match, offset, data)
    return {"offset": size, "complete": size == match.size_bytes}


@router.put("/{match_id}/calibration")
def save_calibration(
    match_id: int, body: CalibrationBody, db: Database = Db, storage: Storage = Files, settings: Settings = Config
) -> dict[str, Any]:
    matches.save_calibration(db, settings, match_id, body)
    return match_detail(db, storage, match_id)


@router.post("/{match_id}/jobs", status_code=201)
def run_again(match_id: int, db: Database = Db, storage: Storage = Files) -> dict[str, Any]:
    matches.run_again(db, match_id)
    return match_detail(db, storage, match_id)


@router.post("/{match_id}/cancel")
def cancel(match_id: int, db: Database = Db, storage: Storage = Files) -> dict[str, Any]:
    jobs.cancel(db, match_id)
    return match_detail(db, storage, match_id)


@router.get("/{match_id}/video")
def video(match_id: int, db: Database = Db, storage: Storage = Files) -> FileResponse:
    match = _match(db, match_id)
    if not match.upload_complete:
        raise HTTPException(409, "The upload is not complete")
    return storage.video_response(match.id, match.source_ext)


@router.get("/{match_id}/runs/{job_id}/{name}")
def result_file(match_id: int, job_id: int, name: str, db: Database = Db, storage: Storage = Files) -> FileResponse:
    if name not in RESULT_FILES:
        raise HTTPException(404, "Unknown result file")
    with db.read() as s:
        job = s.get(Job, job_id)
    if job is None or job.match_id != match_id or job.status != "succeeded":
        raise HTTPException(404, "No such run")
    return storage.result_response(match_id, job.id, job.attempt, name)
