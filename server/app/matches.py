"""Match state changes made by the browser: create, rename, upload, calibrate, run, delete."""

from fastapi import HTTPException
from sqlalchemy import delete
from sqlalchemy.orm import Session

from . import jobs
from .db import Database, utcnow
from .models import Match
from .schemas import CalibrationBody, NewMatch
from .settings import Settings
from .storage import VIDEO_TYPES, Storage, TooLarge


def get_match(s: Session, match_id: int) -> Match:
    match = s.get(Match, match_id)
    if match is None:
        raise HTTPException(404, "Match not found")
    return match


def create(db: Database, storage: Storage, settings: Settings, body: NewMatch) -> int:
    ext = body.filename.rpartition(".")[2].lower() if "." in body.filename else ""
    if ext not in VIDEO_TYPES:
        raise HTTPException(422, f"The video must be one of: {', '.join(VIDEO_TYPES)}")
    if not settings.min_video_seconds <= body.duration_s <= settings.max_video_seconds:
        raise HTTPException(
            422,
            f"The video must last {settings.min_video_seconds:g} to {settings.max_video_seconds:g} seconds",
        )
    if body.size_bytes > settings.max_upload_bytes:
        raise TooLarge(f"The video may be at most {settings.max_upload_bytes} bytes")
    storage.ensure_free_space(body.size_bytes)
    with db.write() as s:
        match = Match(
            title=body.title,
            original_filename=body.filename,
            source_ext=ext,
            size_bytes=body.size_bytes,
            fingerprint=body.fingerprint,
            upload_complete=False,
            duration_s=body.duration_s,
            frame_width=body.frame_width,
            frame_height=body.frame_height,
            created_at=utcnow(),
        )
        s.add(match)
        s.flush()
        match_id = match.id
    # The folder is made after the row commits, so the cleanup sweep never sees it without one.
    try:
        storage.create_match_dir(match_id)
    except OSError:
        delete_match(db, storage, match_id)
        raise
    return match_id


def rename(db: Database, match_id: int, title: str) -> None:
    with db.write() as s:
        get_match(s, match_id).title = title


def delete_match(db: Database, storage: Storage, match_id: int) -> None:
    """Deletes the match and its jobs (an active job is cancelled with it), then its files."""
    with db.write() as s:
        if s.execute(delete(Match).where(Match.id == match_id)).rowcount == 0:
            raise HTTPException(404, "Match not found")
    storage.delete_match_dir(match_id)


def write_upload(db: Database, storage: Storage, match: Match, offset: int, data: bytes) -> int:
    """Writes one chunk of the video; the chunk that completes it may queue the first job."""
    storage.ensure_free_space(len(data))
    size = storage.write_upload_chunk(match.id, match.source_ext, match.size_bytes, offset, data)
    if size == match.size_bytes:
        with db.write() as s:
            stored = get_match(s, match.id)
            if not stored.upload_complete:
                stored.upload_complete = True
                if stored.calibration is not None:
                    jobs.enqueue(s, stored)
    return size


def save_calibration(db: Database, settings: Settings, match_id: int, body: CalibrationBody) -> None:
    if len(body.points) < settings.min_calibration_points:
        raise HTTPException(422, f"At least {settings.min_calibration_points} points are needed")
    with db.write() as s:
        match = get_match(s, match_id)
        if (body.frame_width, body.frame_height) != (match.frame_width, match.frame_height):
            raise HTTPException(
                422,
                f"The calibration frame is {body.frame_width}x{body.frame_height}, "
                f"the video is {match.frame_width}x{match.frame_height}",
            )
        if jobs.active_job(s, match_id) is not None:
            raise HTTPException(409, "A job is active for this match")
        match.calibration = body.model_dump(mode="json")
        if match.upload_complete:
            jobs.enqueue(s, match)


def run_again(db: Database, match_id: int) -> None:
    with db.write() as s:
        match = get_match(s, match_id)
        if not match.upload_complete:
            raise HTTPException(409, "The upload is not complete")
        if match.calibration is None:
            raise HTTPException(409, "The match has no calibration")
        jobs.enqueue(s, match)
