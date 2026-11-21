"""GET /api/config and GET /api/workers."""

from typing import Any

from fastapi import APIRouter, Depends

from ..auth import check_origin, current_user
from ..db import Database
from ..deps import get_db, get_settings
from ..settings import Settings
from ..storage import VIDEO_TYPES
from ..views import worker_list

router = APIRouter(prefix="/api", dependencies=[Depends(check_origin), Depends(current_user)])


@router.get("/config")
def config(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "min_video_seconds": settings.min_video_seconds,
        "max_video_seconds": settings.max_video_seconds,
        "max_upload_bytes": settings.max_upload_bytes,
        "chunk_bytes": settings.chunk_bytes,
        "calibration_max_error_m": settings.calibration_max_error_m,
        "min_calibration_points": settings.min_calibration_points,
        "extensions": list(VIDEO_TYPES),
        "worker_online_seconds": settings.worker_online_seconds,
    }


@router.get("/workers")
def workers(settings: Settings = Depends(get_settings), db: Database = Depends(get_db)) -> dict[str, Any]:
    return worker_list(db, settings)
