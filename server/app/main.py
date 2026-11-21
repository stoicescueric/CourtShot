"""The FastAPI application. Run exactly one process: `uvicorn app.main:app --workers 1`."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from . import sweeps
from .auth import LoginThrottle
from .db import Database
from .routes import auth, info, matches, worker
from .settings import Settings
from .storage import NoSpace, OffsetBeyondEnd, Storage, TooLarge
from .web import SecurityHeaders, dashboard_router


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()  # type: ignore[call-arg]  # DATA_DIR comes from the environment
    db = Database(settings.db_path)
    storage = Storage(settings.data_dir, settings.free_space_reserve_bytes)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if not db.is_migrated():
            raise RuntimeError(f"{settings.db_path} is not migrated: run `uv run python -m app.cli migrate`")
        task = asyncio.create_task(sweeps.run(db, storage, settings))
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            db.close()

    app = FastAPI(title="CourtShot", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.db = db
    app.state.storage = storage
    app.state.throttle = LoginThrottle()
    app.add_middleware(SecurityHeaders)
    _add_error_handlers(app)
    for module in (auth, info, matches, worker):
        app.include_router(module.router)
    app.include_router(dashboard_router(settings.web_dist))  # last: it catches every other GET
    return app


def _error(status: int, detail: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"detail": detail, **extra}, status_code=status)


def _add_error_handlers(app: FastAPI) -> None:
    """Every error answers {"detail": "<message>"} (server/API.md, Conventions)."""

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"] if part != "body")
        return _error(422, f"{where}: {first['msg']}" if where else first["msg"])

    @app.exception_handler(IntegrityError)
    async def conflict(_: Request, exc: IntegrityError) -> JSONResponse:
        return _error(409, "The change conflicts with the current state")

    @app.exception_handler(FileNotFoundError)
    async def missing(_: Request, exc: FileNotFoundError) -> JSONResponse:
        return _error(404, "Not Found")

    @app.exception_handler(OffsetBeyondEnd)
    async def beyond(_: Request, exc: OffsetBeyondEnd) -> JSONResponse:
        return _error(409, str(exc), offset=exc.offset)

    @app.exception_handler(TooLarge)
    async def too_large(_: Request, exc: TooLarge) -> JSONResponse:
        return _error(413, str(exc))

    @app.exception_handler(NoSpace)
    async def no_space(_: Request, exc: NoSpace) -> JSONResponse:
        return _error(507, str(exc))


def __getattr__(name: str) -> FastAPI:
    # `uvicorn app.main:app` builds the app from the environment; importing this module does not.
    if name == "app":
        return create_app()
    raise AttributeError(name)
