"""Request-scoped access to the application's shared objects, and capped body reading."""

from fastapi import HTTPException, Request

from .db import Database
from .settings import Settings
from .storage import Storage


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


async def read_body(request: Request, cap: int) -> bytes:
    """The whole request body, refused with 413 as soon as it passes `cap` bytes."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > cap:
        raise HTTPException(413, f"A chunk may be at most {cap} bytes")
    body = bytearray()
    async for part in request.stream():
        body += part
        if len(body) > cap:
            raise HTTPException(413, f"A chunk may be at most {cap} bytes")
    return bytes(body)
