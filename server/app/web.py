"""Security headers on every response, and the built dashboard at /."""

from contextlib import suppress
from pathlib import Path

from fastapi import APIRouter, HTTPException
from starlette.datastructures import MutableHeaders
from starlette.responses import FileResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
        "style-src 'self'; script-src 'self'; connect-src 'self'; font-src 'self'; "
        "worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    ),
}

NOT_BUILT = "The CourtShot dashboard is not built. Run `npm ci && npm run build` in server/web.\n"


class SecurityHeaders:
    """Pure ASGI middleware, so streamed and ranged file responses pass through untouched."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)

        await self.app(scope, receive, send_with_headers)


def dashboard_router(web_dist: Path) -> APIRouter:
    """Files of the built dashboard; any other path outside /api gets index.html."""
    router = APIRouter()

    @router.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def dashboard(path: str) -> Response:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "Not Found")
        root = web_dist.resolve()
        index = root / "index.html"
        if not index.is_file():
            return PlainTextResponse(NOT_BUILT)
        with suppress(OSError, ValueError):  # a NUL byte or a name too long for the file system
            candidate = (root / path).resolve()
            if candidate != index and candidate.is_relative_to(root) and candidate.is_file():
                return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return router
