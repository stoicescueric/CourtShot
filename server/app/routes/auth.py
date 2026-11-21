"""POST /api/auth/login, POST /api/auth/logout, GET /api/auth/me."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import delete, select

from .. import models
from ..auth import (
    SESSION_MAX,
    check_origin,
    client_address,
    create_session,
    current_user,
    token_hash,
    verify_password,
)
from ..db import Database
from ..deps import get_db, get_settings
from ..schemas import LoginBody
from ..settings import Settings

router = APIRouter(prefix="/api/auth", dependencies=[Depends(check_origin)])


@router.post("/login")
def login(
    body: LoginBody,
    request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
    db: Database = Depends(get_db),
) -> dict[str, str]:
    throttle = request.app.state.throttle
    address = client_address(request, settings.trusted_proxy)
    stamp = throttle.admit(address)  # before any hashing
    with db.read() as s:
        user = s.scalar(select(models.User).where(models.User.username == body.username))
    if not verify_password(user.password_hash if user else None, body.password) or user is None:
        raise HTTPException(401, "Wrong username or password")
    throttle.forgive(address, stamp)
    response.set_cookie(
        settings.session_cookie,
        create_session(db, user.id),
        max_age=int(SESSION_MAX.total_seconds()),
        path="/",
        secure=settings.secure_cookies,
        httponly=True,
        samesite="strict",
    )
    return {"username": user.username}


@router.post("/logout", status_code=204)
def logout(
    request: Request,
    settings: Settings = Depends(get_settings),
    db: Database = Depends(get_db),
) -> Response:
    token = request.cookies.get(settings.session_cookie)
    if token:
        with db.write() as s:
            s.execute(delete(models.Session).where(models.Session.token_hash == token_hash(token)))
    response = Response(status_code=204)
    response.delete_cookie(
        settings.session_cookie, path="/", secure=settings.secure_cookies, httponly=True, samesite="strict"
    )
    return response


@router.get("/me")
def me(user: models.User = Depends(current_user)) -> dict[str, str]:
    return {"username": user.username}
