"""Sign-in, sessions and worker tokens (design 8.4)."""

import hashlib
import secrets
import threading
import time
from collections import deque
from datetime import timedelta
from ipaddress import ip_address
from math import ceil
from typing import NoReturn

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Depends, HTTPException, Request
from sqlalchemy import select, update

from . import models
from .db import Database, utcnow
from .deps import get_db, get_settings
from .settings import Settings

SESSION_IDLE = timedelta(days=7)
SESSION_MAX = timedelta(days=30)
SESSION_TOUCH = timedelta(minutes=1)

_hasher = PasswordHasher()  # Argon2id
_hash_slots = threading.Semaphore(2)  # bounds the CPU and memory of hashing
HASH_WAIT_SECONDS = 10


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _hashing_slot() -> None:
    if not _hash_slots.acquire(timeout=HASH_WAIT_SECONDS):
        raise HTTPException(429, "Too many sign-ins at once", headers={"Retry-After": "1"})


def hash_password(password: str) -> str:
    _hashing_slot()
    try:
        return _hasher.hash(password)
    finally:
        _hash_slots.release()


# Verified against when the user is unknown, so that case costs the same time.
_DUMMY_HASH = _hasher.hash(secrets.token_hex(16))


def verify_password(password_hash: str | None, password: str) -> bool:
    """Checks a password; an unknown user (None) costs one verification all the same."""
    _hashing_slot()
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerificationError, InvalidHashError):
        return False
    finally:
        _hash_slots.release()


class LoginThrottle:
    """Failed sign-ins per client address, plus a global failure budget.

    An admitted attempt counts as a failure until it is forgiven on success, so parallel
    requests from one address cannot slip past the limit. When the global budget is spent,
    sign-ins from everywhere are spaced out; the account is never locked.
    """

    def __init__(
        self,
        per_address: int = 5,
        global_budget: int = 100,
        window_s: float = 900,
        spacing_s: float = 3,
    ) -> None:
        self.per_address = per_address
        self.global_budget = global_budget
        self.window_s = window_s
        self.spacing_s = spacing_s
        self._by_address: dict[str, deque[float]] = {}
        self._global: deque[float] = deque()
        self._last_admitted = float("-inf")
        self._lock = threading.Lock()

    def admit(self, address: str) -> float:
        """Counts an attempt as a failure and returns its stamp, or raises 429."""
        with self._lock:
            now = time.monotonic()
            self._expire(now)
            failures = self._by_address.setdefault(address, deque())
            if len(failures) >= self.per_address:
                _throttled(failures[0] + self.window_s - now)
            if len(self._global) >= self.global_budget and now - self._last_admitted < self.spacing_s:
                _throttled(self._last_admitted + self.spacing_s - now)
            self._last_admitted = now
            failures.append(now)
            self._global.append(now)
            return now

    def forgive(self, address: str, stamp: float) -> None:
        with self._lock:
            for failures in (self._by_address.get(address), self._global):
                if failures and stamp in failures:
                    failures.remove(stamp)

    def _expire(self, now: float) -> None:
        # ponytail: scans every address on each sign-in; fine for a lab, a timer-wheel if
        # an attacker with very many addresses ever makes it slow.
        cutoff = now - self.window_s
        for failures in [self._global, *self._by_address.values()]:
            while failures and failures[0] < cutoff:
                failures.popleft()
        for address in [a for a, failures in self._by_address.items() if not failures]:
            del self._by_address[address]


def _throttled(wait_s: float) -> NoReturn:
    raise HTTPException(429, "Too many sign-in attempts", headers={"Retry-After": str(max(1, ceil(wait_s)))})


def client_address(request: Request, trusted_proxy: str) -> str:
    """The peer, or the right-most X-Forwarded-For entry when the peer is the trusted proxy."""
    peer = request.client.host if request.client else ""
    if trusted_proxy and peer == trusted_proxy:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[-1].strip()
        try:
            return str(ip_address(forwarded))
        except ValueError:
            pass
    return peer


def create_session(db: Database, user_id: int) -> str:
    token = new_token()
    now = utcnow()
    with db.write() as s:
        s.add(
            models.Session(
                token_hash=token_hash(token),
                user_id=user_id,
                created_at=now,
                last_seen_at=now,
                expires_at=now + SESSION_MAX,
            )
        )
    return token


def check_origin(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """Browser routes: every request that is not a GET must come from the public origin."""
    if request.method not in ("GET", "HEAD") and request.headers.get("origin") != settings.public_origin:
        raise HTTPException(403, "Cross-origin request refused")


def current_user(
    request: Request,
    settings: Settings = Depends(get_settings),
    db: Database = Depends(get_db),
) -> models.User:
    token = request.cookies.get(settings.session_cookie)
    if not token:
        raise HTTPException(401, "Not signed in")
    hashed = token_hash(token)
    now = utcnow()
    with db.read() as s:
        row = s.execute(
            select(models.Session, models.User)
            .join(models.User, models.User.id == models.Session.user_id)
            .where(models.Session.token_hash == hashed)
        ).first()
    if row is None:
        raise HTTPException(401, "Not signed in")
    session, user = row
    if now >= session.expires_at or now - session.last_seen_at >= SESSION_IDLE:
        raise HTTPException(401, "Session expired")
    if now - session.last_seen_at >= SESSION_TOUCH:
        with db.write() as s:
            s.execute(update(models.Session).where(models.Session.id == session.id).values(last_seen_at=now))
    return user


def current_worker(request: Request, db: Database = Depends(get_db)) -> models.Worker:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(401, "Worker token required", headers={"WWW-Authenticate": "Bearer"})
    with db.read() as s:
        worker = s.scalar(
            select(models.Worker).where(
                models.Worker.token_hash == token_hash(token.strip()),
                models.Worker.revoked_at.is_(None),
            )
        )
    if worker is None:
        raise HTTPException(401, "Unknown or revoked worker token", headers={"WWW-Authenticate": "Bearer"})
    return worker
