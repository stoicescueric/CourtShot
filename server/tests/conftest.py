"""Shared fixtures: a migrated temporary data folder, the app, a signed-in browser and a worker."""

import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from app.auth import hash_password, new_token, token_hash
from app.cli import alembic_config
from app.db import Database, utcnow
from app.main import create_app
from app.models import User, Worker
from app.settings import Settings

ORIGIN = "https://testserver"
USERNAME = "kerem"
PASSWORD = "correct horse battery staple"
VIDEO = bytes(range(256)) * 400  # 102,400 bytes
CHUNK = 40_000
POINTS = {
    "bottom_left": [100.0, 700.0],
    "bottom_right": [1180.0, 700.0],
    "top_left": [400.0, 200.0],
    "top_right": [880.0, 200.0],
    "net_left": [300.0, 380.0],
    "net_right": [980.0, 380.0],
}
CALIBRATION = {"points": POINTS, "frame_width": 1280, "frame_height": 720}
REPORT = {
    "schema_version": 1,
    "pipeline_version": "0.1.0+test",
    "title": "work",
    "duration_s": 89.96,
    "match": {"rallies": 6, "shots": 41, "time_in_play_s": 50.2},
    "movement_rallies": [{"start_s": 1.0, "end_s": 9.0}],
}


def fingerprint(data: bytes) -> str:
    mib = 1024 * 1024
    return hashlib.sha256(data if len(data) < 2 * mib else data[:mib] + data[-mib:]).hexdigest()


@pytest.fixture(scope="session")
def migrated_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An empty database with every migration applied, copied into each test's data folder."""
    data_dir = tmp_path_factory.mktemp("template")
    command.upgrade(alembic_config(Settings(data_dir=data_dir)), "head")
    return data_dir / "app.db"


@pytest.fixture
def settings(tmp_path: Path, migrated_db: Path) -> Settings:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(migrated_db, data_dir / "app.db")
    return Settings(
        data_dir=data_dir,
        public_origin=ORIGIN,
        free_space_reserve_bytes=0,
        web_dist=tmp_path / "dist",
    )


@pytest.fixture
def app(settings: Settings) -> Iterator[FastAPI]:
    app = create_app(settings)
    yield app
    app.state.db.write_engine.dispose()
    app.state.db.read_engine.dispose()


@pytest.fixture
def db(app: FastAPI) -> Database:
    return app.state.db


def add_user(db: Database, username: str = USERNAME, password: str = PASSWORD) -> None:
    with db.write() as s:
        s.add(User(username=username, password_hash=hash_password(password)))


def add_worker(db: Database, name: str) -> str:
    token = new_token()
    with db.write() as s:
        s.add(Worker(name=name, token_hash=token_hash(token), created_at=utcnow()))
    return token


@pytest.fixture
def anonymous(app: FastAPI) -> TestClient:
    return TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN})


@pytest.fixture
def browser(app: FastAPI, db: Database) -> TestClient:
    add_user(db)
    client = TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN})
    assert client.post("/api/auth/login", json={"username": USERNAME, "password": PASSWORD}).status_code == 200
    return client


class WorkerClient:
    """The worker side of server/API.md."""

    def __init__(self, app: FastAPI, token: str) -> None:
        self.http = TestClient(app, base_url=ORIGIN, headers={"Authorization": f"Bearer {token}"})

    def claim(self) -> Response:
        return self.http.post("/api/worker/claim", json={"info": {"hostname": "lab-mac", "device": "mps"}})

    def claimed(self) -> dict[str, Any]:
        response = self.claim()
        assert response.status_code == 200, response.text
        return response.json()

    def heartbeat(self, job_id: int, attempt: int, stage: str = "tracking", progress: float = 0.5) -> Response:
        return self.http.post(
            f"/api/worker/jobs/{job_id}/heartbeat",
            json={"stage": stage, "progress": progress},
            headers={"X-Job-Attempt": str(attempt)},
        )

    def put_file(self, job_id: int, attempt: int, name: str, data: bytes, offset: int = 0) -> Response:
        return self.http.patch(
            f"/api/worker/jobs/{job_id}/files/{name}",
            params={"offset": offset},
            content=data,
            headers={"X-Job-Attempt": str(attempt)},
        )

    def deliver(self, job_id: int, attempt: int, report: Any = REPORT, extra: tuple[str, ...] = ()) -> None:
        files = {
            "report.json": json.dumps(report).encode(),
            "tracked.mp4": b"tracked-video",
            "plain.mp4": b"plain-video",
            "summary.json": b"{}",
            "positions.csv": b"frame,player,x,y\n",
        } | {name: b"{}" for name in extra}
        for name, data in files.items():
            assert self.put_file(job_id, attempt, name, data).status_code == 200

    def complete(self, job_id: int, attempt: int) -> Response:
        return self.http.post(f"/api/worker/jobs/{job_id}/complete", headers={"X-Job-Attempt": str(attempt)})

    def fail(self, job_id: int, attempt: int, error: str, retryable: bool, log_tail: str | None = None) -> Response:
        return self.http.post(
            f"/api/worker/jobs/{job_id}/fail",
            json={"error": error, "retryable": retryable, "log_tail": log_tail},
            headers={"X-Job-Attempt": str(attempt)},
        )


@pytest.fixture
def worker(app: FastAPI, db: Database) -> WorkerClient:
    return WorkerClient(app, add_worker(db, "lab-mac-1"))


def create_match(client: TestClient, data: bytes = VIDEO, **overrides: Any) -> dict[str, Any]:
    body = {
        "title": "final_f",
        "filename": "final_f.mp4",
        "size_bytes": len(data),
        "fingerprint": fingerprint(data),
        "duration_s": 90.0,
        "frame_width": 1280,
        "frame_height": 720,
    } | overrides
    response = client.post("/api/matches", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def upload(client: TestClient, match_id: int, data: bytes = VIDEO, start: int = 0) -> Response:
    response = None
    for offset in range(start, len(data), CHUNK):
        response = client.patch(
            f"/api/matches/{match_id}/upload", params={"offset": offset}, content=data[offset : offset + CHUNK]
        )
        assert response.status_code == 200, response.text
    assert response is not None
    return response


def queued_match(client: TestClient) -> dict[str, Any]:
    """A match with its video uploaded and its calibration saved: its first job is queued."""
    match = create_match(client)
    upload(client, match["id"])
    response = client.put(f"/api/matches/{match['id']}/calibration", json=CALIBRATION)
    assert response.status_code == 200, response.text
    return response.json()
