"""The administration commands."""

import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.cli import main
from app.db import Database
from app.main import create_app
from app.models import User
from app.settings import Settings
from conftest import ORIGIN


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("COURTSHOT_DATA_DIR", str(tmp_path / "data"))
    main(["migrate"])
    return tmp_path / "data"


def run(capsys, monkeypatch, *argv: str, stdin: str = "") -> str:
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    main(list(argv))
    return capsys.readouterr().out


def test_users(data_dir, capsys, monkeypatch):
    run(capsys, monkeypatch, "create-user", "kerem", "--password-stdin", stdin="a long enough secret\n")
    with pytest.raises(SystemExit, match="exists"):
        run(capsys, monkeypatch, "create-user", "kerem", "--password-stdin", stdin="a long enough secret\n")
    with pytest.raises(SystemExit, match="at least 12"):
        run(capsys, monkeypatch, "create-user", "other", "--password-stdin", stdin="short\n")
    db = Database(data_dir / "app.db")
    with db.read() as s:
        first = s.scalar(select(User.password_hash))
    run(capsys, monkeypatch, "set-password", "kerem", "--password-stdin", stdin="another long secret\n")
    with db.read() as s:
        assert s.scalar(select(User.password_hash)) != first
    with pytest.raises(SystemExit, match="No user"):
        run(capsys, monkeypatch, "set-password", "nobody", "--password-stdin", stdin="another long secret\n")
    db.close()


def test_password_typed_twice(data_dir, capsys, monkeypatch):
    answers = iter(["a long enough secret", "a different secret!"])
    monkeypatch.setattr("getpass.getpass", lambda prompt: next(answers))
    with pytest.raises(SystemExit, match="differ"):
        main(["create-user", "kerem"])


def test_workers(data_dir, capsys, monkeypatch):
    token = run(capsys, monkeypatch, "create-worker", "lab-mac-1").strip().splitlines()[-1]
    app = create_app(Settings(data_dir=data_dir))
    client = TestClient(app, base_url=ORIGIN, headers={"Authorization": f"Bearer {token}"})
    assert client.post("/api/worker/claim", json={"info": {}}).status_code == 204
    assert "lab-mac-1\tactive" in run(capsys, monkeypatch, "list-workers")

    with pytest.raises(SystemExit, match="revoke it first"):
        run(capsys, monkeypatch, "create-worker", "lab-mac-1")
    run(capsys, monkeypatch, "revoke-worker", "lab-mac-1")
    assert client.post("/api/worker/claim", json={"info": {}}).status_code == 401
    assert "revoked" in run(capsys, monkeypatch, "list-workers")

    new_token = run(capsys, monkeypatch, "create-worker", "lab-mac-1").strip().splitlines()[-1]
    assert new_token != token
    client.headers["Authorization"] = f"Bearer {new_token}"
    assert client.post("/api/worker/claim", json={"info": {}}).status_code == 204
    app.state.db.close()


def test_commands_need_a_migrated_database(tmp_path, monkeypatch):
    monkeypatch.setenv("COURTSHOT_DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit, match="not migrated"):
        main(["list-workers"])
