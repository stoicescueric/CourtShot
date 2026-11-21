"""Schema and SQLite settings (design 8.2), and the anti-goals that can be checked by a test."""

import shutil
import sqlite3
import subprocess
import tomllib
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, text

from app.db import Database
from app.models import Base

SERVER = Path(__file__).resolve().parent.parent
REPO = SERVER.parent
STARTING_COMMIT = "c2b1767"
REFERENCE_PATHS = ("src", "tools", "tests", "examples", "run_match.sh", "pyproject.toml", "uv.lock")
# Every package the server may install. A new one fails the lock test below until it is added here on
# purpose, after checking it is no CV or ML library (anti-goal: the server never decodes or analyses video).
ALLOWED_PACKAGES = {
    "alembic", "annotated-doc", "annotated-types", "anyio", "argon2-cffi", "argon2-cffi-bindings", "certifi",
    "cffi", "click", "colorama", "courtshot-server", "fastapi", "h11", "httpcore", "httptools", "httpx", "idna",
    "iniconfig", "mako", "markupsafe", "opentelemetry-api", "packaging", "pluggy", "pycparser", "pydantic",
    "pydantic-core", "pydantic-settings", "pygments", "pytest", "python-dotenv", "pyyaml", "sqlalchemy",
    "starlette", "typing-extensions", "typing-inspection", "uvicorn", "uvloop", "watchfiles", "websockets",
}  # fmt: skip


def test_migrations_match_the_models(migrated_db):
    engine = create_engine(f"sqlite:///{migrated_db}")
    with engine.connect() as connection:
        diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    engine.dispose()
    assert diff == []


def test_partial_index_keeps_its_condition(migrated_db):
    with sqlite3.connect(migrated_db) as connection:
        (sql,) = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'uq_jobs_active_match'"
        ).fetchone()
    assert "UNIQUE" in sql and "WHERE status IN ('queued', 'running')" in sql


def test_ids_are_never_reused(migrated_db):
    with sqlite3.connect(migrated_db) as connection:
        for table in ("matches", "jobs"):
            (sql,) = connection.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()
            assert "AUTOINCREMENT" in sql


def test_pragmas_on_both_engines(tmp_path, migrated_db):
    shutil.copy(migrated_db, tmp_path / "app.db")
    db = Database(tmp_path / "app.db")
    expected = {"journal_mode": "wal", "synchronous": 1, "foreign_keys": 1, "busy_timeout": 5000, "temp_store": 2}
    for engine in (db.write_engine, db.read_engine):
        with engine.connect() as connection:
            for pragma, value in expected.items():
                assert connection.execute(text(f"PRAGMA {pragma}")).scalar() == value
    db.close()


def test_write_transactions_begin_immediate_and_reads_do_not_block(tmp_path, migrated_db):
    shutil.copy(migrated_db, tmp_path / "app.db")
    db = Database(tmp_path / "app.db")
    other = sqlite3.connect(tmp_path / "app.db", timeout=0, isolation_level=None)
    with db.write() as s:
        s.execute(text("SELECT 1"))  # no write yet, but the write lock is already held
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            other.execute("BEGIN IMMEDIATE")
        with db.read() as r:
            assert r.execute(text("SELECT count(*) FROM matches")).scalar() == 0
    other.execute("BEGIN IMMEDIATE")
    other.execute("ROLLBACK")
    other.close()
    db.close()


def test_the_lock_file_has_no_cv_or_ml_package():
    lock = tomllib.loads((SERVER / "uv.lock").read_text())
    names = {package["name"].lower() for package in lock["package"]}
    assert "fastapi" in names
    assert names - ALLOWED_PACKAGES == set()


def test_reference_paths_are_unchanged():
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    pathspecs = [f":(top){path}" for path in REFERENCE_PATHS]
    changed = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--name-only", STARTING_COMMIT, "--", *pathspecs],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    added = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--others", "--exclude-standard", "--", *pathspecs],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert changed == "" and added == ""
