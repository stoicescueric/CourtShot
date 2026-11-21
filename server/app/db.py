"""SQLite access: a write engine (BEGIN IMMEDIATE) and a read engine on one file (design 8.2)."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.orm import Session, sessionmaker

PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA busy_timeout=5000",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA cache_size=-65536",  # 64 MiB
    "PRAGMA mmap_size=268435456",  # 256 MiB
    "PRAGMA journal_size_limit=67108864",  # 64 MiB WAL after a checkpoint
)


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def iso(value: datetime | None) -> str | None:
    return None if value is None else value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _engine(path: Path, begin: str) -> Engine:
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    # SQLAlchemy's recipe for pysqlite: take transaction control away from the driver
    # and emit our own BEGIN, so the write engine can start with BEGIN IMMEDIATE.
    @event.listens_for(engine, "connect")
    def _connect(dbapi_connection: Any, _record: Any) -> None:
        dbapi_connection.isolation_level = None
        cursor = dbapi_connection.cursor()
        for pragma in PRAGMAS:
            cursor.execute(pragma)
        cursor.close()

    @event.listens_for(engine, "begin")
    def _begin(connection: Any) -> None:
        connection.exec_driver_sql(begin)

    return engine


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.write_engine = _engine(path, "BEGIN IMMEDIATE")
        self.read_engine = _engine(path, "BEGIN")
        self._write = sessionmaker(self.write_engine, expire_on_commit=False)
        self._read = sessionmaker(self.read_engine, expire_on_commit=False)

    @contextmanager
    def write(self) -> Iterator[Session]:
        """One write transaction: commits on success, rolls back on any exception."""
        with self._write.begin() as session:
            yield session

    @contextmanager
    def read(self) -> Iterator[Session]:
        with self._read() as session:
            yield session

    def is_migrated(self) -> bool:
        with self.read_engine.connect() as connection:
            return inspect(connection).has_table("alembic_version")

    def close(self) -> None:
        with self.write_engine.begin() as connection:
            connection.exec_driver_sql("PRAGMA optimize")
        self.write_engine.dispose()
        self.read_engine.dispose()
