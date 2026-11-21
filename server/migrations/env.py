"""Alembic environment. Batch mode, because SQLite rebuilds a table to alter it."""

from alembic import context
from sqlalchemy import create_engine, pool

from app.models import Base

config = context.config


def run_migrations() -> None:
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("No database URL: run migrations with `python -m app.cli migrate`")
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    raise RuntimeError("Offline migrations are not supported")
run_migrations()
