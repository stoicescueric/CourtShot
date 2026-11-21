"""Administration on the server machine: `uv run python -m app.cli <command>`."""

import argparse
import getpass
import sys

from alembic import command
from alembic.config import Config
from sqlalchemy import delete, select

from .auth import hash_password, new_token, token_hash
from .db import Database, iso, utcnow
from .models import Session, User, Worker
from .settings import SERVER_DIR, Settings

MIN_PASSWORD_LENGTH = 12


def alembic_config(settings: Settings) -> Config:
    config = Config(str(SERVER_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(SERVER_DIR / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{settings.db_path}")
    return config


def migrate(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    command.upgrade(alembic_config(settings), "head")
    print(f"Database at {settings.db_path} is up to date.")


def read_password(from_stdin: bool) -> str:
    if from_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass("Password: ")
        if getpass.getpass("Repeat it: ") != password:
            sys.exit("The passwords differ.")
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(f"The password must have at least {MIN_PASSWORD_LENGTH} characters.")
    return password


def create_user(db: Database, username: str, from_stdin: bool) -> None:
    password_hash = hash_password(read_password(from_stdin))
    with db.write() as s:
        if s.scalar(select(User).where(User.username == username)) is not None:
            sys.exit(f"User {username!r} exists; use set-password.")
        s.add(User(username=username, password_hash=password_hash))
    print(f"Created user {username!r}.")


def set_password(db: Database, username: str, from_stdin: bool) -> None:
    password_hash = hash_password(read_password(from_stdin))
    with db.write() as s:
        user = s.scalar(select(User).where(User.username == username))
        if user is None:
            sys.exit(f"No user {username!r}.")
        user.password_hash = password_hash
        s.execute(delete(Session).where(Session.user_id == user.id))
    print(f"Password of {username!r} changed; its sessions are signed out.")


def create_worker(db: Database, name: str) -> None:
    """Prints a new token once. A revoked worker's name can be given a new token."""
    token = new_token()
    with db.write() as s:
        worker = s.scalar(select(Worker).where(Worker.name == name))
        if worker is not None and worker.revoked_at is None:
            sys.exit(f"Worker {name!r} exists; revoke it first to issue a new token.")
        if worker is None:
            s.add(Worker(name=name, token_hash=token_hash(token), created_at=utcnow()))
        else:
            worker.token_hash, worker.revoked_at = token_hash(token), None
    print(f"Token for worker {name!r} (shown once, store it on the Mac):\n{token}")


def revoke_worker(db: Database, name: str) -> None:
    with db.write() as s:
        worker = s.scalar(select(Worker).where(Worker.name == name))
        if worker is None:
            sys.exit(f"No worker {name!r}.")
        worker.revoked_at = worker.revoked_at or utcnow()
    print(f"Revoked worker {name!r}. A job it was running returns to the queue when its lease runs out.")


def list_workers(db: Database) -> None:
    with db.read() as s:
        workers = s.scalars(select(Worker).order_by(Worker.name)).all()
    if not workers:
        print("No workers.")
    for w in workers:
        state = f"revoked {iso(w.revoked_at)}" if w.revoked_at else "active"
        print(f"{w.name}\t{state}\tlast seen {iso(w.last_seen_at) or 'never'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate", help="create or upgrade the database")
    for name in ("create-user", "set-password"):
        sub = commands.add_parser(name)
        sub.add_argument("username")
        sub.add_argument("--password-stdin", action="store_true", help="read the password from one line of stdin")
    for name in ("create-worker", "revoke-worker"):
        commands.add_parser(name).add_argument("name")
    commands.add_parser("list-workers")
    args = parser.parse_args(argv)

    settings = Settings()  # type: ignore[call-arg]
    if args.command == "migrate":
        migrate(settings)
        return
    db = Database(settings.db_path)
    if not db.is_migrated():
        sys.exit("The database is not migrated: run `python -m app.cli migrate` first.")
    match args.command:
        case "create-user":
            create_user(db, args.username, args.password_stdin)
        case "set-password":
            set_password(db, args.username, args.password_stdin)
        case "create-worker":
            create_worker(db, args.name)
        case "revoke-worker":
            revoke_worker(db, args.name)
        case "list-workers":
            list_workers(db)


if __name__ == "__main__":
    main()
