"""Background sweeps: cleanup of files and rows, and the lease sweep started by the lifespan."""

import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update

import app.sweeps as sweeps
from app.db import utcnow
from app.main import create_app
from app.models import Job, Match, Session
from app.settings import Settings
from conftest import ORIGIN, create_match, queued_match, upload


def test_cleanup(app, browser, worker, db, settings):
    storage = app.state.storage
    matches = settings.data_dir / "matches"

    ready = queued_match(browser)
    worker.claimed()
    worker.deliver(ready["job"]["id"], 1)
    worker.complete(ready["job"]["id"], 1)
    running = queued_match(browser)
    worker.claimed()
    worker.put_file(running["job"]["id"], 1, "report.json", b"{}")
    stale_run = storage.run_dir(ready["id"], ready["job"]["id"], 7)
    stale_run.mkdir()

    orphan = matches / "999"
    orphan.mkdir()
    (matches / "not-a-match").mkdir()

    abandoned = create_match(browser)
    fresh = create_match(browser)
    with db.write() as s:
        s.execute(update(Match).where(Match.id == abandoned["id"]).values(created_at=utcnow() - timedelta(days=8)))
        s.execute(update(Session).values(last_seen_at=utcnow() - timedelta(days=8)))

    sweeps.cleanup(db, storage, settings)

    assert not orphan.exists() and (matches / "not-a-match").exists()
    assert not stale_run.exists()
    assert storage.run_dir(ready["id"], ready["job"]["id"], 1).is_dir()  # current results
    assert storage.run_dir(running["id"], running["job"]["id"], 1).is_dir()  # running attempt
    assert not (matches / str(abandoned["id"])).exists()
    assert (matches / str(fresh["id"])).is_dir()
    with db.read() as s:
        assert s.get(Match, abandoned["id"]) is None and s.get(Match, fresh["id"]) is not None
        assert s.scalar(select(Session)) is None  # idle session removed


def test_an_old_complete_upload_is_kept(browser, db, settings, app):
    match = create_match(browser)
    upload(browser, match["id"])
    with db.write() as s:
        s.execute(update(Match).values(created_at=utcnow() - timedelta(days=30)))
    sweeps.cleanup(db, app.state.storage, settings)
    assert browser.get(f"/api/matches/{match['id']}").status_code == 200


def test_lifespan_runs_the_lease_sweep(app, browser, worker, db, monkeypatch):
    job_id = queued_match(browser)["job"]["id"]
    worker.claimed()
    with db.write() as s:
        s.execute(update(Job).values(heartbeat_at=utcnow() - timedelta(minutes=10)))
    monkeypatch.setattr(sweeps, "LEASE_SWEEP_SECONDS", 0.05)
    with TestClient(app, base_url=ORIGIN):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with db.read() as s:
                if s.get_one(Job, job_id).status == "queued":
                    break
            time.sleep(0.05)
        else:
            pytest.fail("the lease sweep did not run")


def test_the_app_refuses_an_unmigrated_database(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with pytest.raises(RuntimeError, match="not migrated"), TestClient(app):
        pass
