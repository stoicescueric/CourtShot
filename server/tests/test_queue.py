"""The queue: claim, lease, stale attempts, complete, fail, cancel (design 9 and 13)."""

import threading
from datetime import timedelta

import pytest
from sqlalchemy import select, update

import app.jobs as jobs
from app.db import utcnow
from app.models import Job, Match, Worker
from app.views import match_status
from conftest import CALIBRATION, REPORT, WorkerClient, add_worker, create_match, queued_match, upload


def job_row(db, job_id: int) -> Job:
    with db.read() as s:
        return s.get_one(Job, job_id)


def set_job(db, job_id: int, **values) -> None:
    with db.write() as s:
        s.execute(update(Job).where(Job.id == job_id).values(**values))


def test_life_of_a_match(browser, worker, db):
    match = queued_match(browser)
    assert match["status"] == "queued" and match["job"]["status"] == "queued"

    claim = worker.claimed()
    assert claim == {
        "job_id": match["job"]["id"],
        "attempt": 1,
        "match_id": match["id"],
        "video": {"size_bytes": match["size_bytes"], "ext": "mp4"},
        "calibration": CALIBRATION,
        "limits": {"min_video_seconds": 10, "max_video_seconds": 900, "calibration_max_error_m": 0.25},
        "heartbeat_seconds": 15,
        "chunk_bytes": 8 * 1024 * 1024,
    }
    assert worker.heartbeat(claim["job_id"], 1, "ball3d", 0.7).json() == {}
    detail = browser.get(f"/api/matches/{match['id']}").json()
    assert detail["status"] == "processing"
    expected = {"stage": "ball3d", "progress": 0.7, "worker": "lab-mac-1", "log_tail": None}
    assert detail["job"].items() >= expected.items()

    worker.deliver(claim["job_id"], 1, extra=("scene.json",))
    assert worker.complete(claim["job_id"], 1).json() == {}
    detail = browser.get(f"/api/matches/{match['id']}").json()
    assert detail["status"] == "ready"
    assert detail["summary"] == {"duration_s": 89.96, "rallies": 6, "shots": 41, "time_in_play_s": 50.2}
    assert detail["results"]["job_id"] == claim["job_id"]
    assert detail["results"]["files"] == [
        "report.json",
        "tracked.mp4",
        "plain.mp4",
        "summary.json",
        "positions.csv",
        "scene.json",
    ]
    listed = browser.get("/api/matches").json()
    assert listed["workers_online"] is True
    assert listed["matches"][0]["status"] == "ready" and "calibration" not in listed["matches"][0]


def test_calibration_first_then_upload_queues_the_first_job(browser, db):
    match = create_match(browser)
    saved = browser.put(f"/api/matches/{match['id']}/calibration", json=CALIBRATION).json()
    assert saved["status"] == "uploading" and saved["job"] is None and saved["has_calibration"]
    upload(browser, match["id"])
    detail = browser.get(f"/api/matches/{match['id']}").json()
    assert detail["status"] == "queued"
    assert job_row(db, detail["job"]["id"]).calibration == CALIBRATION


def test_upload_without_calibration_needs_calibration(browser):
    match = create_match(browser)
    upload(browser, match["id"])
    assert browser.get(f"/api/matches/{match['id']}").json()["status"] == "needs_calibration"


def test_nothing_to_claim(worker):
    assert worker.claim().status_code == 204


def test_two_claims_at_once_get_different_jobs(app, browser, db):
    for _ in range(6):
        queued_match(browser)
    for i in range(4):
        add_worker(db, f"mac-{i}")
    with db.read() as s:
        worker_ids = list(s.scalars(select(Worker.id)))
    claimed: list[int] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(worker_ids))

    def claim_all(worker_id: int) -> None:
        try:
            barrier.wait()
            while (result := jobs.claim(db, worker_id, {})) is not None:
                claimed.append(result[0].id)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=claim_all, args=(worker_id,)) for worker_id in worker_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(claimed) == 6 and len(set(claimed)) == 6


def test_oldest_job_is_claimed_first(browser, worker):
    first = queued_match(browser)
    queued_match(browser)
    assert worker.claimed()["job_id"] == first["job"]["id"]


def test_lease_expiry_requeues_then_fails_at_the_attempt_limit(app, browser, worker, db):
    match = queued_match(browser)
    job_id = match["job"]["id"]
    settings = app.state.settings
    for attempt in (1, 2, 3):
        assert worker.claimed()["attempt"] == attempt
        set_job(db, job_id, heartbeat_at=utcnow() - timedelta(seconds=settings.lease_seconds + 1))
        assert jobs.expire_leases(db, settings) == 1
        row = job_row(db, job_id)
        if attempt < 3:
            assert (row.status, row.worker_id, row.stage) == ("queued", None, None)
            assert worker.heartbeat(job_id, attempt).status_code == 409  # the lost lease
    assert (row.status, row.error) == ("failed", jobs.LEASE_LOST)
    assert row.finished_at is not None


def test_a_recent_heartbeat_keeps_the_lease(app, browser, worker, db):
    job_id = queued_match(browser)["job"]["id"]
    worker.claimed()
    set_job(db, job_id, heartbeat_at=utcnow() - timedelta(seconds=app.state.settings.lease_seconds - 5))
    assert jobs.expire_leases(db, app.state.settings) == 0


def race_heartbeat_with_sweep(db, settings, job_id: int, attempt: int, worker_id: int) -> tuple[int, int]:
    """Runs one heartbeat and one lease sweep at the same instant; returns (heartbeat status, jobs expired)."""
    barrier = threading.Barrier(2)
    outcome: dict[str, int] = {}

    def beat() -> None:
        barrier.wait()
        try:
            jobs.heartbeat(db, job_id, attempt, worker_id, "tracking", 0.5)
            outcome["heartbeat"] = 200
        except jobs.HTTPException as exc:
            outcome["heartbeat"] = exc.status_code

    def sweep() -> None:
        barrier.wait()
        outcome["expired"] = jobs.expire_leases(db, settings)

    threads = [threading.Thread(target=beat), threading.Thread(target=sweep)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return outcome["heartbeat"], outcome["expired"]


def test_a_heartbeat_racing_the_lease_sweep_is_not_lost(app, browser, worker, db):
    settings = app.state.settings.model_copy(update={"max_attempts": 1000})
    job_id = queued_match(browser)["job"]["id"]
    attempt = worker.claimed()["attempt"]
    worker_id = job_row(db, job_id).worker_id
    for _ in range(20):
        set_job(db, job_id, heartbeat_at=utcnow() - timedelta(seconds=settings.lease_seconds + 1))
        heartbeat, expired = race_heartbeat_with_sweep(db, settings, job_id, attempt, worker_id)
        status = job_row(db, job_id).status
        # Either the heartbeat committed first and the sweep saw a fresh lease, or the sweep
        # committed first and the heartbeat was refused: never an accepted heartbeat on a lost lease.
        if heartbeat == 200:
            assert (expired, status) == (0, "running")
        else:
            assert (heartbeat, expired, status) == (409, 1, "queued")
            attempt = worker.claimed()["attempt"]


def test_stale_attempts_are_refused(app, browser, worker, db):
    job_id = queued_match(browser)["job"]["id"]
    worker.claimed()
    assert worker.heartbeat(job_id, 2).status_code == 409
    assert worker.put_file(job_id, 2, "report.json", b"{}").status_code == 409
    assert worker.complete(job_id, 2).status_code == 409
    assert worker.fail(job_id, 2, "boom", True).status_code == 409
    assert worker.heartbeat(job_id + 100, 1).status_code == 404
    other = WorkerClient(app, add_worker(db, "lab-mac-2"))
    assert other.heartbeat(job_id, 1).status_code == 409  # not its job
    assert worker.heartbeat(job_id, 1).status_code == 200


def test_missing_attempt_header_is_invalid(browser, worker):
    job_id = queued_match(browser)["job"]["id"]
    worker.claimed()
    response = worker.http.post(f"/api/worker/jobs/{job_id}/heartbeat", json={"stage": "tracking", "progress": 0})
    assert response.status_code == 422 and isinstance(response.json()["detail"], str)


def test_heartbeat_body_is_checked(browser, worker):
    job_id = queued_match(browser)["job"]["id"]
    worker.claimed()
    assert worker.heartbeat(job_id, 1, stage="dancing").status_code == 422
    assert worker.heartbeat(job_id, 1, progress=1.5).status_code == 422


def test_repeated_complete_succeeds(browser, worker, db):
    job_id = queued_match(browser)["job"]["id"]
    worker.claimed()
    worker.deliver(job_id, 1)
    assert worker.complete(job_id, 1).status_code == 200
    assert worker.complete(job_id, 1).status_code == 200
    assert worker.complete(job_id, 2).status_code == 409
    assert worker.heartbeat(job_id, 1).status_code == 409


def test_complete_checks_the_files(browser, worker):
    job_id = queued_match(browser)["job"]["id"]
    worker.claimed()
    worker.put_file(job_id, 1, "report.json", b"{}")
    missing = worker.complete(job_id, 1)
    assert missing.status_code == 422
    assert "tracked.mp4" in missing.json()["detail"] and "report.json" not in missing.json()["detail"]
    worker.deliver(job_id, 1, report=[1, 2])
    assert worker.complete(job_id, 1).status_code == 422
    worker.put_file(job_id, 1, "report.json", b"{not json" + b" " * 40)
    assert worker.complete(job_id, 1).status_code == 422
    assert worker.heartbeat(job_id, 1).status_code == 200  # still running, can deliver again


def test_retryable_failure_waits_the_retry_delay(app, browser, worker, db):
    job_id = queued_match(browser)["job"]["id"]
    worker.claimed()
    assert worker.fail(job_id, 1, "segfault", True, "log " * 5000).json() == {}
    row = job_row(db, job_id)
    assert row.status == "queued" and row.worker_id is None
    delay = row.not_before - utcnow()
    assert timedelta(seconds=55) < delay <= timedelta(seconds=app.state.settings.retry_delay_seconds)
    assert len(row.log_tail) == jobs.LOG_TAIL_MAX
    assert worker.claim().status_code == 204
    set_job(db, job_id, not_before=utcnow() - timedelta(seconds=1))
    assert worker.claimed()["attempt"] == 2


def test_retryable_failure_at_the_attempt_limit_fails(browser, worker, db):
    job_id = queued_match(browser)["job"]["id"]
    for attempt in (1, 2, 3):
        worker.claimed()
        worker.fail(job_id, attempt, f"crash {attempt}", True)
        set_job(db, job_id, not_before=None)
    row = job_row(db, job_id)
    assert (row.status, row.error, row.attempt) == ("failed", "crash 3", 3)


def test_final_failure_is_shown_with_its_log(browser, worker):
    match = queued_match(browser)
    worker.claimed()
    worker.fail(match["job"]["id"], 1, "x" * 900, False, "the log")
    detail = browser.get(f"/api/matches/{match['id']}").json()
    assert detail["status"] == "failed"
    assert detail["job"]["error"] == "x" * 500 and detail["job"]["log_tail"] == "the log"


def test_cancel_stops_the_job_at_its_next_heartbeat(browser, worker):
    match = queued_match(browser)
    worker.claimed()
    cancelled = browser.post(f"/api/matches/{match['id']}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    assert worker.heartbeat(match["job"]["id"], 1).status_code == 409
    assert browser.post(f"/api/matches/{match['id']}/cancel").status_code == 409


def test_delete_cancels_and_the_worker_learns_404(browser, worker, settings):
    match = queued_match(browser)
    worker.claimed()
    assert browser.delete(f"/api/matches/{match['id']}").status_code == 204
    assert worker.heartbeat(match["job"]["id"], 1).status_code == 404
    assert browser.get(f"/api/matches/{match['id']}").status_code == 404
    assert not (settings.data_dir / "matches" / str(match["id"])).exists()


def test_run_again_and_redo_calibration_need_no_active_job(browser, worker):
    match = queued_match(browser)
    mid = match["id"]
    assert browser.post(f"/api/matches/{mid}/jobs").status_code == 409
    assert browser.put(f"/api/matches/{mid}/calibration", json=CALIBRATION).status_code == 409
    worker.claimed()
    worker.deliver(match["job"]["id"], 1)
    worker.complete(match["job"]["id"], 1)
    again = browser.post(f"/api/matches/{mid}/jobs")
    assert again.status_code == 201
    assert again.json()["status"] == "queued" and again.json()["results"]["job_id"] == match["job"]["id"]
    browser.post(f"/api/matches/{mid}/cancel")
    points = dict(CALIBRATION["points"], service_near_center=[640.0, 600.0])
    redone = browser.put(f"/api/matches/{mid}/calibration", json=CALIBRATION | {"points": points})
    assert redone.status_code == 200 and redone.json()["status"] == "queued"
    assert worker.claimed()["calibration"]["points"]["service_near_center"] == [640.0, 600.0]


def test_run_again_needs_an_upload_and_a_calibration(browser):
    match = create_match(browser)
    assert browser.post(f"/api/matches/{match['id']}/jobs").status_code == 409
    upload(browser, match["id"])
    assert browser.post(f"/api/matches/{match['id']}/jobs").status_code == 409


def test_a_second_active_job_is_a_conflict_not_a_server_error(browser, monkeypatch):
    match = queued_match(browser)
    monkeypatch.setattr(jobs, "active_job", lambda s, match_id: None)
    response = browser.post(f"/api/matches/{match['id']}/jobs")
    assert response.status_code == 409


@pytest.mark.parametrize(
    "body",
    [
        CALIBRATION | {"points": dict(list(CALIBRATION["points"].items())[:5])},
        CALIBRATION | {"points": CALIBRATION["points"] | {"centre_spot": [1.0, 1.0]}},
        CALIBRATION | {"points": CALIBRATION["points"] | {"net_left": [1281.0, 1.0]}},
        CALIBRATION | {"frame_width": 1920, "frame_height": 1080},
    ],
)
def test_calibration_is_checked_structurally(browser, body):
    match = create_match(browser)
    response = browser.put(f"/api/matches/{match['id']}/calibration", json=body)
    assert response.status_code == 422 and isinstance(response.json()["detail"], str)


def test_workers_list(browser, worker, db):
    idle = {"name": "lab-mac-1", "last_seen_at": None, "online": False, "job": None}
    assert browser.get("/api/workers").json() == {"workers": [idle]}
    match = queued_match(browser)
    worker.claimed()
    worker.heartbeat(match["job"]["id"], 1, "render", 0.9)
    (listed,) = browser.get("/api/workers").json()["workers"]
    assert listed["online"] is True
    assert listed["job"] == {
        "id": match["job"]["id"],
        "match_id": match["id"],
        "match_title": "final_f",
        "stage": "render",
        "progress": 0.9,
    }


@pytest.mark.parametrize(
    ("complete", "calibrated", "job", "status"),
    [
        (False, True, None, "uploading"),
        (True, False, None, "needs_calibration"),
        (True, True, "queued", "queued"),
        (True, True, "running", "processing"),
        (True, True, "succeeded", "ready"),
        (True, True, "failed", "failed"),
        (True, True, "cancelled", "cancelled"),
    ],
)
def test_match_status(complete, calibrated, job, status):
    assert match_status(complete, calibrated, job) == status


def test_summary_from_report():
    assert jobs.summary_from_report(REPORT) == {"duration_s": 89.96, "rallies": 6, "shots": 41, "time_in_play_s": 50.2}
    no_ball = {"duration_s": 30, "match": None, "movement_rallies": [{}, {}, {}]}
    assert jobs.summary_from_report(no_ball) == {"duration_s": 30, "rallies": 3, "shots": None, "time_in_play_s": None}
    odd = {"duration_s": "long", "match": {"rallies": True, "shots": None}}
    assert jobs.summary_from_report(odd) == {"duration_s": None, "rallies": None, "shots": None, "time_in_play_s": None}


def test_match_rows_are_never_reused(browser, db):
    first = create_match(browser)["id"]
    browser.delete(f"/api/matches/{first}")
    assert create_match(browser)["id"] == first + 1
    with db.read() as s:
        assert s.scalar(select(Match.id)) == first + 1
