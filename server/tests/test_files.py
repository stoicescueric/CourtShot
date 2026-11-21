"""File routes: authentication, names, path tricks, ranges, runs that replace each other (design 13)."""

import json

import pytest

from app.sweeps import cleanup
from conftest import REPORT, VIDEO, create_match, queued_match


@pytest.fixture
def ready(browser, worker) -> dict:
    match = queued_match(browser)
    worker.claimed()
    worker.deliver(match["job"]["id"], 1, extra=("scene.json", "shots.csv"))
    assert worker.complete(match["job"]["id"], 1).status_code == 200
    return browser.get(f"/api/matches/{match['id']}").json()


def test_files_need_a_session_or_a_worker_token(ready, anonymous, worker):
    mid, job_id = ready["id"], ready["results"]["job_id"]
    assert anonymous.get(f"/api/matches/{mid}/video").status_code == 401
    assert anonymous.get(f"/api/matches/{mid}/runs/{job_id}/report.json").status_code == 401
    assert anonymous.get(f"/api/worker/jobs/{job_id}/video", headers={"X-Job-Attempt": "1"}).status_code == 401
    # A worker token opens worker routes only, and only for its running attempt.
    assert worker.http.get(f"/api/matches/{mid}/video").status_code == 401
    assert worker.http.get(f"/api/worker/jobs/{job_id}/video", headers={"X-Job-Attempt": "1"}).status_code == 409


@pytest.mark.parametrize(
    "name", ["app.db", "evil.html", "..%2F..%2F..%2Fapp.db", "%2e%2e", "report.json%00", "REPORT.JSON"]
)
def test_unknown_names_and_path_tricks(ready, browser, name):
    response = browser.get(f"/api/matches/{ready['id']}/runs/{ready['results']['job_id']}/{name}")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"


def test_path_tricks_in_the_url_path(ready, browser):
    for path in (
        f"/api/matches/{ready['id']}/runs/{ready['results']['job_id']}/../../../app.db",
        f"/api/matches/{ready['id']}/runs/..%2F..%2Fapp.db/report.json",
        "/api/matches/..%2Fapp.db/video",
    ):
        assert browser.get(path).status_code in (404, 422)
        assert b"SQLite" not in browser.get(path).content


def test_runs_belong_to_their_match_and_must_have_succeeded(ready, browser, worker):
    other = queued_match(browser)
    job_id = ready["results"]["job_id"]
    assert browser.get(f"/api/matches/{other['id']}/runs/{job_id}/report.json").status_code == 404
    assert browser.get(f"/api/matches/{other['id']}/runs/{other['job']['id']}/report.json").status_code == 404
    assert browser.get(f"/api/matches/{ready['id']}/runs/{job_id}/ball_events.csv").status_code == 404  # absent


def test_result_content_types_and_downloads(ready, browser):
    base = f"/api/matches/{ready['id']}/runs/{ready['results']['job_id']}"
    report = browser.get(f"{base}/report.json")
    assert report.headers["content-type"] == "application/json"
    assert report.json() == REPORT
    assert "content-disposition" not in report.headers
    csv = browser.get(f"{base}/positions.csv")
    assert csv.headers["content-type"] == "text/csv; charset=utf-8"
    assert csv.headers["content-disposition"] == 'attachment; filename="positions.csv"'
    assert browser.get(f"{base}/tracked.mp4").headers["content-type"] == "video/mp4"


def test_byte_ranges(ready, browser, worker):
    video = browser.get(f"/api/matches/{ready['id']}/video", headers={"Range": "bytes=10-19"})
    assert video.status_code == 206
    assert video.content == VIDEO[10:20]
    assert video.headers["content-range"] == f"bytes 10-19/{len(VIDEO)}"
    assert video.headers["content-type"] == "video/mp4"
    tail = browser.get(
        f"/api/matches/{ready['id']}/runs/{ready['results']['job_id']}/tracked.mp4", headers={"Range": "bytes=-5"}
    )
    assert tail.status_code == 206 and tail.content == b"video"
    whole = browser.get(f"/api/matches/{ready['id']}/video")
    assert whole.status_code == 200 and whole.content == VIDEO and whole.headers["accept-ranges"] == "bytes"


def test_worker_reads_the_video_with_ranges(browser, worker):
    match = queued_match(browser)
    claim = worker.claimed()
    response = worker.http.get(
        f"/api/worker/jobs/{claim['job_id']}/video", headers={"X-Job-Attempt": "1", "Range": "bytes=0-99"}
    )
    assert response.status_code == 206 and response.content == VIDEO[:100]
    assert browser.get(f"/api/matches/{match['id']}/video").content == VIDEO


def test_video_of_an_unfinished_upload(browser):
    match = create_match(browser)
    assert browser.get(f"/api/matches/{match['id']}/video").status_code == 409


def test_a_second_run_replaces_the_first_without_mixing(app, ready, browser, worker, settings):
    mid, first = ready["id"], ready["results"]["job_id"]
    second = browser.post(f"/api/matches/{mid}/jobs").json()["job"]["id"]
    worker.claimed()
    worker.deliver(second, 1, report=REPORT | {"duration_s": 12.5})  # no scene.json this time
    queued = browser.get(f"/api/matches/{mid}").json()
    assert queued["results"]["job_id"] == first  # until the second run succeeds
    worker.complete(second, 1)

    detail = browser.get(f"/api/matches/{mid}").json()
    assert detail["results"]["job_id"] == second
    assert "scene.json" not in detail["results"]["files"]
    assert detail["summary"]["duration_s"] == 12.5
    assert json.loads(browser.get(f"/api/matches/{mid}/runs/{second}/report.json").content)["duration_s"] == 12.5
    assert browser.get(f"/api/matches/{mid}/runs/{second}/scene.json").status_code == 404
    assert browser.get(f"/api/matches/{mid}/runs/{first}/scene.json").status_code == 200

    cleanup(app.state.db, app.state.storage, settings)
    assert browser.get(f"/api/matches/{mid}/runs/{first}/report.json").status_code == 404
    assert browser.get(f"/api/matches/{mid}/runs/{second}/report.json").status_code == 200
