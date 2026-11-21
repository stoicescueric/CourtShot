"""Chunked upload of videos and results: resume, retries, offsets, caps, races (design 8.3)."""

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

import app.storage as storage
from app.deps import read_body
from app.models import Job
from conftest import CALIBRATION, CHUNK, VIDEO, create_match, queued_match, upload


def patch(client, match_id: int, offset: int, data: bytes):
    return client.patch(f"/api/matches/{match_id}/upload", params={"offset": offset}, content=data)


def test_resume_after_a_break(browser, settings):
    match = create_match(browser)
    mid = match["id"]
    assert match["upload"] == {"offset": 0, "complete": False}
    assert patch(browser, mid, 0, VIDEO[:CHUNK]).json() == {"offset": CHUNK, "complete": False}
    state = browser.get(f"/api/matches/{mid}/upload").json()
    assert state == {"offset": CHUNK, "size_bytes": len(VIDEO), "complete": False, "fingerprint": match["fingerprint"]}
    assert browser.get(f"/api/matches/{mid}").json()["status"] == "uploading"
    last = upload(browser, mid, start=state["offset"])
    assert last.json() == {"offset": len(VIDEO), "complete": True}
    folder = settings.data_dir / "matches" / str(mid)
    assert (folder / "source.mp4").read_bytes() == VIDEO
    assert not (folder / "source.part").exists()
    assert patch(browser, mid, 0, b"late").json() == {"offset": len(VIDEO), "complete": True}


def test_a_retried_chunk_rewrites_the_same_bytes(browser, settings):
    mid = create_match(browser)["id"]
    patch(browser, mid, 0, VIDEO[:CHUNK])
    patch(browser, mid, CHUNK, VIDEO[CHUNK : 2 * CHUNK])
    assert patch(browser, mid, 0, VIDEO[:CHUNK]).json()["offset"] == 2 * CHUNK
    upload(browser, mid, start=2 * CHUNK)
    assert (settings.data_dir / "matches" / str(mid) / "source.mp4").read_bytes() == VIDEO


def test_an_offset_beyond_the_size_is_refused(browser):
    mid = create_match(browser)["id"]
    patch(browser, mid, 0, VIDEO[:CHUNK])
    response = patch(browser, mid, CHUNK + 1, VIDEO[CHUNK + 1 : 2 * CHUNK])
    assert response.status_code == 409
    assert response.json()["offset"] == CHUNK and isinstance(response.json()["detail"], str)


def test_chunk_cap_with_content_length(app, browser):
    app.state.settings = app.state.settings.model_copy(update={"max_chunk_bytes": 1000, "chunk_bytes": 1000})
    mid = create_match(browser)["id"]
    assert patch(browser, mid, 0, VIDEO[:1001]).status_code == 413
    assert patch(browser, mid, 0, VIDEO[:1000]).status_code == 200


def test_chunk_cap_without_content_length(app, browser):
    app.state.settings = app.state.settings.model_copy(update={"max_chunk_bytes": 1000, "chunk_bytes": 1000})
    mid = create_match(browser)["id"]

    def stream() -> Iterator[bytes]:  # chunked transfer encoding: no Content-Length
        for i in range(0, 50_000, 100):
            yield VIDEO[i : i + 100]

    response = browser.patch(f"/api/matches/{mid}/upload", params={"offset": 0}, content=stream())
    assert response.status_code == 413
    assert browser.get(f"/api/matches/{mid}/upload").json()["offset"] == 0


def test_the_body_is_refused_as_soon_as_it_passes_the_cap():
    consumed = []

    async def stream() -> AsyncIterator[bytes]:
        for i in range(100):
            consumed.append(i)
            yield b"x" * 100

    request = SimpleNamespace(headers={}, stream=stream)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(read_body(request, cap=1000))
    assert refused.value.status_code == 413 and len(consumed) == 11


def test_a_chunk_past_the_declared_size_is_refused(browser):
    mid = create_match(browser, data=VIDEO[:1000])["id"]
    assert patch(browser, mid, 0, VIDEO[:1001]).status_code == 413


def test_new_match_limits(app, browser):
    assert create_match(browser)  # within every limit
    body = {
        "title": "t",
        "filename": "clip.mp4",
        "size_bytes": 100,
        "fingerprint": "0" * 64,
        "duration_s": 60,
        "frame_width": 1280,
        "frame_height": 720,
    }
    assert browser.post("/api/matches", json=body | {"filename": "clip.avi"}).status_code == 422
    assert browser.post("/api/matches", json=body | {"filename": "noextension"}).status_code == 422
    assert browser.post("/api/matches", json=body | {"filename": "CLIP.MOV"}).json()["status"] == "uploading"
    assert browser.post("/api/matches", json=body | {"duration_s": 9.9}).status_code == 422
    assert browser.post("/api/matches", json=body | {"duration_s": 900.1}).status_code == 422
    assert browser.post("/api/matches", json=body | {"fingerprint": "XYZ"}).status_code == 422
    assert browser.post("/api/matches", json=body | {"title": "   "}).status_code == 422
    assert browser.post("/api/matches", json=body | {"title": "x" * 200}).status_code == 201
    assert browser.post("/api/matches", json=body | {"title": "x" * 201}).status_code == 422
    too_big = body | {"size_bytes": app.state.settings.max_upload_bytes + 1}
    assert browser.post("/api/matches", json=too_big).status_code == 413


def test_free_space_reserve(app, browser):
    mid = create_match(browser)["id"]
    app.state.storage.reserve = 10**18
    assert (
        browser.post(
            "/api/matches",
            json={
                "title": "t",
                "filename": "clip.mp4",
                "size_bytes": 100,
                "fingerprint": "0" * 64,
                "duration_s": 60,
                "frame_width": 1280,
                "frame_height": 720,
            },
        ).status_code
        == 507
    )
    assert patch(browser, mid, 0, VIDEO[:CHUNK]).status_code == 507


def test_upload_of_an_unknown_match(browser):
    assert patch(browser, 999, 0, b"x").status_code == 404
    assert browser.get("/api/matches/999/upload").status_code == 404


def test_two_requests_at_once(browser, db, settings):
    """The same chunks sent twice in parallel, as a retrying browser would; then the final chunk twice."""
    match = create_match(browser)
    mid = match["id"]
    browser.put(f"/api/matches/{mid}/calibration", json=CALIBRATION)
    results: list[int] = []

    def send(offset: int) -> None:
        barrier.wait()
        response = patch(browser, mid, offset, VIDEO[offset : offset + CHUNK])
        results.append(response.status_code)

    for offset in range(0, len(VIDEO), CHUNK):
        barrier = threading.Barrier(2)
        threads = [threading.Thread(target=send, args=(offset,)) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert set(results) == {200}
    assert (settings.data_dir / "matches" / str(mid) / "source.mp4").read_bytes() == VIDEO
    with db.read() as s:
        assert s.scalar(select(func.count(Job.id)).where(Job.match_id == mid)) == 1


def test_two_chunks_at_different_offsets_at_once(browser, settings):
    mid = create_match(browser)["id"]
    statuses: dict[int, int] = {}
    barrier = threading.Barrier(2)

    def send(offset: int) -> None:
        barrier.wait()
        statuses[offset] = patch(browser, mid, offset, VIDEO[offset : offset + CHUNK]).status_code

    threads = [threading.Thread(target=send, args=(offset,)) for offset in (0, CHUNK)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert statuses[0] == 200 and statuses[CHUNK] in (200, 409)
    offset = browser.get(f"/api/matches/{mid}/upload").json()["offset"]
    upload(browser, mid, start=offset)
    assert (settings.data_dir / "matches" / str(mid) / "source.mp4").read_bytes() == VIDEO


def test_result_upload_resume_and_offsets(browser, worker):
    job_id = queued_match(browser)["job"]["id"]
    worker.claimed()
    attempt = {"X-Job-Attempt": "1"}
    assert worker.http.get(f"/api/worker/jobs/{job_id}/files/report.json", headers=attempt).json() == {"offset": 0}
    assert worker.put_file(job_id, 1, "report.json", b'{"a":').json() == {"offset": 5}
    assert worker.http.get(f"/api/worker/jobs/{job_id}/files/report.json", headers=attempt).json() == {"offset": 5}
    assert worker.put_file(job_id, 1, "report.json", b'{"a":').json() == {"offset": 5}  # retried
    beyond = worker.put_file(job_id, 1, "report.json", b"1}", offset=6)
    assert beyond.status_code == 409 and beyond.json()["offset"] == 5
    assert worker.put_file(job_id, 1, "report.json", b"1}", offset=5).json() == {"offset": 7}


def test_result_names_and_caps(browser, worker, monkeypatch):
    job_id = queued_match(browser)["job"]["id"]
    worker.claimed()
    assert worker.put_file(job_id, 1, "evil.html", b"<script>").status_code == 422
    attempt = {"X-Job-Attempt": "1"}
    assert worker.http.get(f"/api/worker/jobs/{job_id}/files/app.db", headers=attempt).status_code == 422
    monkeypatch.setitem(storage.RESULT_FILES, "summary.json", storage.ResultFile("application/json", 10, True))
    assert worker.put_file(job_id, 1, "summary.json", b"x" * 11).status_code == 413
    monkeypatch.setattr(storage, "MAX_RUN_BYTES", 20)
    assert worker.put_file(job_id, 1, "report.json", b"x" * 15).status_code == 200
    assert worker.put_file(job_id, 1, "positions.csv", b"x" * 6).status_code == 413


def test_result_chunk_cap_without_content_length(app, browser, worker):
    job_id = queued_match(browser)["job"]["id"]
    worker.claimed()
    app.state.settings = app.state.settings.model_copy(update={"max_chunk_bytes": 1000, "chunk_bytes": 1000})
    response = worker.http.patch(
        f"/api/worker/jobs/{job_id}/files/tracked.mp4",
        params={"offset": 0},
        content=iter([b"x" * 600, b"x" * 600]),
        headers={"X-Job-Attempt": "1"},
    )
    assert response.status_code == 413


@pytest.mark.parametrize("ext", ["mp4", "mov", "m4v", "webm"])
def test_every_video_extension_is_accepted(browser, settings, ext):
    mid = create_match(browser, filename=f"clip.{ext.upper()}")["id"]
    upload(browser, mid)
    assert (settings.data_dir / "matches" / str(mid) / f"source.{ext}").is_file()
