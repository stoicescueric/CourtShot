"""The worker loop against the stub server: claim, download, run, upload, complete, and what goes wrong."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import CALIBRATION, LIMITS
from fakes import result_files
from stub_server import StubServer

from courtshot.worker.runner import Runner, WorkRootBusy

Make = Callable[..., Runner]


def claim_and_run(stub: StubServer, runner: Runner) -> None:
    claim = runner.client.claim({"hostname": "test"})
    assert claim is not None
    runner.run_job(claim)


def wait_for(condition: Callable[[], object], timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_happy_path(stub: StubServer, make_runner: Make, video_bytes: bytes) -> None:
    job = stub.add_job(video_bytes, CALIBRATION, LIMITS)
    runner = make_runner(device="cpu", fit_workers=2)
    claim_and_run(stub, runner)

    assert job.status == "succeeded"
    report = json.loads(job.files["report.json"])
    assert report["video"] == "source.mp4"  # a plain constant-rate clip runs untouched
    assert report["calibration"] == CALIBRATION
    assert report["options"] == {"pipeline_version": runner.version, "device": "cpu", "fit_workers": "2",
                                 "max_error_m": "0.25", "ball_model": None}
    expected = result_files(report)
    del expected["notes.txt"]  # not a result file name: never uploaded
    assert {name: bytes(data) for name, data in job.files.items()} == expected
    stages = [stage for stage, _ in job.heartbeats]
    assert stages[0] == "download"
    progress = [p for _, p in job.heartbeats]
    assert progress == sorted(progress) and 0 <= progress[-1] <= 1
    assert all(r["attempt"] == "1" for r in stub.log if "/jobs/" in r["path"])
    assert not (runner.settings.work_root / "match-7").exists()  # deleted after delivery


def test_resumes_an_interrupted_download(stub: StubServer, make_runner: Make, video_bytes: bytes) -> None:
    job = stub.add_job(video_bytes, CALIBRATION, LIMITS)
    stub.inject("video", "drop", after_bytes=len(video_bytes) // 2)
    claim_and_run(stub, make_runner())

    ranges = [r["range"] for r in stub.requests("/video")]
    assert ranges[0] is None and ranges[1] is not None and ranges[1] != "bytes=0-"
    assert job.status == "succeeded"  # the stitched file is a valid video: prepare probed it


def test_resumes_a_download_left_by_an_earlier_attempt(stub: StubServer, make_runner: Make,
                                                         video_bytes: bytes) -> None:
    job = stub.add_job(video_bytes, CALIBRATION, LIMITS, attempt=2)
    runner = make_runner()
    (runner.settings.work_root / "match-7").mkdir(parents=True)
    (runner.settings.work_root / "match-7" / "source.mp4").write_bytes(video_bytes[:3000])
    claim_and_run(stub, runner)

    assert [r["range"] for r in stub.requests("/video")] == ["bytes=3000-"]
    assert job.status == "succeeded"


def test_resumes_an_interrupted_upload(stub: StubServer, make_runner: Make, video_bytes: bytes) -> None:
    job = stub.add_job(video_bytes, CALIBRATION, LIMITS)
    stub.inject("chunk", "drop_after_write", name="tracked.mp4")  # answer lost: the same chunk again is harmless
    stub.inject("chunk", "drop", name="tracked.mp4", times=2)  # outlasts the client's retries: start over
    stub.inject("chunk", "500", name="plain.mp4")
    stub.inject("chunk", "rewind", name="plain.mp4")  # the server holds less than we sent: 409 with its offset
    claim_and_run(stub, make_runner())

    assert job.status == "succeeded"
    assert bytes(job.files["tracked.mp4"]) == result_files(json.loads(job.files["report.json"]))["tracked.mp4"]
    assert bytes(job.files["plain.mp4"]) == result_files(json.loads(job.files["report.json"]))["plain.mp4"]
    offsets = [r for r in stub.requests("/files/tracked.mp4", "GET")]
    assert len(offsets) == 2  # asked for the offset again and continued from it
    sent = sum(r["bytes"] for r in stub.requests("/files/tracked.mp4", "PATCH"))
    assert sent < 2 * len(job.files["tracked.mp4"])  # not started from scratch


def test_lost_answer_to_complete_is_not_a_lost_job(stub: StubServer, make_runner: Make, video_bytes: bytes) -> None:
    job = stub.add_job(video_bytes, CALIBRATION, LIMITS)
    stub.inject("complete", "drop_after_write", times=2)  # outlasts the client's retries; the job did succeed
    runner = make_runner()
    claim_and_run(stub, runner)

    assert job.status == "succeeded"
    assert len(stub.requests("/complete")) == 3  # asked again and got 200, though heartbeats now answer 409
    assert not (runner.settings.work_root / "match-7").exists()  # delivered: the folder goes


@pytest.mark.parametrize(("refusal", "folder_kept"), [("cancel", True), ("delete", False)])
def test_refusal_kills_the_process_group(stub: StubServer, make_runner: Make, video_bytes: bytes,
                                         refusal: str, folder_kept: bool) -> None:
    job = stub.add_job(video_bytes, CALIBRATION, LIMITS)
    runner = make_runner("fakes:run_hang")
    worker = threading.Thread(target=claim_and_run, args=(stub, runner))
    worker.start()
    pids_file = runner.settings.work_root / "match-7" / "cache" / "pids.json"
    wait_for(pids_file.exists)
    wait_for(lambda: ("tracking" in [stage for stage, _ in job.heartbeats]))
    pids = json.loads(pids_file.read_text())

    getattr(stub, refusal)(job.id)  # the next heartbeat answers 409 (cancel) or 404 (delete)
    worker.join(timeout=15)

    assert not worker.is_alive()
    wait_for(lambda: not alive(pids["job"]) and not alive(pids["child"]), timeout=5)
    assert job.failure is None  # nothing to report for a job that is no longer ours
    assert (runner.settings.work_root / "match-7").exists() == folder_kept


@pytest.mark.parametrize(
    ("fake", "error", "retryable", "in_log"),
    [
        ("fakes:run_user_error", "the court calibration is mirrored", False, "checking the calibration"),
        ("fakes:run_crash", "RuntimeError: boom", True, "Traceback"),
        ("fakes:run_killed", "the job was killed (Killed: 9)", True, "using too much memory"),
    ],
)
def test_failure_reports(stub: StubServer, make_runner: Make, video_bytes: bytes, fake: str, error: str,
                         retryable: bool, in_log: str) -> None:
    job = stub.add_job(video_bytes, CALIBRATION, LIMITS)
    runner = make_runner(fake)
    claim_and_run(stub, runner)

    assert job.failure is not None
    assert job.failure["error"] == error
    assert job.failure["retryable"] is retryable
    assert in_log in job.failure["log_tail"] and len(job.failure["log_tail"]) <= 8000
    assert (runner.settings.work_root / "match-7" / "source.mp4").exists()  # kept for the retry


def test_a_killed_job_leaves_no_processes(stub: StubServer, make_runner: Make, video_bytes: bytes) -> None:
    stub.add_job(video_bytes, CALIBRATION, LIMITS)
    runner = make_runner("fakes:run_killed")
    claim_and_run(stub, runner)

    pids = json.loads((runner.settings.work_root / "match-7" / "cache" / "pids.json").read_text())
    wait_for(lambda: not alive(pids["child"]), timeout=5)
    assert not (runner.settings.work_root / "match-7" / "job-group.json").exists()


def test_failure_of_a_deleted_job_drops_its_folder(stub: StubServer, make_runner: Make, video_bytes: bytes) -> None:
    job = stub.add_job(video_bytes, CALIBRATION, LIMITS)
    stub.inject("fail", "404")
    runner = make_runner("fakes:run_user_error")
    claim_and_run(stub, runner)

    assert job.failure is None
    assert not (runner.settings.work_root / "match-7").exists()


def test_prepare_failure_is_not_retryable(stub: StubServer, make_runner: Make, video_bytes: bytes) -> None:
    job = stub.add_job(video_bytes, {**CALIBRATION, "frame_width": 1920, "frame_height": 1080}, LIMITS)
    claim_and_run(stub, make_runner())

    assert job.failure is not None and job.failure["retryable"] is False
    assert "160x120" in job.failure["error"] and "1920x1080" in job.failure["error"]


def test_stop_kills_the_job_and_reports_it_retryable(stub: StubServer, make_runner: Make,
                                                       video_bytes: bytes) -> None:
    job = stub.add_job(video_bytes, CALIBRATION, LIMITS)
    runner = make_runner("fakes:run_hang")
    worker = threading.Thread(target=runner.serve)
    worker.start()
    pids_file = runner.settings.work_root / "match-7" / "cache" / "pids.json"
    wait_for(pids_file.exists)
    pids = json.loads(pids_file.read_text())

    runner.stop()  # what SIGTERM and SIGINT do
    worker.join(timeout=15)

    assert not worker.is_alive()
    assert job.failure == {"error": "the worker was stopped", "retryable": True, "log_tail": job.failure["log_tail"]}
    wait_for(lambda: not alive(pids["job"]) and not alive(pids["child"]), timeout=5)


def test_leftover_process_group_is_killed_at_startup(make_runner: Make, video_bytes: bytes, tmp_path: Path) -> None:
    runner = make_runner("fakes:run_hang")
    match = runner.settings.work_root / "match-3"
    (match / "cache").mkdir(parents=True)
    (match / "source.mp4").write_bytes(video_bytes)
    job_file = match / "job.json"
    job_file.write_text(json.dumps({
        "video": str(match / "source.mp4"), "prepared": str(match / "prepared.mp4"), "calibration": CALIBRATION,
        "limits": LIMITS, "work_dir": str(match / "cache"), "out_dir": str(match / "out"),
        "progress": str(match / "progress.json"), "failure": str(match / "failure.json"),
        "ffmpeg": runner.settings.ffmpeg, "ffprobe": runner.settings.ffprobe,
        "stage_weights": dict(runner.settings.stage_weights), "runner": "fakes:run_hang",
        "pipeline_version": "test", "device": None, "fit_workers": None, "ball_model": None,
    }))
    # Started by a parent that exits at once, like a worker that died: the group is an orphan.
    starter = ("import subprocess as s, sys; print(s.Popen([sys.executable, '-m', 'courtshot.worker.job_main', "
               f"{str(job_file)!r}], start_new_session=True, stdout=s.DEVNULL, stderr=s.DEVNULL).pid)")
    leader = int(subprocess.run([sys.executable, "-c", starter], capture_output=True, text=True, check=True).stdout)
    (match / "job-group.json").write_text(json.dumps({"pgid": leader, "job_file": str(job_file)}))
    wait_for((match / "cache" / "pids.json").exists)
    pids = json.loads((match / "cache" / "pids.json").read_text())

    # A recorded group that is not ours (another command line) must survive.
    stranger = subprocess.Popen(["sleep", "30"], start_new_session=True)
    (runner.settings.work_root / "match-4").mkdir()
    (runner.settings.work_root / "match-4" / "job-group.json").write_text(
        json.dumps({"pgid": stranger.pid, "job_file": "/elsewhere/job.json"}))

    runner.clean_work_root()

    wait_for(lambda: not alive(pids["job"]) and not alive(pids["child"]), timeout=5)
    assert stranger.poll() is None
    stranger.send_signal(signal.SIGKILL)
    stranger.wait()
    assert not (match / "job-group.json").exists()


def test_one_worker_per_work_root(stub: StubServer, make_runner: Make) -> None:
    first = make_runner()
    worker = threading.Thread(target=first.serve)
    worker.start()
    wait_for(lambda: stub.requests("/claim"))  # locked, cleaned up and polling

    with pytest.raises(WorkRootBusy):
        make_runner().serve()
    first.stop()
    worker.join(timeout=15)
    assert not worker.is_alive()


def test_old_work_folders_are_swept(make_runner: Make) -> None:
    runner = make_runner()
    old, recent = runner.settings.work_root / "match-1", runner.settings.work_root / "match-2"
    old.mkdir(parents=True)
    recent.mkdir()
    eight_days_ago = time.time() - 8 * 24 * 3600
    os.utime(old, (eight_days_ago, eight_days_ago))

    runner.clean_work_root()

    assert not old.exists() and recent.exists()
