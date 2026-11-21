"""An in-memory stand-in for the server's worker routes (server/API.md), with injectable faults."""

from __future__ import annotations

import json
import re
import socket
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

REQUIRED = ("report.json", "tracked.mp4", "plain.mp4", "summary.json", "positions.csv")
KNOWN = (*REQUIRED, "scene.json", "shots.csv", "ball_events.csv")
JOB_ROUTE = re.compile(r"^/api/worker/jobs/(\d+)/(video|heartbeat|complete|fail|files/([\w.]+))$")


@dataclass
class StubJob:
    id: int
    attempt: int
    match_id: int
    video: bytes
    calibration: dict[str, Any]
    limits: dict[str, Any]
    status: str = "queued"
    files: dict[str, bytearray] = field(default_factory=dict)
    heartbeats: list[tuple[str, float]] = field(default_factory=list)
    failure: dict[str, Any] | None = None


@dataclass
class Fault:
    route: str  # "video", "chunk", "heartbeat", ...
    action: str  # "drop", "drop_after_write", "500", "rewind", "404"
    times: int = 1
    after_bytes: int = 0  # "drop" on "video": bytes sent before the connection is cut
    name: str | None = None  # "chunk": only for this file


class StubServer:
    def __init__(self, token: str = "test-token", heartbeat_seconds: float = 0.1, chunk_bytes: int = 64 * 1024):
        self.token = token
        self.heartbeat_seconds = heartbeat_seconds
        self.chunk_bytes = chunk_bytes
        self.lock = threading.Lock()
        self.jobs: dict[int, StubJob] = {}
        self.queue: list[int] = []
        self.faults: list[Fault] = []
        self.log: list[dict[str, Any]] = []  # every request: method, path, headers that matter, answer
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self))
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self) -> StubServer:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def add_job(self, video: bytes, calibration: dict[str, Any], limits: dict[str, Any], *,
                job_id: int = 1, attempt: int = 1, match_id: int = 7) -> StubJob:
        job = StubJob(job_id, attempt, match_id, video, calibration, limits)
        with self.lock:
            self.jobs[job_id] = job
            self.queue.append(job_id)
        return job

    def claim_body(self, job: StubJob) -> dict[str, Any]:
        return {
            "job_id": job.id,
            "attempt": job.attempt,
            "match_id": job.match_id,
            "video": {"size_bytes": len(job.video), "ext": "mp4"},
            "calibration": job.calibration,
            "limits": job.limits,
            "heartbeat_seconds": self.heartbeat_seconds,
            "chunk_bytes": self.chunk_bytes,
        }

    def cancel(self, job_id: int) -> None:
        with self.lock:
            self.jobs[job_id].status = "cancelled"

    def delete(self, job_id: int) -> None:
        with self.lock:
            del self.jobs[job_id]

    def inject(self, route: str, action: str, **kwargs: Any) -> None:
        with self.lock:
            self.faults.append(Fault(route, action, **kwargs))

    def requests(self, path_end: str, method: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            return [r for r in self.log if r["path"].endswith(path_end) and method in (None, r["method"])]

    def take_fault(self, route: str, name: str | None = None) -> Fault | None:
        with self.lock:
            for fault in self.faults:
                if fault.route == route and fault.times > 0 and fault.name in (None, name):
                    fault.times -= 1
                    return fault
        return None


def _handler(stub: StubServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            self._route()

        def do_POST(self) -> None:
            self._route()

        def do_PATCH(self) -> None:
            self._route()

        def _route(self) -> None:
            url = urlparse(self.path)
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            entry = {"method": self.command, "path": url.path, "range": self.headers.get("Range"),
                     "attempt": self.headers.get("X-Job-Attempt"), "query": parse_qs(url.query), "bytes": len(body)}
            with stub.lock:
                stub.log.append(entry)
            if self.headers.get("Authorization") != f"Bearer {stub.token}":
                return self._json(401, {"detail": "no valid token"})
            if url.path == "/api/worker/claim" and self.command == "POST":
                return self._claim()
            match = JOB_ROUTE.match(url.path)
            if not match:
                return self._json(404, {"detail": "no such route"})
            job_id, action, name = int(match[1]), match[2], match[3]
            with stub.lock:
                job = stub.jobs.get(job_id)
            if job is None:
                return self._json(404, {"detail": "job not found"})
            attempt_ok = self.headers.get("X-Job-Attempt") == str(job.attempt)
            if action == "complete" and attempt_ok and job.status in ("running", "succeeded"):
                return self._complete(job)
            if not attempt_ok or job.status != "running":
                return self._json(409, {"detail": "job is not running for this attempt"})
            if action == "video":
                return self._video(job)
            if action == "heartbeat":
                fault = stub.take_fault("heartbeat")
                if fault and fault.action == "500":
                    return self._json(500, {"detail": "injected"})
                data = json.loads(body)
                job.heartbeats.append((data["stage"], data["progress"]))
                return self._json(200, {})
            if action == "fail":
                fault = stub.take_fault("fail")
                if fault and fault.action == "404":  # the match was deleted meanwhile
                    return self._json(404, {"detail": "job not found"})
                job.failure = json.loads(body)
                job.status = "failed"
                return self._json(200, {})
            return self._file(job, name, body)

        def _claim(self) -> None:
            with stub.lock:
                job = stub.jobs.get(stub.queue.pop(0)) if stub.queue else None
                if job is not None:
                    job.status = "running"
            if job is None:
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._json(200, stub.claim_body(job))

        def _complete(self, job: StubJob) -> None:
            missing = [n for n in REQUIRED if n not in job.files]
            if missing:
                return self._json(422, {"detail": f"missing {missing}"})
            job.status = "succeeded"
            fault = stub.take_fault("complete")
            if fault and fault.action == "drop_after_write":  # done, but the answer is lost
                return self._cut()
            self._json(200, {})

        def _video(self, job: StubJob) -> None:
            start = 0
            if rng := self.headers.get("Range"):
                start = int(re.match(r"bytes=(\d+)-", rng)[1])
            data = job.video[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(data)))
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(job.video) - 1}/{len(job.video)}")
            self.end_headers()
            fault = stub.take_fault("video")
            if fault and fault.action == "drop":
                self.wfile.write(data[: fault.after_bytes])
                return self._cut()
            self.wfile.write(data)

        def _file(self, job: StubJob, name: str, body: bytes) -> None:
            if name not in KNOWN:
                return self._json(422, {"detail": "unknown result file"})
            held = job.files.setdefault(name, bytearray())
            if self.command == "GET":
                return self._json(200, {"offset": len(held)})
            offset = int(parse_qs(urlparse(self.path).query)["offset"][0])
            fault = stub.take_fault("chunk", name)
            if fault and fault.action == "drop":
                return self._cut()
            if fault and fault.action == "500":
                return self._json(500, {"detail": "injected"})
            if fault and fault.action == "rewind":  # the server lost its last bytes
                del held[len(held) // 2:]
            if offset > len(held):
                return self._json(409, {"detail": "offset beyond the current size", "offset": len(held)})
            held[offset: offset + len(body)] = body
            if fault and fault.action == "drop_after_write":
                return self._cut()
            self._json(200, {"offset": len(held)})

        def _cut(self) -> None:
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            self.close_connection = True

        def _json(self, status: int, data: dict[str, Any]) -> None:
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler
