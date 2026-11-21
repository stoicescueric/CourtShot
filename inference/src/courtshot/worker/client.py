"""HTTP client for the worker routes of the CourtShot server (server/API.md)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

GONE_STATUSES = (404, 409, 410)
NETWORK_ERRORS = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)


class JobGone(Exception):
    """The server refused a call about the job (404, 409 or 410): stop working on it."""

    def __init__(self, status: int, detail: str = "") -> None:
        super().__init__(f"the server answered {status}: {detail}".rstrip(": "))
        self.status = status


class ServerUnavailable(Exception):
    """A network error or a 5xx answer that outlived the retries."""


@dataclass(frozen=True)
class JobRef:
    id: int
    attempt: int


class ServerClient:
    """One requests.Session with the bearer token. Not shared across threads: use copy()."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: tuple[float, float] = (10.0, 120.0),
        tries: int = 4,
        backoff: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.tries = tries
        self.backoff = backoff
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def copy(self, *, timeout: tuple[float, float] | None = None, tries: int | None = None) -> ServerClient:
        return ServerClient(self.base_url, self.token, timeout=timeout or self.timeout, tries=tries or self.tries,
                            backoff=self.backoff)

    def claim(self, info: dict[str, Any]) -> dict[str, Any] | None:
        # One try: a lost answer to a claim is covered by the lease, and the loop polls again anyway.
        response = self._request("POST", "/claim", tries=1, json={"info": info})
        if response.status_code == 204:
            return None
        response.raise_for_status()
        claim: dict[str, Any] = response.json()
        return claim

    def heartbeat(self, job: JobRef, stage: str, progress: float) -> None:
        self._job_call("POST", job, "/heartbeat", json={"stage": stage, "progress": progress})

    def complete(self, job: JobRef) -> None:
        self._job_call("POST", job, "/complete")

    def fail(self, job: JobRef, error: str, retryable: bool, log_tail: str) -> None:
        body = {"error": error[:500], "retryable": retryable, "log_tail": log_tail[-8000:]}
        self._job_call("POST", job, "/fail", json=body)

    def file_offset(self, job: JobRef, name: str) -> int:
        return int(self._job_call("GET", job, f"/files/{name}").json()["offset"])

    def send_chunk(self, job: JobRef, name: str, offset: int, data: bytes) -> int:
        """Write `data` at `offset`; returns the server's offset afterwards.

        A 409 that carries an offset means our offset was beyond the server's file: continue from its offset.
        """
        response = self._request(
            "PATCH", f"/jobs/{job.id}/files/{name}", job=job, params={"offset": offset}, data=data,
            headers={"Content-Type": "application/octet-stream"},
        )
        if response.status_code == 409 and isinstance(_body(response).get("offset"), int):
            return int(_body(response)["offset"])
        _raise_for(response)
        return int(response.json()["offset"])

    def open_video(self, job: JobRef, start: int) -> requests.Response:
        """The job's video from byte `start`, streamed: 206 for a range, 200 for the whole file."""
        headers = {"Range": f"bytes={start}-"} if start else {}
        response = self._request("GET", f"/jobs/{job.id}/video", job=job, headers=headers, stream=True)
        _raise_for(response)
        return response

    def _job_call(self, method: str, job: JobRef, path: str, **kwargs: Any) -> requests.Response:
        response = self._request(method, f"/jobs/{job.id}{path}", job=job, **kwargs)
        _raise_for(response)
        return response

    def _request(
        self, method: str, path: str, *, job: JobRef | None = None, tries: int | None = None,
        headers: dict[str, str] | None = None, **kwargs: Any,
    ) -> requests.Response:
        """Retries with backoff on network errors and 5xx; returns any other answer."""
        headers = dict(headers or {})
        if job is not None:
            headers["X-Job-Attempt"] = str(job.attempt)
        url = f"{self.base_url}/api/worker{path}"
        failure = ""
        for attempt in range(tries or self.tries):
            if attempt:
                time.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                response = self.session.request(method, url, headers=headers, timeout=self.timeout, **kwargs)
            except NETWORK_ERRORS as exc:
                failure = f"{type(exc).__name__}: {exc}"
                continue
            if response.status_code < 500:
                return response
            failure = f"{response.status_code} {_body(response).get('detail', '')}".strip()
            response.close()
        raise ServerUnavailable(f"{method} {path} failed: {failure}")


def _body(response: requests.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _raise_for(response: requests.Response) -> None:
    if response.status_code in GONE_STATUSES:
        raise JobGone(response.status_code, str(_body(response).get("detail", "")))
    response.raise_for_status()
