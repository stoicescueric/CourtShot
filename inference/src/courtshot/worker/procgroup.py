"""The job's process group (job_main, the 3D-fit pool, ffmpeg): record it, find it again, kill it."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path

from courtshot.worker.progress import write_json

logger = logging.getLogger(__name__)

JOB_MODULE = "courtshot.worker.job_main"


def record(path: Path, pgid: int, job_file: Path) -> None:
    write_json(path, {"pgid": pgid, "job_file": str(job_file)})


def kill_group(pgid: int, grace: float, leader: subprocess.Popen[bytes] | None = None) -> None:
    """SIGTERM to the group, `grace` seconds to end, then SIGKILL. `leader` is reaped if it is our child."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if leader is not None:
            leader.poll()  # a zombie leader still counts as a member of the group
        if not _group_alive(pgid):
            return
        time.sleep(0.1)
    logger.warning("process group %d outlived SIGTERM; sending SIGKILL", pgid)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    if leader is not None:
        leader.wait()


def kill_recorded(path: Path, grace: float) -> None:
    """Kill the group a previous run of the worker recorded in `path`, if it is still alive and ours."""
    try:
        recorded = json.loads(path.read_text())
        pgid, job_file = int(recorded["pgid"]), str(recorded["job_file"])
    except (OSError, ValueError, KeyError, TypeError):
        path.unlink(missing_ok=True)
        return
    members = _group_members(pgid)
    leader = members.get(pgid)
    # The leader's pid cannot be reused while its group lives on, so a group without a leader is still ours.
    if members and (leader is None or (JOB_MODULE in leader and job_file in leader)):
        logger.warning("killing process group %d left over from a previous run (%s)", pgid, job_file)
        kill_group(pgid, grace)
    path.unlink(missing_ok=True)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, not ours to signal
        pass
    return True


def _group_members(pgid: int) -> dict[int, str]:
    """pid -> command line of every process in the group."""
    listing = subprocess.run(["ps", "-axww", "-o", "pid=,pgid=,command="], capture_output=True, text=True).stdout
    members = {}
    for line in listing.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and fields[1] == str(pgid):
            members[int(fields[0])] = fields[2]
    return members
