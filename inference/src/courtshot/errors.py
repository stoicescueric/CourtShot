"""Failures a job reports to the server."""

from __future__ import annotations


class JobFailure(Exception):
    """A job failed for a reason worth showing to the user.

    retryable=False: the input is the problem (bad calibration, video too short,
    four players never visible together, ...); running it again fails the same way.
    retryable=True: a transient problem; the server may queue the job again.
    """

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
