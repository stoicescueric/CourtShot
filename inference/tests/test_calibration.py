"""
Tests for court.checked_calibration: the fit from clicked points and every
reason a calibration is refused before a job runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from courtshot.court import COURT_MODEL, checked_calibration, pixel_to_court
from courtshot.errors import JobFailure

REPO = Path(__file__).resolve().parents[2]
W, H = 1920, 1080


def _clicks() -> dict[str, tuple[float, float]]:
    """All twelve points of final_f, as clicked."""
    saved = json.loads((REPO / "videos" / "final_f_calibration.json").read_text(encoding="utf-8"))
    return {name: tuple(p) for name, p in saved["labeled_keypoints"].items()}


def _check(points, max_error_m: float = 0.25):
    return checked_calibration(points, W, H, max_error_m)


def _refused(points, match: str, max_error_m: float = 0.25) -> JobFailure:
    with pytest.raises(JobFailure, match=match) as info:
        _check(points, max_error_m)
    assert info.value.retryable is False
    return info.value


class TestAccepted:
    def test_clicked_points_of_final_f(self):
        cal = _check(_clicks())
        assert cal.reprojection_error < 0.05  # well below the 0.25 m limit
        assert (cal.frame_width, cal.frame_height) == (W, H)
        assert len(cal.court_keypoints_px) == len(cal.court_keypoints_m) == 12
        x, y = pixel_to_court(_clicks()["service_near_center"], np.array(cal.homography_matrix))
        assert abs(x - 5.0) < 0.1 and abs(y - 3.0) < 0.1

    def test_six_points_are_enough(self):
        six = {n: p for n, p in _clicks().items() if n.startswith("service")}
        assert len(_check(six).court_keypoints_px) == 6

    def test_unknown_names_are_ignored(self):
        assert len(_check(dict(_clicks(), elbow=(10.0, 10.0))).court_keypoints_px) == 12


class TestRefused:
    def test_fewer_than_six_known_points(self):
        five = dict(list(_clicks().items())[:5])
        _refused(dict(five, elbow=(10.0, 10.0)), "at least 6")

    def test_point_outside_the_frame(self):
        _refused(dict(_clicks(), top_left=(597.3, -4.0)), "top_left.*outside")
        _refused(dict(_clicks(), bottom_right=(W + 1.0, 954.5)), "bottom_right.*outside")
        _refused(dict(_clicks(), net_left=(float("nan"), 527.3)), "net_left.*outside")

    def test_error_above_the_limit_names_the_worst_point(self):
        moved = dict(_clicks(), net_right=(1451.5 + 120.0, 524.0 + 60.0))
        failure = _refused(moved, "off by")
        assert "net_right" in failure.message
        _check(moved, max_error_m=10.0)  # the same points pass a looser limit

    def test_left_and_right_swapped(self):
        mirrored = {name: (W - x, y) for name, (x, y) in _clicks().items()}
        _refused(mirrored, "mirrored")

    def test_near_and_far_swapped(self):
        flipped = {name: (x, H - y) for name, (x, y) in _clicks().items()}
        _refused(flipped, "mirrored")

    def test_one_pair_clicked_the_wrong_way_round(self):
        clicks = _clicks()
        swapped = dict(
            clicks, service_near_left=clicks["service_near_right"], service_near_right=clicks["service_near_left"]
        )
        _refused(swapped, "service_near_left must be left of service_near_center")

    def test_points_on_one_line_give_a_matrix_that_cannot_be_inverted(self):
        # in order along both court axes, so not mirrored, but all on one image line
        line = {n: (100.0 + 40.0 * (x + y), 1000.0 - 30.0 * (x + y)) for n, (x, y) in COURT_MODEL.KEYPOINTS.items()}
        _refused(line, "cannot be inverted")
