"""Tests for the court model and the pixel -> court homography."""

import numpy as np
import pytest

from courtshot.court import (
    COURT_MODEL,
    calibration_from_keypoints,
    court_to_pixel,
    fit_homography,
    pixel_to_court,
)


class TestPadelCourtModel:
    def test_keypoint_count(self):
        assert len(COURT_MODEL.KEYPOINTS) == 12

    def test_keypoints_within_bounds(self):
        for name, (x, y) in COURT_MODEL.KEYPOINTS.items():
            assert 0 <= x <= 10, f"{name} x={x} out of bounds"
            assert 0 <= y <= 20, f"{name} y={y} out of bounds"

    def test_line_count(self):
        assert len(COURT_MODEL.LINES) == 8

    def test_lines_reference_valid_keypoints(self):
        for a, b in COURT_MODEL.LINES:
            assert a in COURT_MODEL.KEYPOINTS, f"Unknown keypoint: {a}"
            assert b in COURT_MODEL.KEYPOINTS, f"Unknown keypoint: {b}"

    def test_court_dimensions(self):
        assert COURT_MODEL.COURT_WIDTH == 10.0
        assert COURT_MODEL.COURT_LENGTH == 20.0
        assert COURT_MODEL.NET_Y == 10.0


class TestHomography:
    def test_identity_homography(self):
        """When pixel coords == metre coords, H should be ~identity."""
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)]
        H, error = fit_homography(points, points)
        np.testing.assert_allclose(H, np.eye(3), atol=1e-6)
        assert error < 1e-6

    def test_known_scale_transform(self):
        """Pixel coords are 100x the metre coords."""
        px = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 2000.0), (0.0, 2000.0)]
        m = [(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)]
        H, error = fit_homography(px, m)
        assert error < 0.01
        np.testing.assert_allclose(pixel_to_court((500.0, 1000.0), H), (5.0, 10.0), atol=0.1)

    def test_pixel_to_court_roundtrip(self):
        px = [(100.0, 50.0), (900.0, 50.0), (800.0, 500.0), (200.0, 500.0)]
        m = [(0.0, 20.0), (10.0, 20.0), (10.0, 0.0), (0.0, 0.0)]
        H, _ = fit_homography(px, m)
        original = (500.0, 300.0)
        back = court_to_pixel(pixel_to_court(original, H), H)
        np.testing.assert_allclose(back, original, atol=0.5)

    def test_collinear_points_fail(self):
        pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        with pytest.raises(ValueError):
            fit_homography(pts, pts)


class TestCalibrationFromKeypoints:
    # a camera looking at the court: metres -> pixels by a plain scale and shift
    CLICKS = {name: (100.0 + 50.0 * x, 900.0 - 40.0 * y) for name, (x, y) in COURT_MODEL.KEYPOINTS.items()}

    def test_all_twelve_points(self):
        cal = calibration_from_keypoints(self.CLICKS, 1280, 1000)
        assert (cal.frame_width, cal.frame_height) == (1280, 1000)
        assert cal.reprojection_error < 1e-6
        assert len(cal.court_keypoints_px) == len(cal.court_keypoints_m) == 12
        H = np.array(cal.homography_matrix)
        np.testing.assert_allclose(pixel_to_court((350.0, 500.0), H), (5.0, 10.0), atol=1e-6)

    def test_unknown_names_are_skipped(self):
        clicks = dict(self.CLICKS, not_a_keypoint=(1.0, 2.0))
        assert len(calibration_from_keypoints(clicks, 1280, 1000).court_keypoints_px) == 12

    def test_needs_four_points(self):
        three = dict(list(self.CLICKS.items())[:3])
        with pytest.raises(ValueError):
            calibration_from_keypoints(three, 1280, 1000)
