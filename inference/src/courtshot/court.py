"""
The padel court and the homography that maps image pixels onto its floor.

Court frame: origin at the near-left corner, x = 0..10 m across, y = 0..20 m
along, net at y = 10. The homography is only valid for points on the floor.

Original author: Xingnan Zhu.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from courtshot.errors import JobFailure
from courtshot.schemas.tracking import CourtCalibration

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PadelCourtModel:
    """Reference geometry of a padel court (20 m x 10 m)."""

    COURT_WIDTH: float = 10.0
    COURT_LENGTH: float = 20.0
    NET_Y: float = 10.0
    SERVICE_NEAR_Y: float = 3.0
    SERVICE_FAR_Y: float = 17.0

    # The points a calibration can be clicked on, in metres.
    KEYPOINTS: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "bottom_left": (0.0, 0.0),
            "bottom_right": (10.0, 0.0),
            "top_left": (0.0, 20.0),
            "top_right": (10.0, 20.0),
            "net_left": (0.0, 10.0),
            "net_right": (10.0, 10.0),
            "service_near_left": (0.0, 3.0),
            "service_near_center": (5.0, 3.0),
            "service_near_right": (10.0, 3.0),
            "service_far_left": (0.0, 17.0),
            "service_far_center": (5.0, 17.0),
            "service_far_right": (10.0, 17.0),
        }
    )

    # The court lines, as pairs of keypoints.
    LINES: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("bottom_left", "bottom_right"),
            ("top_left", "top_right"),
            ("bottom_left", "top_left"),
            ("bottom_right", "top_right"),
            ("net_left", "net_right"),
            ("service_near_left", "service_near_right"),
            ("service_far_left", "service_far_right"),
            ("service_near_center", "service_far_center"),
        ]
    )


COURT_MODEL = PadelCourtModel()


def fit_homography(
    keypoints_px: list[tuple[float, float]],
    keypoints_m: list[tuple[float, float]],
) -> tuple[np.ndarray, float]:
    """Pixel -> court-metres homography, least squares on all points, and its
    RMS error in metres.

    All points are used: OpenCV's RANSAC threshold is in destination units
    (metres here), so at any useful setting it rejects nothing or hides a bad
    click instead of reporting it.

    Raises ValueError when no homography fits (fewer than 4 usable points).
    """
    H, error, _ = _fit(keypoints_px, keypoints_m)
    return H, error


def _fit(
    keypoints_px: list[tuple[float, float]], keypoints_m: list[tuple[float, float]]
) -> tuple[np.ndarray, float, np.ndarray]:
    """The homography, its RMS error and the error of each point (metres)."""
    src = np.array(keypoints_px, dtype=np.float64)
    dst = np.array(keypoints_m, dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None or not np.all(np.isfinite(H)):
        raise ValueError("Homography computation failed")
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    per_point = np.linalg.norm(projected - dst, axis=1)
    return H, float(np.sqrt(np.mean(per_point**2))), per_point


def calibration_from_keypoints(
    keypoints_px: dict[str, tuple[float, float]],
    frame_width: int,
    frame_height: int,
) -> CourtCalibration:
    """A CourtCalibration from court keypoints marked in the image.

    keypoints_px maps keypoint names (see COURT_MODEL.KEYPOINTS) to pixel
    positions; at least 4 are needed. Raises ValueError otherwise.
    """
    names = _known(keypoints_px)
    if len(names) < 4:
        raise ValueError(f"Need at least 4 keypoints, got {len(names)}. Valid names: {list(COURT_MODEL.KEYPOINTS)}")
    px_coords = [_pair(keypoints_px[n]) for n in names]
    m_coords = [COURT_MODEL.KEYPOINTS[n] for n in names]
    H, error = fit_homography(px_coords, m_coords)
    return CourtCalibration(
        frame_width=frame_width,
        frame_height=frame_height,
        homography_matrix=H.tolist(),
        court_keypoints_px=px_coords,
        court_keypoints_m=m_coords,
        reprojection_error=error,
    )


def _known(keypoints_px: dict[str, tuple[float, float]]) -> list[str]:
    for name in keypoints_px:
        if name not in COURT_MODEL.KEYPOINTS:
            logger.warning("Unknown keypoint '%s', skipping", name)
    return [n for n in keypoints_px if n in COURT_MODEL.KEYPOINTS]


def _pair(p) -> tuple[float, float]:
    x, y = p
    return float(x), float(y)


# ---------------------------------------------------------------------------
# checking clicked points
# ---------------------------------------------------------------------------

MIN_CHECKED_POINTS = 6


def checked_calibration(
    points: dict[str, tuple[float, float]],
    frame_width: int,
    frame_height: int,
    max_error_m: float,
) -> CourtCalibration:
    """A calibration from clicked points, refused when it cannot be trusted.

    Raises JobFailure when there are fewer than MIN_CHECKED_POINTS known
    points, a point lies outside the frame, the fitted matrix cannot be
    inverted, the RMS error is above max_error_m, or the clicks are mirrored
    (a near point not below a far one, a left point not left of a right one).
    """
    names = _known(points)
    if len(names) < MIN_CHECKED_POINTS:
        raise JobFailure(
            f"The court calibration needs at least {MIN_CHECKED_POINTS} marked points; it has {len(names)}."
        )
    px = {n: _pair(points[n]) for n in names}
    for n, (x, y) in px.items():
        if not (np.isfinite(x) and np.isfinite(y) and 0 <= x <= frame_width and 0 <= y <= frame_height):
            raise JobFailure(
                f"Calibration point {n} ({x:g}, {y:g}) lies outside the {frame_width}x{frame_height} frame."
            )
    _check_not_mirrored(px)

    try:
        H, error, per_point = _fit(list(px.values()), [COURT_MODEL.KEYPOINTS[n] for n in names])
    except ValueError:
        raise JobFailure("No court fits the calibration points. Are they all on one line?") from None
    if np.linalg.cond(H) >= 1.0 / np.finfo(float).eps:
        raise JobFailure("The fitted court cannot be inverted. Are the calibration points all on one line?")
    if error > max_error_m:
        worst = names[int(np.argmax(per_point))]
        raise JobFailure(
            f"The court calibration is off by {error:.2f} m on average (limit {max_error_m:.2f} m); "
            f"the worst point is {worst}, {float(per_point.max()):.2f} m off."
        )
    return CourtCalibration(
        frame_width=frame_width,
        frame_height=frame_height,
        homography_matrix=H.tolist(),
        court_keypoints_px=list(px.values()),
        court_keypoints_m=[COURT_MODEL.KEYPOINTS[n] for n in names],
        reprojection_error=error,
    )


def _check_not_mirrored(px: dict[str, tuple[float, float]]) -> None:
    """Points across the court (same court y) must run left to right in the
    image, and of two points along it (same court x) the nearer must be lower."""
    court = COURT_MODEL.KEYPOINTS
    for a, (xa, ya) in px.items():
        for b, (xb, yb) in px.items():
            (ca_x, ca_y), (cb_x, cb_y) = court[a], court[b]
            if ca_y == cb_y and ca_x < cb_x and not xa < xb:
                raise JobFailure(f"Calibration point {a} must be left of {b} in the image: the points look mirrored.")
            if ca_x == cb_x and ca_y < cb_y and not ya > yb:
                raise JobFailure(
                    f"Calibration point {a} must be lower in the image than {b}: the points look mirrored."
                )


def pixel_to_court(point_px: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """An image point mapped onto the court floor, in metres."""
    x, y = cv2.perspectiveTransform(np.array([[point_px]], dtype=np.float64), H)[0, 0]
    return float(x), float(y)


def court_to_pixel(point_m: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """A point on the court floor (metres) mapped into the image."""
    x, y = cv2.perspectiveTransform(np.array([[point_m]], dtype=np.float64), np.linalg.inv(H))[0, 0]
    return float(x), float(y)
