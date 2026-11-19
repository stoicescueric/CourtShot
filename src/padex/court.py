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

from padex.schemas.tracking import CourtCalibration

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
    """Pixel -> court-metres homography (RANSAC) and its RMS error in metres
    on the points it kept.

    Raises ValueError when no homography fits (fewer than 4 usable points).
    """
    src = np.array(keypoints_px, dtype=np.float64)
    dst = np.array(keypoints_m, dtype=np.float64)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        raise ValueError("Homography computation failed")

    inliers = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(src[inliers].reshape(-1, 1, 2), H).reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((projected - dst[inliers]) ** 2, axis=1))))
    return H, error


def calibration_from_keypoints(
    keypoints_px: dict[str, tuple[float, float]],
    frame_width: int,
    frame_height: int,
) -> CourtCalibration:
    """A CourtCalibration from court keypoints marked in the image.

    keypoints_px maps keypoint names (see COURT_MODEL.KEYPOINTS) to pixel
    positions; at least 4 are needed. Raises ValueError otherwise.
    """
    px_coords: list[tuple[float, float]] = []
    m_coords: list[tuple[float, float]] = []
    for name, px in keypoints_px.items():
        if name not in COURT_MODEL.KEYPOINTS:
            logger.warning("Unknown keypoint '%s', skipping", name)
            continue
        px_coords.append(px)
        m_coords.append(COURT_MODEL.KEYPOINTS[name])

    if len(px_coords) < 4:
        raise ValueError(f"Need at least 4 keypoints, got {len(px_coords)}. Valid names: {list(COURT_MODEL.KEYPOINTS)}")

    H, error = fit_homography(px_coords, m_coords)
    return CourtCalibration(
        frame_width=frame_width,
        frame_height=frame_height,
        homography_matrix=H.tolist(),
        court_keypoints_px=px_coords,
        court_keypoints_m=m_coords,
        reprojection_error=error,
    )


def pixel_to_court(point_px: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """An image point mapped onto the court floor, in metres."""
    x, y = cv2.perspectiveTransform(np.array([[point_px]], dtype=np.float64), H)[0, 0]
    return float(x), float(y)


def court_to_pixel(point_m: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """A point on the court floor (metres) mapped into the image."""
    x, y = cv2.perspectiveTransform(np.array([[point_m]], dtype=np.float64), np.linalg.inv(H))[0, 0]
    return float(x), float(y)
