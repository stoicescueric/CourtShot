"""
Tracking data: what the detectors write for every frame, and the court
calibration that turns image pixels into court metres.

Court frame: origin at the near-left corner, x = 0..10 m across, y = 0..20 m
along, net at y = 10.

These classes are what a tracking cache holds, so they live at a fixed import
path (see courtshot.tracking.cache).

Original author: Xingnan Zhu.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Position2D(BaseModel):
    """A point on the court floor, in metres."""

    x: float = Field(..., ge=0, le=10, description="Across the court (0-10 m)")
    y: float = Field(..., ge=0, le=20, description="Along the court (0-20 m)")


class BoundingBox(BaseModel):
    """A box in image pixels."""

    x1: float
    y1: float
    x2: float
    y2: float


class PoseKeypoint(BaseModel):
    """One body keypoint (COCO name) in image pixels."""

    name: str
    x: float
    y: float
    confidence: float = Field(..., ge=0, le=1)


class PlayerFrame(BaseModel):
    """One player in one frame."""

    frame_id: int
    timestamp_ms: float
    player_id: str
    team_id: str | None = None
    bbox: BoundingBox
    position: Position2D | None = None  # feet on the court; None without a calibration
    confidence: float = Field(..., ge=0, le=1)
    keypoints: list[PoseKeypoint] = Field(default_factory=list)


class BallFrame(BaseModel):
    """The ball in one frame; bbox is None when it was not found."""

    frame_id: int
    timestamp_ms: float
    bbox: BoundingBox | None = None
    confidence: float = Field(0.0, ge=0, le=1)
    candidates: list[tuple[float, float, float]] = Field(
        default_factory=list,
        description="All ball-like detections in this frame as (x, y, confidence) "
        "in pixels, the one in bbox first. Lets trajectory linking recover the "
        "real ball when the top detection is a reflection or a spare ball.",
    )


class CourtCalibration(BaseModel):
    """Homography from image pixels to court metres (valid on the floor)."""

    schema_version: str = "0.1.0"
    frame_width: int
    frame_height: int
    homography_matrix: list[list[float]] = Field(..., description="3x3 homography matrix as nested list")
    court_keypoints_px: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Court keypoints in pixel coordinates",
    )
    court_keypoints_m: list[tuple[float, float]] = Field(
        default_factory=list,
        description="The same keypoints in metres",
    )
    reprojection_error: float | None = None
