"""Request bodies (server/API.md). Responses are built in views.py."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

CALIBRATION_POINTS = (
    "bottom_left",
    "bottom_right",
    "top_left",
    "top_right",
    "net_left",
    "net_right",
    "service_near_left",
    "service_near_center",
    "service_near_right",
    "service_far_left",
    "service_far_center",
    "service_far_right",
)

Stage = Literal["download", "prepare", "tracking", "appearance", "analysis", "ball3d", "render", "upload"]

Title = Annotated[str, Field(min_length=1, max_length=200)]


class LoginBody(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=1024)


class NewMatch(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: Title
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_s: FiniteFloat = Field(gt=0)
    frame_width: int = Field(gt=0, le=16384)
    frame_height: int = Field(gt=0, le=16384)


class TitleBody(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: Title


class CalibrationBody(BaseModel):
    """Structural check only: known names, points inside the frame. The fit is the worker's."""

    points: dict[str, tuple[FiniteFloat, FiniteFloat]]
    frame_width: int = Field(gt=0, le=16384)
    frame_height: int = Field(gt=0, le=16384)

    @model_validator(mode="after")
    def _points_inside_frame(self) -> "CalibrationBody":
        unknown = sorted(set(self.points) - set(CALIBRATION_POINTS))
        if unknown:
            raise ValueError(f"unknown point names: {', '.join(unknown)}")
        for name, (x, y) in self.points.items():
            if not (0 <= x <= self.frame_width and 0 <= y <= self.frame_height):
                raise ValueError(f"{name} is outside the frame")
        return self


class WorkerInfo(BaseModel):
    hostname: str | None = Field(None, max_length=200)
    chip: str | None = Field(None, max_length=200)
    memory_gb: FiniteFloat | None = None
    device: str | None = Field(None, max_length=200)
    pipeline_version: str | None = Field(None, max_length=200)


class ClaimBody(BaseModel):
    info: WorkerInfo


class HeartbeatBody(BaseModel):
    stage: Stage
    progress: FiniteFloat = Field(ge=0, le=1)


class FailBody(BaseModel):
    error: str = Field(min_length=1)
    retryable: bool
    log_tail: str | None = None
