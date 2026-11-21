"""Server settings, read from COURTSHOT_* environment variables."""

from ipaddress import ip_address
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

GIB = 1024**3
MIB = 1024**2

SERVER_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COURTSHOT_", frozen=True)

    data_dir: Path
    # Scheme, host and port only, exactly as browsers send it in the Origin header.
    public_origin: str = Field("http://localhost:8000", pattern=r"^https?://[^/\s]+$")

    min_video_seconds: float = Field(10, gt=0)
    max_video_seconds: float = Field(900, gt=0)
    max_upload_bytes: int = Field(8 * GIB, gt=0)
    chunk_bytes: int = Field(8 * MIB, gt=0)
    max_chunk_bytes: int = Field(16 * MIB, gt=0)

    calibration_max_error_m: float = Field(0.25, gt=0)
    # Never below 6: the worker refuses a calibration with fewer points (API.md, Calibration).
    min_calibration_points: int = Field(6, ge=6)

    lease_seconds: int = Field(90, gt=0)
    max_attempts: int = Field(3, ge=1)
    retry_delay_seconds: int = Field(60, ge=0)
    heartbeat_seconds: int = Field(15, gt=0)
    worker_online_seconds: int = Field(30, gt=0)

    free_space_reserve_bytes: int = Field(5 * GIB, ge=0)
    trusted_proxy: str = ""
    secure_cookies: bool = True
    incomplete_upload_days: int = Field(7, gt=0)

    web_dist: Path = SERVER_DIR / "web" / "dist"

    @field_validator("trusted_proxy")
    @classmethod
    def _ip_or_empty(cls, value: str) -> str:
        return str(ip_address(value)) if value else value

    @model_validator(mode="after")
    def _consistent(self) -> "Settings":
        if self.chunk_bytes > self.max_chunk_bytes:
            raise ValueError("CHUNK_BYTES must not exceed MAX_CHUNK_BYTES")
        if self.min_video_seconds > self.max_video_seconds:
            raise ValueError("MIN_VIDEO_SECONDS must not exceed MAX_VIDEO_SECONDS")
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def session_cookie(self) -> str:
        # The __Host- prefix needs Secure, Path=/ and no Domain; browsers refuse it otherwise.
        return "__Host-courtshot_session" if self.secure_cookies else "courtshot_session"
