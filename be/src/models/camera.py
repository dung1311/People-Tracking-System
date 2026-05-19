"""Camera model — enhanced with calibration and source type."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class CameraBase(SQLModel):
    name: str = Field(index=True)
    source_type: str = Field(default="video")  # "rtsp" | "video" | "webcam"
    source_uri: str  # RTSP URL, MinIO object key, or device index
    description: Optional[str] = None
    is_active: bool = Field(default=True)
    location: Optional[str] = None  # Physical location
    resolution: Optional[str] = None  # e.g. "1920x1080"
    fps: Optional[int] = None

    # Calibration
    has_calibration: bool = Field(default=False)
    calibration_path: Optional[str] = None  # MinIO object key


class Camera(CameraBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_by: Optional[int] = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
