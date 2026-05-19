"""Camera request/response schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class CameraCreate(BaseModel):
    name: str
    source_type: str = "video"  # "rtsp" | "video" | "webcam"
    source_uri: str
    description: Optional[str] = None
    is_active: bool = True
    location: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None


class CameraRead(BaseModel):
    id: int
    name: str
    source_type: str
    source_uri: str
    description: Optional[str] = None
    is_active: bool
    location: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    has_calibration: bool
    calibration_path: Optional[str] = None
    created_by: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    source_type: Optional[str] = None
    source_uri: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    location: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
