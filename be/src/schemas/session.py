"""Tracking session schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class SessionCreate(BaseModel):
    name: str
    camera_ids: list[int]
    sct_config_id: Optional[int] = None  # Use stored config; None = default
    mct_config_id: Optional[int] = None


class SessionRead(BaseModel):
    id: int
    name: str
    status: str
    camera_ids: list[int]
    sct_config: Optional[dict] = None
    mct_config: Optional[dict] = None
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    total_frames: int
    total_global_ids: int
    avg_fps: float
    output_video_path: Optional[str] = None
    created_by: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionUpdate(BaseModel):
    name: Optional[str] = None
