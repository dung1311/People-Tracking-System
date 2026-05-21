from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
from models.tracking_session import TrackingSessionBase

class TrackingSessionCreate(BaseModel):
    name: str
    camera_ids: List[int]
    sct_config_id: Optional[int] = None
    mct_config_id: Optional[int] = None

class TrackingSessionRead(TrackingSessionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class TrackingSessionUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    total_frames: Optional[int] = None
    total_global_ids: Optional[int] = None
    avg_fps: Optional[float] = None
