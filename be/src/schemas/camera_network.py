from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
from models.camera_network import CameraNetworkBase
from schemas.camera import CameraRead

class CameraNetworkCreate(BaseModel):
    name: str
    sct_config_id: Optional[int] = None
    mct_config_id: Optional[int] = None

class CameraNetworkRead(CameraNetworkBase):
    id: int
    created_at: datetime
    cameras: List[CameraRead] = []

    class Config:
        from_attributes = True

class CameraNetworkUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    total_frames: Optional[int] = None
    total_global_ids: Optional[int] = None
    avg_fps: Optional[float] = None
