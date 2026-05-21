from typing import Optional
from models.camera import CameraBase

class CameraCreate(CameraBase):
    pass

class CameraRead(CameraBase):
    id: int
    
class CameraUpdate(CameraBase):
    name: Optional[str] = None
    source: Optional[str] = None
    source_type: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    config_path: Optional[str] = None
    location: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    has_calibration: Optional[bool] = None
    calibration_path: Optional[str] = None
