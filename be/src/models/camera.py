from typing import Optional
from datetime import datetime
from sqlmodel import Field, SQLModel

class CameraBase(SQLModel):
    name: str = Field(index=True)
    source: str  # RTSP URL, file path, or device index
    source_type: str = Field(default="video")  # "rtsp" | "video" | "webcam"
    description: Optional[str] = None
    is_active: bool = Field(default=True)
    config_path: str = Field(default="configs/sct_config.yaml")
    
    location: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    
    # Calibration (mandatory for MCT)
    has_calibration: bool = Field(default=False)
    calibration_path: Optional[str] = None  # MinIO or local path

class Camera(CameraBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
