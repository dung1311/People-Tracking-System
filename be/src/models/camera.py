from typing import Optional
from datetime import datetime
from sqlmodel import Field, SQLModel, Relationship
from sqlalchemy import Column, JSON

class CameraBase(SQLModel):
    network_id: Optional[int] = Field(default=None, foreign_key="cameranetwork.id")

    name: str = Field(index=True)
    source: str  # RTSP URL, file path, or device index
    source_type: str = Field(default="video")  # "rtsp" | "video" | "webcam"
    description: Optional[str] = None
    is_active: bool = Field(default=True)
    is_primary: bool = Field(default=True)
    config_path: str = Field(default="configs/sct_config.yaml")
    
    location: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    
    # Calibration (mandatory for MCT)
    has_calibration: bool = Field(default=False)
    calibration_path: Optional[str] = None  # MinIO or local path
    
    # ROIs configured for this camera
    rois: Optional[list[dict]] = Field(default=None, sa_column=Column(JSON))

class Camera(CameraBase, table=True):
    network: Optional["CameraNetwork"] = Relationship(back_populates="cameras")
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
