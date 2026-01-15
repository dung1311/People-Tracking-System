from typing import Optional
from datetime import datetime
from sqlmodel import Field, SQLModel

class CameraBase(SQLModel):
    name: str = Field(index=True)
    source: str  # RTSP URL or file path
    description: Optional[str] = None
    is_active: bool = Field(default=True)
    config_path: str = Field(default="configs/sct_config.yaml")

class Camera(CameraBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
