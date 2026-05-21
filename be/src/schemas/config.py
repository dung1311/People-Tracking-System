from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from models.tracking_config import TrackingConfigBase

class TrackingConfigCreate(TrackingConfigBase):
    pass

class TrackingConfigRead(TrackingConfigBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class TrackingConfigUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config_data: Optional[dict] = None
    is_default: Optional[bool] = None
