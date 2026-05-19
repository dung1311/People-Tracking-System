"""Tracking config schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ConfigCreate(BaseModel):
    name: str
    description: Optional[str] = None
    config_type: str  # "sct" | "mct"
    config_data: dict
    is_default: bool = False


class ConfigRead(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    config_type: str
    config_data: dict
    is_default: bool
    created_by: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConfigUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config_data: Optional[dict] = None
    is_default: Optional[bool] = None


class ConfigValidateRequest(BaseModel):
    config_type: str  # "sct" | "mct"
    config_data: dict
