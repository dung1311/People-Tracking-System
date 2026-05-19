"""Tracking configuration stored in DB instead of YAML files."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel


class TrackingConfigBase(SQLModel):
    name: str = Field(index=True)
    description: Optional[str] = None
    config_type: str = Field(index=True)  # "sct" | "mct"
    config_data: dict = Field(sa_column=Column(JSON), default_factory=dict)
    is_default: bool = Field(default=False)


class TrackingConfig(TrackingConfigBase, table=True):
    __tablename__ = "tracking_config"
    id: Optional[int] = Field(default=None, primary_key=True)
    created_by: Optional[int] = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
