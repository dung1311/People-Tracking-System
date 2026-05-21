from typing import Optional
from datetime import datetime
from sqlmodel import Field, SQLModel, Column, JSON

class TrackingConfigBase(SQLModel):
    name: str = Field(index=True)
    description: Optional[str] = None
    config_type: str = Field(index=True)  # "sct" | "mct"
    config_data: dict = Field(default_factory=dict, sa_column=Column(JSON))
    is_default: bool = Field(default=False)

class TrackingConfig(TrackingConfigBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
