from datetime import datetime
from typing import Optional

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel


class GalleryTrack(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    person_id: int = Field(index=True, unique=True)
    tracker_id: int = Field(index=True)
    is_mapped: bool = True
    cam_id: int
    frame_id: int
    bbox: list[float] = Field(sa_column=Column(JSON), default_factory=list)
    score: float = 0.0
    class_id: int = 0
    state: str = "UNCONFIRM"
    lost_age: int = 0
    hits: int = 1
    features: list = Field(sa_column=Column(JSON), default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.now)
