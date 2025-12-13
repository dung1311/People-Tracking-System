from typing import Optional, Any
from datetime import datetime
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON
from pgvector.sqlalchemy import VECTOR

class TrackBase(SQLModel):
    camera_id: int = Field(foreign_key="camera.id")
    person_id: int  # ID assigned by the tracker
    frame_id: int
    bbox: list[float] = Field(sa_column=Column(JSON)) # [x1, y1, x2, y2]
    score: float
    class_id: int
    timestamp: datetime = Field(default_factory=datetime.now())
    feature: Any = Field(sa_type=VECTOR(512))
    
class Track(TrackBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
