from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel

class VideoSegmentBase(SQLModel):
    camera_id: int = Field(foreign_key="camera.id", index=True)
    file_path: str
    start_time: datetime = Field(index=True)
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
class VideoSegment(VideoSegmentBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    __tablename__ = "video_segment"
