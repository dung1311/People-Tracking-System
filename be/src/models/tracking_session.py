"""Tracking session — lifecycle wrapper around MCTPipeline3."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel


class TrackingSessionBase(SQLModel):
    name: str = Field(index=True)
    status: str = Field(default="created", index=True)  # created|running|stopping|completed|failed

    # Config snapshot (frozen when session is started)
    sct_config: Optional[dict] = Field(sa_column=Column(JSON), default=None)
    mct_config: Optional[dict] = Field(sa_column=Column(JSON), default=None)

    # Cameras participating
    camera_ids: list[int] = Field(sa_column=Column(JSON), default_factory=list)

    # Timing
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    # Output paths (MinIO object keys)
    output_video_path: Optional[str] = None
    output_txt_dir: Optional[str] = None

    # Stats
    total_frames: int = Field(default=0)
    total_global_ids: int = Field(default=0)
    avg_fps: float = Field(default=0.0)


class TrackingSession(TrackingSessionBase, table=True):
    __tablename__ = "tracking_session"
    id: Optional[int] = Field(default=None, primary_key=True)
    created_by: Optional[int] = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
