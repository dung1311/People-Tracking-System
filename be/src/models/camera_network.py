from typing import Optional, List
from datetime import datetime
from sqlmodel import Field, SQLModel, Column, JSON, Relationship

class CameraNetworkBase(SQLModel):
    name: str
    status: str = Field(default="created")  # "created" | "running" | "stopping" | "completed" | "failed"
    sct_config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    mct_config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    output_video_path: Optional[str] = None
    output_txt_dir: Optional[str] = None
    total_frames: int = Field(default=0)
    total_global_ids: int = Field(default=0)
    avg_fps: float = Field(default=0.0)

class CameraNetwork(CameraNetworkBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    cameras: List["Camera"] = Relationship(
        back_populates="network", 
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
