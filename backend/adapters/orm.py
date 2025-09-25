from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Table, ARRAY, MetaData, Date, Float
from sqlalchemy.orm import mapper, relationship

from ..domain.model import STrack

metadata = MetaData()

stracks = Table(
    "stracks", metadata,
    Column("track_id", Integer, primary_key=True),
    Column("start_time", Date),
    Column("end_time", Date),
    Column("start_frame", Integer),
    Column("end_frame", Integer),
    Column("bboxes", ARRAY(Float)),  # Storing as a flat list for simplicity
)

embeddings = Table(
    "embeddings", metadata,
    Column("id", Integer, primary_key=True),
    Column("embeddings", Vector(512)),  # Assuming 512-dim embeddings
)
