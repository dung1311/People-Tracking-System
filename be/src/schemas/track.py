from models.track import TrackBase
from datetime import datetime

class TrackCreate(TrackBase):
    pass

class TrackRead(TrackBase):
    id: int
