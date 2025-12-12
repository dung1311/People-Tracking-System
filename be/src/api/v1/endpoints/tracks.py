from typing import List

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from database.session import get_session
from models.track import Track
from schemas.track import TrackRead

router = APIRouter()

@router.get("/", response_model=List[TrackRead])
def read_tracks(
    *,
    session: Session = Depends(get_session),
    camera_id: int | None = None,
    track_id: int | None = None,
    offset: int = 0,
    limit: int = Query(default=100, le=1000),
):
    query = select(Track)
    if camera_id:
        query = query.where(Track.camera_id == camera_id)
    if track_id:
        query = query.where(Track.track_id == track_id)
        
    query = query.offset(offset).limit(limit)
    tracks = session.exec(query).all()
    return tracks
