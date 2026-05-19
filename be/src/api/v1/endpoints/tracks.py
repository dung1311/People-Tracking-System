"""Track query endpoints."""

from typing import List

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from api.v1.deps import get_current_user
from database.session import get_session
from models.track import Track
from models.user import User
from schemas.track import TrackRead

router = APIRouter()


@router.get("/", response_model=List[TrackRead])
def read_tracks(
    *,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
    camera_id: int | None = None,
    person_id: int | None = None,
    offset: int = 0,
    limit: int = Query(default=100, le=1000),
):
    query = select(Track)
    if camera_id:
        query = query.where(Track.camera_id == camera_id)
    if person_id:
        query = query.where(Track.person_id == person_id)

    query = query.offset(offset).limit(limit)
    tracks = session.exec(query).all()
    return tracks
