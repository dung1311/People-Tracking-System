from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from database.session import get_session
from models.camera import Camera
from models.track import Track
from schemas.camera import CameraCreate, CameraRead, CameraUpdate

router = APIRouter()

@router.post("/", response_model=CameraRead)
def create_camera(*, session: Session = Depends(get_session), camera: CameraCreate):
    db_camera = Camera.from_orm(camera)
    session.add(db_camera)
    session.commit()
    session.refresh(db_camera)
    return db_camera

@router.get("/", response_model=List[CameraRead])
def read_cameras(
    *,
    session: Session = Depends(get_session),
    offset: int = 0,
    limit: int = Query(default=100, le=100),
):
    cameras = session.exec(select(Camera).offset(offset).limit(limit)).all()
    return cameras

@router.get("/{camera_id}", response_model=CameraRead)
def read_camera(*, session: Session = Depends(get_session), camera_id: int):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera

@router.patch("/{camera_id}", response_model=CameraRead)
def update_camera(
    *,
    session: Session = Depends(get_session),
    camera_id: int,
    camera: CameraUpdate,
):
    db_camera = session.get(Camera, camera_id)
    if not db_camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    camera_data = camera.dict(exclude_unset=True)
    for key, value in camera_data.items():
        setattr(db_camera, key, value)
        
    session.add(db_camera)
    session.commit()
    session.refresh(db_camera)
    return db_camera

@router.delete("/{camera_id}")
def delete_camera(*, session: Session = Depends(get_session), camera_id: int):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    tracks = session.exec(select(Track).where(Track.camera_id == camera_id)).all()
    for track in tracks:
        session.delete(track)
        
    session.delete(camera)
    session.commit()
    return {"ok": True}
