from typing import List
import os

from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile
from sqlmodel import Session, select
import shutil

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
    
    
    # Stop the pipeline first to prevent new insertions
    from core.stream_manager import stream_manager
    stream_manager.stop_pipeline(camera_id)
    
    # Cascade delete related records
    # Using delete() statement is more efficient than fetching and iterating
    from sqlalchemy import delete
    from models.video_segment import VideoSegment
    
    # Delete Tracks
    session.exec(delete(Track).where(Track.camera_id == camera_id))
    
    # Delete Video Segments
    session.exec(delete(VideoSegment).where(VideoSegment.camera_id == camera_id))
        
    session.delete(camera)
    session.commit()
    return {"ok": True}

@router.get("/{camera_id}/stream")
async def stream_camera(
    camera_id: int,
    session: Session = Depends(get_session)
):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    from core.stream_manager import stream_manager
    import cv2
    from fastapi.responses import StreamingResponse
    import io

    # Check if source is valid (simple check)
    source = camera.source
    # Logic to resolve path similar to frames.py if needed, or trust stream_manager
    # For now, let's duplicate the resolution logic or assume absolute/correct paths or StreamManager handles it.
    # The StreamManager code I wrote just takes `source`.
    # Let's do a quick fix for relative paths here as well if it's a file.
    if source == "mct_demo.mp4" or (not source.startswith("rtsp") and not os.path.exists(source)):
         # Try looking in expected places
         if os.path.exists(f"../{source}"):
             source = f"../{source}"
         elif os.path.exists(f"../data/videos/{os.path.basename(source)}"):
             source = f"../data/videos/{os.path.basename(source)}"

    pipeline = stream_manager.create_pipeline(camera_id, source)
    
    def frame_generator():
        # pipeline.run_generator() yields frames
        # We need to encode them to jpg
        try:
             for frame in pipeline.run_generator():
                 ret, buffer = cv2.imencode('.jpg', frame)
                 if ret:
                     yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Stream error: {e}")
            
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.post("/upload-config")
async def upload_config(file: UploadFile = File(...)):
    """
    Upload a custom config file (YAML).
    Returns the saved path which can be assigned to a camera.
    """
    upload_dir = "configs/custom"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_location = f"{upload_dir}/{file.filename}"
    
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
        
    return {"path": file_location}
