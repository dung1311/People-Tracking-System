import os
import json
import time
import logging
from typing import List
from pydantic import BaseModel
import cv2
import numpy as np

from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Query
from fastapi.responses import StreamingResponse, JSONResponse
from sqlmodel import Session, select
from sqlalchemy import delete

from database.session import get_session
from models.camera import Camera
from models.track import Track
from models.video_segment import VideoSegment
from models.user import UserRole
from schemas.camera import CameraCreate, CameraRead, CameraUpdate
from api.v1.deps import RoleChecker, get_current_active_user
from core.minio_client import get_minio_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Role checkers
admin_only = Depends(RoleChecker([UserRole.ADMIN]))
operator_or_admin = Depends(RoleChecker([UserRole.ADMIN, UserRole.OPERATOR]))
any_user = Depends(get_current_active_user)

@router.post("/", response_model=CameraRead)
def create_camera(
    *,
    session: Session = Depends(get_session),
    camera: CameraCreate,
    current_user=admin_only
):
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
    current_user=any_user
):
    cameras = session.exec(select(Camera).offset(offset).limit(limit)).all()
    return cameras

@router.get("/{camera_id}", response_model=CameraRead)
def read_camera(
    *,
    session: Session = Depends(get_session),
    camera_id: int,
    current_user=any_user
):
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
    current_user=operator_or_admin
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
def delete_camera(
    *,
    session: Session = Depends(get_session),
    camera_id: int,
    current_user=admin_only
):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Stop active pipelines if any
    try:
        from core.stream_manager import stream_manager
        stream_manager.stop_pipeline(camera_id)
    except Exception:
        pass
        
    # Delete related database records
    session.exec(delete(Track).where(Track.camera_id == camera_id))
    session.exec(delete(VideoSegment).where(VideoSegment.camera_id == camera_id))
    
    # Delete physical/MinIO files if present
    minio = get_minio_client()
    try:
        if camera.calibration_path:
            minio.delete_file("calibrations", camera.calibration_path)
    except Exception as e:
        logger.warning(f"Could not delete calibration from storage: {e}")
        
    session.delete(camera)
    session.commit()
    return {"ok": True}

@router.post("/{camera_id}/calibration")
async def upload_calibration(
    camera_id: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    current_user=operator_or_admin
):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    try:
        # Validate JSON content
        content = await file.read()
        json_data = json.loads(content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e}")
        
    # Upload to MinIO
    minio = get_minio_client()
    object_name = f"{camera_id}/calibration.json"
    minio.upload_file(
        bucket_name="calibrations",
        object_name=object_name,
        file_data=content,
        content_type="application/json"
    )
    
    # Update camera in db
    camera.has_calibration = True
    camera.calibration_path = object_name
    session.add(camera)
    session.commit()
    session.refresh(camera)
    
    return {"ok": True, "calibration_path": object_name}

@router.get("/{camera_id}/calibration")
def get_calibration(
    camera_id: int,
    session: Session = Depends(get_session),
    current_user=any_user
):
    camera = session.get(Camera, camera_id)
    if not camera or not camera.calibration_path:
        raise HTTPException(status_code=404, detail="Calibration file not found for this camera")
        
    minio = get_minio_client()
    import uuid
    temp_path = f"data/temp_calib_{camera_id}_{uuid.uuid4().hex}.json"
    try:
        minio.download_file("calibrations", camera.calibration_path, temp_path)
        with open(temp_path, "r") as f:
            data = json.load(f)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return data
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to read calibration: {e}")

@router.delete("/{camera_id}/calibration")
def remove_calibration(
    camera_id: int,
    session: Session = Depends(get_session),
    current_user=admin_only
):
    camera = session.get(Camera, camera_id)
    if not camera or not camera.calibration_path:
        raise HTTPException(status_code=404, detail="Calibration file not found")
        
    minio = get_minio_client()
    try:
        minio.delete_file("calibrations", camera.calibration_path)
    except Exception as e:
        logger.warning(f"Could not delete calibration from storage: {e}")
        
    camera.has_calibration = False
    camera.calibration_path = None
    session.add(camera)
    session.commit()
    session.refresh(camera)
    return {"ok": True}

@router.post("/{camera_id}/video")
async def upload_video(
    camera_id: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    current_user=operator_or_admin
):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    # Read video content
    content = await file.read()
    
    # Upload video to MinIO
    minio = get_minio_client()
    object_name = f"{camera_id}/{file.filename}"
    saved_path = minio.upload_file(
        bucket_name="videos",
        object_name=object_name,
        file_data=content,
        content_type="video/mp4"
    )
    
    # Extract resolution, fps, and a premium thumbnail
    # To use OpenCV, we need a local file. We can write to a temporary file, extract, then delete.
    temp_video_path = f"data/temp_video_{camera_id}.mp4"
    os.makedirs("data", exist_ok=True)
    with open(temp_video_path, "wb") as f:
        f.write(content)
        
    try:
        cap = cv2.VideoCapture(temp_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = f"{width}x{height}"
        
        # Read a frame at 10% progress (avoiding black frames at start)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames // 10, total_frames - 1))
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Encode frame to jpg
            _, img_encoded = cv2.imencode('.jpg', frame)
            # Upload thumbnail
            minio.upload_file(
                bucket_name="thumbnails",
                object_name=f"{camera_id}/thumb.jpg",
                file_data=img_encoded.tobytes(),
                content_type="image/jpeg"
            )
            logger.info(f"Generated thumbnail successfully for camera {camera_id}")
            
    except Exception as e:
        logger.warning(f"Could not extract video metadata or thumbnail: {e}")
        fps = 25
        resolution = "1920x1080"
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
    # Update camera record
    # Source URI is the key we use to resolve the file (we'll fetch using minio_client or use local path)
    camera.source = f"videos/{object_name}"
    camera.source_type = "video"
    camera.resolution = resolution
    camera.fps = fps
    
    session.add(camera)
    session.commit()
    session.refresh(camera)
    
    return {
        "ok": True,
        "source": camera.source,
        "resolution": resolution,
        "fps": fps
    }

@router.get("/{camera_id}/thumbnail")
def get_thumbnail(
    camera_id: int,
    session: Session = Depends(get_session),
    current_user=any_user
):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    minio = get_minio_client()
    thumbnail_url = minio.get_presigned_url("thumbnails", f"{camera_id}/thumb.jpg")
    
    # If using local storage fallback, it returns the local path like `/static/thumbnails/cam_id/thumb.jpg`
    # which is perfectly servable. In the frontend, we append base URL if needed, or rely on routing.
    return JSONResponse(content={"url": thumbnail_url})

class ConnectionCheckRequest(BaseModel):
    source: str
    source_type: str = "rtsp"  # "rtsp", "video", or "webcam"

@router.post("/check-connection")
def check_camera_connection(
    payload: ConnectionCheckRequest,
    current_user=operator_or_admin
):
    source = payload.source
    if not source:
        raise HTTPException(status_code=400, detail="Source cannot be empty")
        
    # 1. Resolve source path
    # If source is digit, convert to int (webcam)
    if source.isdigit():
        source = int(source)
    elif isinstance(source, str) and source.startswith("videos/"):
        # Resolve from MinIO or fallback directory
        minio = get_minio_client()
        if not minio.fallback_mode:
            object_name = source.replace("videos/", "", 1)
            try:
                minio.client.stat_object("videos", object_name)
            except Exception:
                return {
                    "ok": False,
                    "message": f"Video file '{object_name}' not found in MinIO bucket 'videos'."
                }
            import uuid
            temp_check_path = f"data/temp_check_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
            os.makedirs("data", exist_ok=True)
            try:
                minio.download_file("videos", object_name, temp_check_path)
                source = temp_check_path
            except Exception as e:
                return {
                    "ok": False,
                    "message": f"Failed to download video from MinIO: {e}"
                }
        else:
            source = f"data/{source}"
            
    # If local file path, verify it exists
    if isinstance(source, str) and not source.startswith("rtsp") and not os.path.exists(source):
        # Fallback to standard demo file if that is what they tested
        if source == "mct_demo.mp4" and os.path.exists("mct_demo.mp4"):
            source = "mct_demo.mp4"
        elif source == "mct_demo.mp4" and os.path.exists("../mct_demo.mp4"):
            source = "../mct_demo.mp4"
        else:
            return {
                "ok": False,
                "message": f"Local video file not found at: {source}"
            }
            
    # 2. Attempt to open stream with OpenCV
    temp_file_to_clean = None
    if isinstance(source, str) and "temp_check_" in source:
        temp_file_to_clean = source

    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return {
                "ok": False,
                "message": f"OpenCV could not open video source: {payload.source}"
            }
            
        # Try to read one frame to ensure stream works
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {
                "ok": False,
                "message": f"Could not read frames from video source: {payload.source}"
            }
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return {
            "ok": True,
            "message": "Connection check successful",
            "resolution": f"{width}x{height}",
            "fps": int(fps) if fps > 0 else 25
        }
    except Exception as e:
        return {
            "ok": False,
            "message": f"Connection check failed with exception: {e}"
        }
    finally:
        if temp_file_to_clean and os.path.exists(temp_file_to_clean):
            try:
                os.remove(temp_file_to_clean)
            except Exception:
                pass

@router.get("/{camera_id}/stream")
async def stream_camera(
    camera_id: int,
    session: Session = Depends(get_session),
    current_user=any_user
):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Resolve the video source path
    # If the source is in MinIO format like 'videos/1/video.mp4', we must download it locally
    # to feed into cv2.VideoCapture, or use the direct local fallback path if available!
    source = camera.source
    minio = get_minio_client()
    
    if source.startswith("videos/"):
        # Download from MinIO to a temporary file for streaming if running on MinIO
        if not minio.fallback_mode:
            local_cache_dir = "data/cache"
            os.makedirs(local_cache_dir, exist_ok=True)
            local_source = os.path.join(local_cache_dir, f"cam_{camera_id}_stream.mp4")
            
            # Download only if it doesn't exist or is empty
            if not os.path.exists(local_source) or os.path.getsize(local_source) == 0:
                logger.info(f"Downloading stream source for camera {camera_id} from MinIO...")
                object_name = source.replace("videos/", "", 1)
                minio.download_file("videos", object_name, local_source)
            source = local_source
        else:
            # Fallback path is local path: 'data/videos/{camera_id}/{filename}'
            source = f"data/{source}"
            
    # If it is a local path that cv2 can read, verify it exists
    if not source.startswith("rtsp") and not os.path.exists(source):
        # Check standard project directory
        if os.path.exists("mct_demo.mp4"):
            source = "mct_demo.mp4"
            
    from core.stream_manager import stream_manager
    pipeline = stream_manager.create_pipeline(camera_id, source)
    
    def frame_generator():
        try:
             for frame in pipeline.run_generator():
                 ret, buffer = cv2.imencode('.jpg', frame)
                 if ret:
                     yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"Stream error: {e}")
            
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")
