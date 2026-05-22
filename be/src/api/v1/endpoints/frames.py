import cv2
import os
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import Session
import io
import threading

from database.session import get_session
from models.camera import Camera

router = APIRouter()
_download_lock = threading.Lock()

@router.get("/{camera_id}/{frame_id}")
def get_frame(
    camera_id: int,
    frame_id: int,
    session: Session = Depends(get_session)
):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Resolve source path
    # Assuming source is relative to project root if not absolute
    # be/src is current dir when running main.py usually?
    # actually main.py is in be/src.
    # mct_demo.mp4 is in be/.
    # So path should be ../mct_demo.mp4 if running from src?
    # Let's try to handle both absolute and relative.
    
    source = camera.source
    
    # Resolve from MinIO if stored in MinIO
    if isinstance(source, str) and source.startswith("videos/"):
        from core.minio_client import get_minio_client
        minio = get_minio_client()
        if not minio.fallback_mode:
            object_name = source.replace("videos/", "", 1)
            cached_video = f"data/temp_frames_cam_{camera_id}.mp4"
            if not os.path.exists(cached_video) or os.path.getsize(cached_video) == 0:
                with _download_lock:
                    if not os.path.exists(cached_video) or os.path.getsize(cached_video) == 0:
                        os.makedirs("data", exist_ok=True)
                        minio.download_file("videos", object_name, cached_video)
            source = cached_video
        else:
            source = f"data/{source}"
    
    # Quick fix for the demo file context
    if source == "mct_demo.mp4":
        # Check if file exists in current cwd or parent
        if not os.path.exists(source):
            if os.path.exists(f"../{source}"):
                source = f"../{source}"
    
    if not os.path.exists(source) and not source.startswith("rtsp"):
         # Try looking in data/videos
         if os.path.exists(f"../data/videos/{os.path.basename(source)}"):
             source = f"../data/videos/{os.path.basename(source)}"
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video source")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=404, detail="Frame not found")
        
    # Encode to JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        raise HTTPException(status_code=500, detail="Could not encode frame")
        
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
