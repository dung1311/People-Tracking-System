"""Frame extraction endpoint."""

import io
import os

import cv2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from api.v1.deps import get_current_user
from database.session import get_session
from models.camera import Camera
from models.user import User

router = APIRouter()


@router.get("/{camera_id}/{frame_id}")
def get_frame(
    camera_id: int,
    frame_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    source = camera.source_uri

    # Resolve source path for local files
    if not source.startswith("rtsp") and not os.path.isabs(source):
        for candidate in [source, f"data/videos/{source}", f"../{source}"]:
            if os.path.exists(candidate):
                source = candidate
                break

    # If source is a MinIO path, download first
    if source.startswith("videos/"):
        from core.minio_client import get_minio

        minio = get_minio()
        try:
            data = minio.get_file(source)
            tmp_path = f"/tmp/frame_extract_{camera_id}.mp4"
            with open(tmp_path, "wb") as f:
                f.write(data)
            source = tmp_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch video: {e}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video source")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=404, detail="Frame not found")

    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        raise HTTPException(status_code=500, detail="Could not encode frame")

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
