"""Video streaming endpoint for single camera preview."""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from api.v1.deps import get_current_user
from core.stream_manager import stream_manager
from database.session import get_session
from models.camera import Camera
from models.user import User
from utils.load_config import load_config

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/camera/{camera_id}")
async def stream_camera(
    camera_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Stream video from a specific camera via MJPEG."""
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    source = camera.source_uri
    if source.isdigit():
        source = int(source)
    elif not source.startswith("rtsp") and not os.path.isabs(source):
        for candidate in [source, f"data/videos/{source}", f"../{source}"]:
            if os.path.exists(candidate):
                source = candidate
                break

    # If MinIO path, resolve
    if isinstance(source, str) and source.startswith("videos/"):
        from core.minio_client import get_minio

        minio = get_minio()
        tmp_path = f"data/session_tmp/stream_{camera_id}.mp4"
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        try:
            data = minio.get_file(source)
            with open(tmp_path, "wb") as f:
                f.write(data)
            source = tmp_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch video: {e}")

    # Load config
    config_path = "configs/sct_config.yaml"
    if not os.path.exists(config_path):
        config_path = os.path.join(os.getcwd(), config_path)
    if not os.path.exists(config_path):
        raise HTTPException(status_code=500, detail="SCT config not found")

    config = load_config(config_path)

    from pipelines.sct_pipeline import SCTPipeline

    input_config = {"video_path": source}
    pipeline = SCTPipeline(config, input_config, camera_id)

    return StreamingResponse(
        pipeline.stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
