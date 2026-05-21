import os
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from utils.load_config import load_config
from pipelines.sct_pipeline import SCTPipeline
from models.camera import Camera
from database.session import get_session
from sqlmodel import select

router = APIRouter()
logger = logging.getLogger(__name__)

# Config is now loaded dynamically per camera

@router.get("/camera/{camera_id}")
async def stream_camera(camera_id: int):
    """
    Stream video from a specific camera ID.
    This starts a new pipeline instance for the stream.
    """
    # 1. Get Camera details from DB (to get source URL)
    # Since we can't inject session easily into StreamingResponse without context managing,
    # we'll do a quick lookup.
    session_gen = get_session()
    session = next(session_gen)
    try:
        camera = session.get(Camera, camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        # Determine source. If it's an integer string, convert to int (webcam)
        video_source = camera.source
        if video_source.isdigit():
            video_source = int(video_source)
        elif isinstance(video_source, str) and video_source.startswith("videos/"):
            from core.minio_client import get_minio_client
            minio = get_minio_client()
            if not minio.fallback_mode:
                object_name = video_source.replace("videos/", "", 1)
                cached_video = f"data/temp_stream_cam_{camera_id}.mp4"
                os.makedirs("data", exist_ok=True)
                minio.download_file("videos", object_name, cached_video)
                video_source = cached_video
            else:
                video_source = f"data/{video_source}"
            
    except Exception as e:
        logger.error(f"Error fetching camera: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        session.close()

    # 2. Load Config
    # Default path if camera doesn't have one
    config_path = camera.config_path if camera and camera.config_path else "configs/sct_config.yaml"
    
    # Resolve relative path if needed
    if not os.path.exists(config_path):
       # Try relative to cwd
       config_path = os.path.join(os.getcwd(), config_path)
       
    if not os.path.exists(config_path):
        # Fallback to default absolute check
        default_path = os.path.join(os.getcwd(), "configs/sct_config.yaml")
        if os.path.exists(default_path):
             logger.warning(f"Config {config_path} not found. Using default.")
             config_path = default_path
        else:
             logger.error(f"Config file not found at {config_path}")
             raise HTTPException(status_code=500, detail=f"Configuration file not found: {config_path}")
        
    config = load_config(config_path)
    
    # 3. Initialize Pipeline
    # Note: Creating a pipeline per request is heavy. 
    # If multiple clients view the same cam, we should share the stream.
    # But for now, we follow the user's request for a pipeline.
    
    # 3. Initialize Pipeline
    input_config = {"video_path": video_source}
    pipeline = SCTPipeline(config, input_config, camera_id)
    
    # 4. Stream
    return StreamingResponse(
        pipeline.stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
