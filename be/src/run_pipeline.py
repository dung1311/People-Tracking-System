import logging
from sqlmodel import Session, select

from pipelines.sct_pipeline import Pipeline
from utils.load_config import load_config
from core.log_setup import configure_logging
from database.session import init_db, engine
from models.camera import Camera

logger = logging.getLogger(__name__)

configure_logging()

def get_or_create_camera(session: Session, name: str, source: str) -> int:
    camera = session.exec(select(Camera).where(Camera.name == name)).first()
    if not camera:
        camera = Camera(name=name, source=source, description="Auto-created by pipeline")
        session.add(camera)
        session.commit()
        session.refresh(camera)
    return camera.id

if __name__ == "__main__":
    init_db()
    
    cfg = load_config("/home/dungnt/People-Tracking-System/be/configs/sct_config.yaml")
    video_path = "/home/dungnt/People-Tracking-System/be/data/videos/video_2min.mp4"
    
    with Session(engine) as session:
        camera_id = get_or_create_camera(session, "Camera 1", video_path)
    
    pipeline = Pipeline(cfg, {"video_path": video_path}, camera_id=camera_id)
    pipeline.run()