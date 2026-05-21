import logging
from sqlmodel import Session, select
from database.session import engine
from models.user import User, UserRole
from models.tracking_config import TrackingConfig
from core.security import get_password_hash

logger = logging.getLogger(__name__)

def seed_db():
    with Session(engine) as session:
        # 1. Seed default Admin
        admin_exists = session.exec(select(User).where(User.role == UserRole.ADMIN)).first()
        if not admin_exists:
            logger.info("Seeding default admin user...")
            default_admin = User(
                username="admin",
                email="admin@example.com",
                role=UserRole.ADMIN,
                hashed_password=get_password_hash("adminpassword123"),
                is_active=True
            )
            session.add(default_admin)
            session.commit()
            logger.info("Default admin user 'admin' created successfully with password 'adminpassword123'")
        else:
            logger.info("Admin user already exists")

        # 2. Seed default SCT Configuration
        sct_exists = session.exec(select(TrackingConfig).where(TrackingConfig.config_type == "sct", TrackingConfig.is_default == True)).first()
        if not sct_exists:
            logger.info("Seeding default SCT configuration...")
            default_sct = TrackingConfig(
                name="Default SCT Configuration",
                description="Standard parameters for Single Camera Tracking",
                config_type="sct",
                is_default=True,
                config_data={
                    "DETECTOR": {
                        "model_path": "yolo11n.pt",
                        "conf_thres": 0.35,
                        "iou_thres": 0.45,
                        "img_size": 640
                    },
                    "TRACKER": {
                        "reid_model_path": "weights/osnet_x0_25_msmt17.pt",
                        "max_age": 30,
                        "min_hits": 3,
                        "distance_metric": "cosine",
                        "match_threshold": 0.7
                    }
                }
            )
            session.add(default_sct)
            session.commit()
            logger.info("Default SCT configuration profile seeded")

        # 3. Seed default MCT Configuration
        mct_exists = session.exec(select(TrackingConfig).where(TrackingConfig.config_type == "mct", TrackingConfig.is_default == True)).first()
        if not mct_exists:
            logger.info("Seeding default MCT configuration...")
            default_mct = TrackingConfig(
                name="Default MCT Configuration",
                description="Standard parameters for Multi Camera Tracking using Pipeline 2",
                config_type="mct",
                is_default=True,
                config_data={
                    "MATCHING": {
                        "thresholds": {
                            "spatial_thresh": 50.0,
                            "reid_thresh": 0.45,
                            "time_thresh": 30
                        }
                    },
                    "GLOBAL_TRACK": {
                        "max_lost_frames": 100,
                        "confirm_frames": 3
                    },
                    "OUTPUT": {
                        "video": "outputs/mct_output.mp4",
                        "txt_dir": "outputs/txt",
                        "fps": 25,
                        "draw_local": True
                    }
                }
            )
            session.add(default_mct)
            session.commit()
            logger.info("Default MCT configuration profile seeded")
