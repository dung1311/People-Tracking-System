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
        import os
        admin_username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
        admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "adminpassword123")
        admin_email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com")

        admin_exists = session.exec(select(User).where(User.username == admin_username)).first()
        if not admin_exists:
            logger.info(f"Seeding default admin user '{admin_username}'...")
            default_admin = User(
                username=admin_username,
                email=admin_email,
                role=UserRole.ADMIN,
                hashed_password=get_password_hash(admin_password),
                is_active=True
            )
            session.add(default_admin)
            session.commit()
            logger.info(f"Default admin user '{admin_username}' created successfully")
        else:
            logger.info(f"Admin user '{admin_username}' already exists")

        # 2. Seed default SCT Configuration
        sct_exists = session.exec(select(TrackingConfig).where(TrackingConfig.config_type == "sct", TrackingConfig.is_default == True)).first()
        
        import os
        import yaml
        
        # Auto-detect CUDA GPU availability
        device_str = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device_str = "cuda"
                logger.info("CUDA GPU detected! Defaulting seeded/migrated configurations to 'cuda' device.")
            else:
                logger.info("CUDA GPU not available. Seeding/migrating using 'cpu' device.")
        except Exception as e:
            logger.warning(f"Could not check CUDA availability via PyTorch: {e}. Defaulting to 'cpu'.")

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sct_yaml_path = os.path.join(base_dir, "configs", "sct_config.yaml")
        correct_sct_data = {}
        if os.path.exists(sct_yaml_path):
            try:
                with open(sct_yaml_path, "r") as f:
                    correct_sct_data = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading sct_config.yaml: {e}")
        
        # Enforce device selection in loaded config
        if isinstance(correct_sct_data, dict):
            if "DETECTION" in correct_sct_data and "yolov11" in correct_sct_data["DETECTION"]:
                correct_sct_data["DETECTION"]["yolov11"]["device"] = device_str
            if "POSE_ESTIMATION" in correct_sct_data and "rtmpose" in correct_sct_data["POSE_ESTIMATION"]:
                correct_sct_data["POSE_ESTIMATION"]["rtmpose"]["device"] = device_str

        if not correct_sct_data or "DETECTION" not in correct_sct_data:
            logger.info("Using hardcoded fallback for default SCT configuration structure.")
            correct_sct_data = {
                "DETECTION": {
                    "name": "yolov11",
                    "yolov11": {
                        "model_path": "weights/yolo11x.pt",
                        "task": "detect",
                        "imgsz": 640,
                        "conf_thres": 0.5,
                        "iou_thres": 0.7,
                        "device": "cpu",
                        "classes": 0,
                        "max_det": 300
                    }
                },
                "POSE_ESTIMATION": {
                    "name": "rtmpose",
                    "rtmpose": {
                        "device": "cpu",
                        "model_path": "weights/rtmpose-l_256x192/end2end.onnx",
                        "input_size": [192, 256]
                    }
                },
                "TRACKING": {
                    "name": "sort",
                    "sort": {
                        "max_age": 30,
                        "min_hits": 3,
                        "iou_threshold": 0.3
                    }
                },
                "TRACK_MANAGER": {
                    "is_join_track": True,
                    "smooth_factor": 0.1,
                    "appearance_threshold": 0.5,
                    "GALLERY": {
                        "max_live_time": 24000,
                        "min_hits": 3,
                        "max_features": 50
                    },
                    "MATCHING": {
                        "distance_threshold": 0.5
                    },
                    "EMBEDDING": {
                        "name": "fastreid",
                        "fastreid": "configs/osnet-ain_x1.0-ibn_512_256x192_ccdmmps.yaml"
                    }
                }
            }

        if not sct_exists:
            logger.info("Seeding default SCT configuration...")
            default_sct = TrackingConfig(
                name="Default SCT Configuration",
                description="Standard parameters for Single Camera Tracking",
                config_type="sct",
                is_default=True,
                config_data=correct_sct_data
            )
            session.add(default_sct)
            session.commit()
            logger.info("Default SCT configuration profile seeded")
        else:
            # Check if existing config has old DETECTOR structure or outdated device config, if so migrate/update it
            cfg_data = sct_exists.config_data
            needs_update = False
            if not isinstance(cfg_data, dict) or "DETECTION" not in cfg_data:
                logger.info("Updating existing default SCT configuration with new structure...")
                cfg_data = correct_sct_data
                needs_update = True
            else:
                # Ensure device configurations are updated to match active GPU status
                if "DETECTION" in cfg_data and "yolov11" in cfg_data["DETECTION"]:
                    if cfg_data["DETECTION"]["yolov11"].get("device") != device_str:
                        logger.info(f"Updating YOLOv11 device from {cfg_data['DETECTION']['yolov11'].get('device')} to {device_str}")
                        # Force SQLAlchemy/SQLModel change tracking by copying dictionaries
                        new_cfg = dict(cfg_data)
                        new_cfg["DETECTION"] = dict(new_cfg["DETECTION"])
                        new_cfg["DETECTION"]["yolov11"] = dict(new_cfg["DETECTION"]["yolov11"])
                        new_cfg["DETECTION"]["yolov11"]["device"] = device_str
                        cfg_data = new_cfg
                        needs_update = True
                if "POSE_ESTIMATION" in cfg_data and "rtmpose" in cfg_data["POSE_ESTIMATION"]:
                    if cfg_data["POSE_ESTIMATION"]["rtmpose"].get("device") != device_str:
                        logger.info(f"Updating RTMPose device from {cfg_data['POSE_ESTIMATION']['rtmpose'].get('device')} to {device_str}")
                        new_cfg = dict(cfg_data)
                        new_cfg["POSE_ESTIMATION"] = dict(new_cfg["POSE_ESTIMATION"])
                        new_cfg["POSE_ESTIMATION"]["rtmpose"] = dict(new_cfg["POSE_ESTIMATION"]["rtmpose"])
                        new_cfg["POSE_ESTIMATION"]["rtmpose"]["device"] = device_str
                        cfg_data = new_cfg
                        needs_update = True

            if needs_update:
                sct_exists.config_data = cfg_data
                session.add(sct_exists)
                session.commit()
                logger.info("Existing default SCT configuration successfully updated/migrated")

        # 3. Seed default MCT Configuration
        mct_config_item = session.exec(select(TrackingConfig).where(TrackingConfig.config_type == "mct", TrackingConfig.is_default == True)).first()
        
        correct_mct_data = {
            "MATCHING": {
                "weights": {
                    "homography": 0.5,
                    "visual": 0.5
                },
                "thresholds": {
                    "homography": 10.0,
                    "visual_gate": 0.5,
                    "homo_gate": 10.0,
                    "combined": 0.6,
                    "reid": 0.4
                }
            },
            "GLOBAL_TRACK": {
                "max_lost_age": 300,
                "feature_smooth": 0.1
            },
            "OUTPUT": {
                "video": "outputs/mct_output.mp4",
                "txt_dir": "outputs/txt",
                "fps": 25,
                "draw_local": False
            }
        }

        if not mct_config_item:
            logger.info("Seeding default MCT configuration...")
            default_mct = TrackingConfig(
                name="Default MCT Configuration",
                description="Standard parameters for Multi Camera Tracking using Pipeline 2",
                config_type="mct",
                is_default=True,
                config_data=correct_mct_data
            )
            session.add(default_mct)
            session.commit()
            logger.info("Default MCT configuration profile seeded")
        else:
            # Check if existing config has old MATCHING structure, if so migrate it
            cfg_data = mct_config_item.config_data
            if not isinstance(cfg_data, dict) or "MATCHING" not in cfg_data or "weights" not in cfg_data["MATCHING"]:
                logger.info("Updating existing default MCT configuration with new weights/thresholds structure...")
                # Merge existing draw_local choice if any
                draw_local = cfg_data.get("OUTPUT", {}).get("draw_local", False) if isinstance(cfg_data, dict) else False
                correct_mct_data["OUTPUT"]["draw_local"] = draw_local
                
                mct_config_item.config_data = correct_mct_data
                session.add(mct_config_item)
                session.commit()
                logger.info("Existing default MCT configuration successfully migrated")

        # 4. Migrate existing CameraNetworks if they have old SCT/MCT configurations
        from models.camera_network import CameraNetwork
        networks = session.exec(select(CameraNetwork)).all()
        networks_migrated = False
        for net in networks:
            net_changed = False
            # Check SCT config
            if not isinstance(net.sct_config, dict) or "DETECTION" not in net.sct_config:
                logger.info(f"Migrating sct_config for CameraNetwork ID {net.id} ({net.name})...")
                net.sct_config = correct_sct_data
                net_changed = True
            else:
                # Ensure device configurations in existing CameraNetwork SCT configs are updated to match active GPU status
                sct_updated = False
                if "DETECTION" in net.sct_config and "yolov11" in net.sct_config["DETECTION"]:
                    if net.sct_config["DETECTION"]["yolov11"].get("device") != device_str:
                        logger.info(f"Updating YOLOv11 device for CameraNetwork ID {net.id} to {device_str}")
                        # Force SQLModel/SQLAlchemy to detect nested JSON dictionary changes by cloning or copying
                        new_sct = dict(net.sct_config)
                        new_sct["DETECTION"] = dict(new_sct["DETECTION"])
                        new_sct["DETECTION"]["yolov11"] = dict(new_sct["DETECTION"]["yolov11"])
                        new_sct["DETECTION"]["yolov11"]["device"] = device_str
                        net.sct_config = new_sct
                        sct_updated = True
                if "POSE_ESTIMATION" in net.sct_config and "rtmpose" in net.sct_config["POSE_ESTIMATION"]:
                    if net.sct_config["POSE_ESTIMATION"]["rtmpose"].get("device") != device_str:
                        logger.info(f"Updating RTMPose device for CameraNetwork ID {net.id} to {device_str}")
                        new_sct = dict(net.sct_config)
                        new_sct["POSE_ESTIMATION"] = dict(new_sct["POSE_ESTIMATION"])
                        new_sct["POSE_ESTIMATION"]["rtmpose"] = dict(new_sct["POSE_ESTIMATION"]["rtmpose"])
                        new_sct["POSE_ESTIMATION"]["rtmpose"]["device"] = device_str
                        net.sct_config = new_sct
                        sct_updated = True
                if sct_updated:
                    net_changed = True
            
            # Check MCT config
            if not isinstance(net.mct_config, dict) or "MATCHING" not in net.mct_config or "weights" not in net.mct_config["MATCHING"]:
                logger.info(f"Migrating mct_config for CameraNetwork ID {net.id} ({net.name})...")
                draw_local = net.mct_config.get("OUTPUT", {}).get("draw_local", False) if isinstance(net.mct_config, dict) else False
                net_mct_data = dict(correct_mct_data)
                net_mct_data["OUTPUT"]["draw_local"] = draw_local
                net.mct_config = net_mct_data
                net_changed = True
                
            if net_changed:
                session.add(net)
                networks_migrated = True
                
        if networks_migrated:
            session.commit()
            logger.info("Existing CameraNetworks successfully migrated to new configuration formats")
