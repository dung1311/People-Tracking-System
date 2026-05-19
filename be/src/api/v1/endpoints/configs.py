"""Tracking configuration management endpoints."""

import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from api.v1.deps import get_current_user, require_admin, require_operator
from database.session import get_session
from models.tracking_config import TrackingConfig
from models.user import User
from schemas.config import ConfigCreate, ConfigRead, ConfigUpdate, ConfigValidateRequest

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Default configs (embedded) ──

DEFAULT_SCT_CONFIG = {
    "DETECTION": {
        "name": "yolov11",
        "yolov11": {
            "model_path": "weights/yolo11x.pt",
            "task": "detect",
            "imgsz": 640,
            "conf_thres": 0.5,
            "iou_thres": 0.7,
            "device": "cuda",
            "classes": 0,
            "max_det": 300,
        },
    },
    "POSE_ESTIMATION": {
        "name": "rtmpose",
        "rtmpose": {
            "device": "cuda",
            "model_path": "weights/rtmpose-l_256x192/end2end.onnx",
            "input_size": [192, 256],
        },
    },
    "TRACKING": {
        "name": "sort",
        "sort": {"max_age": 10, "min_hits": 3, "iou_threshold": 0.3},
        "ocsort": {
            "det_thresh": 0.4,
            "max_age": 30,
            "min_hits": 3,
            "iou_threshold": 0.3,
            "delta_t": 3,
            "asso_func": "iou",
            "inertia": 0.2,
            "use_byte": False,
        },
    },
    "TRACK_MANAGER": {
        "is_join_track": True,
        "smooth_factor": 0.1,
        "appearance_threshold": 0.5,
        "GALLERY": {"max_live_time": 24000, "min_hits": 3, "max_features": 50},
        "MATCHING": {"distance_threshold": 0.4},
        "EMBEDDING": {
            "name": "fastreid",
            "fastreid": "configs/osnet-ain_x1.0-ibn_512_256x192_ccdmmps.yaml",
        },
    },
}

DEFAULT_MCT_CONFIG = {
    "MATCHING": {
        "weights": {"homography": 0.5, "visual": 0.5},
        "thresholds": {
            "homography": 10.0,
            "visual_gate": 0.5,
            "homo_gate": 10.0,
            "combined": 0.6,
            "reid": 0.4,
        },
    },
    "GLOBAL_TRACK": {"max_lost_age": 300, "feature_smooth": 0.1},
    "OUTPUT": {
        "video": "outputs/mct_output.mp4",
        "txt_dir": "outputs/txt",
        "fps": 25,
        "draw_local": False,
    },
    "MCT_PERFORMANCE": {"shared_models": True, "enable_pose_full_body": True},
}


# ── CRUD ──

@router.post("/", response_model=ConfigRead, status_code=201)
def create_config(
    body: ConfigCreate,
    session: Session = Depends(get_session),
    user: User = Depends(require_operator),
):
    if body.config_type not in ("sct", "mct"):
        raise HTTPException(status_code=400, detail="config_type must be 'sct' or 'mct'")

    config = TrackingConfig(
        **body.model_dump(),
        created_by=user.id,
    )
    session.add(config)
    session.commit()
    session.refresh(config)
    return config


@router.get("/", response_model=List[ConfigRead])
def list_configs(
    *,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
    config_type: str | None = None,
    offset: int = 0,
    limit: int = Query(default=50, le=100),
):
    query = select(TrackingConfig)
    if config_type:
        query = query.where(TrackingConfig.config_type == config_type)
    return session.exec(query.offset(offset).limit(limit)).all()


@router.get("/defaults")
def get_defaults():
    """Return the built-in default SCT and MCT configs."""
    return {"sct": DEFAULT_SCT_CONFIG, "mct": DEFAULT_MCT_CONFIG}


@router.get("/{config_id}", response_model=ConfigRead)
def get_config(
    config_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    config = session.get(TrackingConfig, config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")
    return config


@router.patch("/{config_id}", response_model=ConfigRead)
def update_config(
    config_id: int,
    body: ConfigUpdate,
    session: Session = Depends(get_session),
    user: User = Depends(require_operator),
):
    config = session.get(TrackingConfig, config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")

    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(config, key, value)
    config.updated_at = datetime.utcnow()

    session.add(config)
    session.commit()
    session.refresh(config)
    return config


@router.delete("/{config_id}")
def delete_config(
    config_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(require_admin),
):
    config = session.get(TrackingConfig, config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")
    session.delete(config)
    session.commit()
    return {"ok": True}


@router.post("/{config_id}/duplicate", response_model=ConfigRead, status_code=201)
def duplicate_config(
    config_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(require_operator),
):
    """Duplicate an existing config."""
    original = session.get(TrackingConfig, config_id)
    if not original:
        raise HTTPException(status_code=404, detail="Config not found")

    new_config = TrackingConfig(
        name=f"{original.name} (copy)",
        description=original.description,
        config_type=original.config_type,
        config_data=original.config_data.copy(),
        is_default=False,
        created_by=user.id,
    )
    session.add(new_config)
    session.commit()
    session.refresh(new_config)
    return new_config


@router.post("/validate")
def validate_config(body: ConfigValidateRequest):
    """Validate a config structure without saving."""
    errors = []

    if body.config_type == "sct":
        required_sections = ["DETECTION", "TRACKING", "TRACK_MANAGER"]
        for section in required_sections:
            if section not in body.config_data:
                errors.append(f"Missing required section: {section}")

        # Validate DETECTION
        detection = body.config_data.get("DETECTION", {})
        if "name" not in detection:
            errors.append("DETECTION.name is required")

        # Validate TRACK_MANAGER
        tm = body.config_data.get("TRACK_MANAGER", {})
        if tm.get("is_join_track", True):
            for sub in ["GALLERY", "MATCHING", "EMBEDDING"]:
                if sub not in tm:
                    errors.append(f"TRACK_MANAGER.{sub} is required when is_join_track=True")

    elif body.config_type == "mct":
        required_sections = ["MATCHING", "GLOBAL_TRACK"]
        for section in required_sections:
            if section not in body.config_data:
                errors.append(f"Missing required section: {section}")
    else:
        errors.append("config_type must be 'sct' or 'mct'")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }
