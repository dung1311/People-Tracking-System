import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from datetime import datetime

from database.session import get_session
from models.camera_network import CameraNetwork
from models.tracking_config import TrackingConfig
from models.camera import Camera
from models.user import UserRole
from schemas.camera_network import CameraNetworkCreate, CameraNetworkRead, CameraNetworkUpdate
from api.v1.deps import RoleChecker, get_current_active_user
from core.session_manager import get_session_manager
from core.minio_client import get_minio_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Role checkers
admin_only = Depends(RoleChecker([UserRole.ADMIN]))
operator_or_admin = Depends(RoleChecker([UserRole.ADMIN, UserRole.OPERATOR]))
any_user = Depends(get_current_active_user)

@router.post("/", response_model=CameraNetworkRead)
def create_network(
    *,
    session: Session = Depends(get_session),
    network_in: CameraNetworkCreate,
    current_user=operator_or_admin
):
    # 1. (Camera check removed, cameras are added separately)
            
    # 2. Retrieve SCT Config profile
    sct_config_data = {}
    if network_in.sct_config_id:
        sct_profile = session.get(TrackingConfig, network_in.sct_config_id)
        if sct_profile and sct_profile.config_type == "sct":
            sct_config_data = sct_profile.config_data
    else:
        # Load default SCT config
        default_sct = session.exec(
            select(TrackingConfig).where(TrackingConfig.config_type == "sct", TrackingConfig.is_default == True)
        ).first()
        if default_sct:
            sct_config_data = default_sct.config_data

    # Fallback to local configs/sct_config.yaml or hardcoded structure if missing/empty/old
    if not isinstance(sct_config_data, dict) or "DETECTION" not in sct_config_data:
        import os
        import yaml
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sct_yaml_path = os.path.join(base_dir, "configs", "sct_config.yaml")
        loaded_sct = {}
        if os.path.exists(sct_yaml_path):
            try:
                with open(sct_yaml_path, "r") as f:
                    loaded_sct = yaml.safe_load(f)
            except Exception:
                pass
        if loaded_sct and "DETECTION" in loaded_sct:
            sct_config_data = loaded_sct
        else:
            sct_config_data = {
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
            
    # 3. Retrieve MCT Config profile
    mct_config_data = {}
    if network_in.mct_config_id:
        mct_profile = session.get(TrackingConfig, network_in.mct_config_id)
        if mct_profile and mct_profile.config_type == "mct":
            mct_config_data = mct_profile.config_data
    else:
        # Load default MCT config
        default_mct = session.exec(
            select(TrackingConfig).where(TrackingConfig.config_type == "mct", TrackingConfig.is_default == True)
        ).first()
        if default_mct:
            mct_config_data = default_mct.config_data

    # Fallback to standard MCT configuration structure if missing/empty/old
    if not isinstance(mct_config_data, dict) or "MATCHING" not in mct_config_data or "weights" not in mct_config_data.get("MATCHING", {}):
        mct_config_data = {
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
            
    # 4. Create frozen Session record
    db_network = CameraNetwork(
        name=network_in.name,
        status="created",
        sct_config=sct_config_data,
        mct_config=mct_config_data,
        created_at=datetime.utcnow()
    )
    session.add(db_network)
    session.commit()
    session.refresh(db_network)
    return db_network

@router.get("/", response_model=List[CameraNetworkRead])
def read_networks(
    *,
    session: Session = Depends(get_session),
    current_user=any_user
):
    networks = session.exec(select(CameraNetwork).order_by(CameraNetwork.created_at.desc())).all()
    return networks

@router.get("/{network_id}", response_model=CameraNetworkRead)
def read_network(
    *,
    session: Session = Depends(get_session),
    network_id: int,
    current_user=any_user
):
    db_network = session.get(CameraNetwork, network_id)
    if not db_network:
        raise HTTPException(status_code=404, detail="Camera network not found")
    return db_network

@router.post("/{network_id}/start")
def start_network_tracking(
    network_id: int,
    session: Session = Depends(get_session),
    current_user=operator_or_admin
):
    db_network = session.get(CameraNetwork, network_id)
    if not db_network:
        raise HTTPException(status_code=404, detail="Camera network not found")
        
    if db_network.status in ("running", "stopping"):
        raise HTTPException(
            status_code=400,
            detail=f"Network is already in state: {db_network.status}"
        )
        
    mgr = get_session_manager()
    try:
        mgr.start_session(network_id)
        return {"ok": True, "status": "running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start background tracking: {e}")

@router.post("/{network_id}/stop")
def stop_network_tracking(
    network_id: int,
    session: Session = Depends(get_session),
    current_user=operator_or_admin
):
    db_network = session.get(CameraNetwork, network_id)
    if not db_network:
        raise HTTPException(status_code=404, detail="Camera network not found")
        
    if db_network.status != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Network is not running (current state: {db_network.status})"
        )
        
    mgr = get_session_manager()
    try:
        mgr.stop_session(network_id)
        return {"ok": True, "status": "stopping"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop background tracking: {e}")

@router.delete("/{network_id}")
def delete_network(
    *,
    session: Session = Depends(get_session),
    network_id: int,
    current_user=admin_only
):
    db_network = session.get(CameraNetwork, network_id)
    if not db_network:
        raise HTTPException(status_code=404, detail="Camera network not found")
        
    # Stop if running
    if db_network.status == "running":
        try:
            get_session_manager().stop_session(network_id)
        except Exception:
            pass
            
    # Delete from MinIO
    minio = get_minio_client()
    try:
        if db_network.output_video_path:
            minio.delete_file("recordings", db_network.output_video_path.replace("recordings/", "", 1))
    except Exception as e:
        logger.warning(f"Failed to delete session video from storage: {e}")
        
    session.delete(db_network)
    session.commit()
    return {"ok": True}

@router.get("/{network_id}/output")
def get_network_output_url(
    network_id: int,
    session: Session = Depends(get_session),
    current_user=any_user
):
    db_network = session.get(CameraNetwork, network_id)
    if not db_network:
        raise HTTPException(status_code=404, detail="Camera network not found")
        
    if not db_network.output_video_path:
        raise HTTPException(
            status_code=404, 
            detail="Output video not available for this session. It might have failed or is still running."
        )
        
    minio = get_minio_client()
    object_name = db_network.output_video_path.replace("recordings/", "", 1)
    presigned_url = minio.get_presigned_url("recordings", object_name)
    return {"url": presigned_url}
