import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from datetime import datetime

from database.session import get_session
from models.tracking_session import TrackingSession
from models.tracking_config import TrackingConfig
from models.camera import Camera
from models.user import UserRole
from schemas.session import TrackingSessionCreate, TrackingSessionRead, TrackingSessionUpdate
from api.v1.deps import RoleChecker, get_current_active_user
from core.session_manager import get_session_manager
from core.minio_client import get_minio_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Role checkers
admin_only = Depends(RoleChecker([UserRole.ADMIN]))
operator_or_admin = Depends(RoleChecker([UserRole.ADMIN, UserRole.OPERATOR]))
any_user = Depends(get_current_active_user)

@router.post("/", response_model=TrackingSessionRead)
def create_session(
    *,
    session: Session = Depends(get_session),
    session_in: TrackingSessionCreate,
    current_user=operator_or_admin
):
    # 1. Fetch camera records to verify they exist
    cameras = session.exec(select(Camera).where(Camera.id.in_(session_in.camera_ids))).all()
    if len(cameras) != len(session_in.camera_ids):
        raise HTTPException(
            status_code=400,
            detail="One or more specified camera IDs do not exist."
        )
        
    # Check if cameras have calibration
    for cam in cameras:
        if not cam.has_calibration:
            raise HTTPException(
                status_code=400,
                detail=f"Camera '{cam.name}' does not have calibration. Calibration is required for multicam tracking."
            )
            
    # 2. Retrieve SCT Config profile
    sct_config_data = {}
    if session_in.sct_config_id:
        sct_profile = session.get(TrackingConfig, session_in.sct_config_id)
        if sct_profile and sct_profile.config_type == "sct":
            sct_config_data = sct_profile.config_data
    else:
        # Load default SCT config
        default_sct = session.exec(
            select(TrackingConfig).where(TrackingConfig.config_type == "sct", TrackingConfig.is_default == True)
        ).first()
        if default_sct:
            sct_config_data = default_sct.config_data
            
    # 3. Retrieve MCT Config profile
    mct_config_data = {}
    if session_in.mct_config_id:
        mct_profile = session.get(TrackingConfig, session_in.mct_config_id)
        if mct_profile and mct_profile.config_type == "mct":
            mct_config_data = mct_profile.config_data
    else:
        # Load default MCT config
        default_mct = session.exec(
            select(TrackingConfig).where(TrackingConfig.config_type == "mct", TrackingConfig.is_default == True)
        ).first()
        if default_mct:
            mct_config_data = default_mct.config_data
            
    # 4. Create frozen Session record
    db_session = TrackingSession(
        name=session_in.name,
        status="created",
        sct_config=sct_config_data,
        mct_config=mct_config_data,
        camera_ids=session_in.camera_ids,
        created_at=datetime.utcnow()
    )
    session.add(db_session)
    session.commit()
    session.refresh(db_session)
    return db_session

@router.get("/", response_model=List[TrackingSessionRead])
def read_sessions(
    *,
    session: Session = Depends(get_session),
    current_user=any_user
):
    sessions = session.exec(select(TrackingSession).order_by(TrackingSession.created_at.desc())).all()
    return sessions

@router.get("/{session_id}", response_model=TrackingSessionRead)
def read_session(
    *,
    session: Session = Depends(get_session),
    session_id: int,
    current_user=any_user
):
    db_session = session.get(TrackingSession, session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail="Tracking session not found")
    return db_session

@router.post("/{session_id}/start")
def start_session_tracking(
    session_id: int,
    session: Session = Depends(get_session),
    current_user=operator_or_admin
):
    db_session = session.get(TrackingSession, session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail="Tracking session not found")
        
    if db_session.status in ("running", "stopping"):
        raise HTTPException(
            status_code=400,
            detail=f"Session is already in state: {db_session.status}"
        )
        
    mgr = get_session_manager()
    try:
        mgr.start_session(session_id)
        return {"ok": True, "status": "running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start background tracking: {e}")

@router.post("/{session_id}/stop")
def stop_session_tracking(
    session_id: int,
    session: Session = Depends(get_session),
    current_user=operator_or_admin
):
    db_session = session.get(TrackingSession, session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail="Tracking session not found")
        
    if db_session.status != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Session is not running (current state: {db_session.status})"
        )
        
    mgr = get_session_manager()
    try:
        mgr.stop_session(session_id)
        return {"ok": True, "status": "stopping"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop background tracking: {e}")

@router.delete("/{session_id}")
def delete_session(
    *,
    session: Session = Depends(get_session),
    session_id: int,
    current_user=admin_only
):
    db_session = session.get(TrackingSession, session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail="Tracking session not found")
        
    # Stop if running
    if db_session.status == "running":
        try:
            get_session_manager().stop_session(session_id)
        except Exception:
            pass
            
    # Delete from MinIO
    minio = get_minio_client()
    try:
        if db_session.output_video_path:
            minio.delete_file("recordings", db_session.output_video_path.replace("recordings/", "", 1))
    except Exception as e:
        logger.warning(f"Failed to delete session video from storage: {e}")
        
    session.delete(db_session)
    session.commit()
    return {"ok": True}

@router.get("/{session_id}/output")
def get_session_output_url(
    session_id: int,
    session: Session = Depends(get_session),
    current_user=any_user
):
    db_session = session.get(TrackingSession, session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail="Tracking session not found")
        
    if not db_session.output_video_path:
        raise HTTPException(
            status_code=404, 
            detail="Output video not available for this session. It might have failed or is still running."
        )
        
    minio = get_minio_client()
    object_name = db_session.output_video_path.replace("recordings/", "", 1)
    presigned_url = minio.get_presigned_url("recordings", object_name)
    return {"url": presigned_url}
