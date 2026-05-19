"""Tracking session lifecycle endpoints."""

import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from api.v1.deps import get_current_user, require_admin, require_operator
from api.v1.endpoints.configs import DEFAULT_MCT_CONFIG, DEFAULT_SCT_CONFIG
from core.session_manager import get_session_manager
from database.session import get_session
from models.camera import Camera
from models.tracking_config import TrackingConfig
from models.tracking_session import TrackingSession
from models.user import User
from schemas.session import SessionCreate, SessionRead, SessionUpdate

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=SessionRead, status_code=201)
def create_session(
    body: SessionCreate,
    session: Session = Depends(get_session),
    user: User = Depends(require_operator),
):
    """Create a new tracking session (does not start it yet)."""
    # Validate cameras exist and have calibration
    cameras = []
    for cam_id in body.camera_ids:
        cam = session.get(Camera, cam_id)
        if not cam:
            raise HTTPException(status_code=404, detail=f"Camera {cam_id} not found")
        if not cam.has_calibration:
            raise HTTPException(
                status_code=400,
                detail=f"Camera {cam_id} ('{cam.name}') has no calibration. "
                       f"Upload calibration before creating a session.",
            )
        cameras.append(cam)

    if len(cameras) < 2:
        raise HTTPException(
            status_code=400,
            detail="MCT requires at least 2 cameras",
        )

    # Resolve configs
    sct_config = DEFAULT_SCT_CONFIG
    mct_config = DEFAULT_MCT_CONFIG

    if body.sct_config_id:
        cfg = session.get(TrackingConfig, body.sct_config_id)
        if not cfg or cfg.config_type != "sct":
            raise HTTPException(status_code=404, detail="SCT config not found")
        sct_config = cfg.config_data

    if body.mct_config_id:
        cfg = session.get(TrackingConfig, body.mct_config_id)
        if not cfg or cfg.config_type != "mct":
            raise HTTPException(status_code=404, detail="MCT config not found")
        mct_config = cfg.config_data

    ts = TrackingSession(
        name=body.name,
        camera_ids=body.camera_ids,
        sct_config=sct_config,
        mct_config=mct_config,
        created_by=user.id,
    )
    session.add(ts)
    session.commit()
    session.refresh(ts)
    return ts


@router.get("/", response_model=List[SessionRead])
def list_sessions(
    *,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
    status_filter: str | None = None,
    offset: int = 0,
    limit: int = Query(default=50, le=100),
):
    query = select(TrackingSession).order_by(TrackingSession.created_at.desc())
    if status_filter:
        query = query.where(TrackingSession.status == status_filter)
    return session.exec(query.offset(offset).limit(limit)).all()


@router.get("/{session_id}", response_model=SessionRead)
def get_tracking_session(
    session_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    ts = session.get(TrackingSession, session_id)
    if not ts:
        raise HTTPException(status_code=404, detail="Session not found")
    return ts


@router.post("/{session_id}/start")
def start_session(
    session_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(require_operator),
):
    """Start the tracking pipeline for a session."""
    ts = session.get(TrackingSession, session_id)
    if not ts:
        raise HTTPException(status_code=404, detail="Session not found")

    if ts.status not in ("created", "completed", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start session in status '{ts.status}'",
        )

    mgr = get_session_manager()
    if mgr.is_running(session_id):
        raise HTTPException(status_code=400, detail="Session is already running")

    # Build camera info
    cameras_info = []
    for cam_id in ts.camera_ids:
        cam = session.get(Camera, cam_id)
        if not cam:
            raise HTTPException(status_code=404, detail=f"Camera {cam_id} not found")
        cameras_info.append({
            "id": cam.id,
            "source_uri": cam.source_uri,
            "calibration_path": cam.calibration_path,
        })

    # Reset stats
    ts.status = "running"
    ts.started_at = datetime.utcnow()
    ts.stopped_at = None
    ts.total_frames = 0
    ts.total_global_ids = 0
    ts.avg_fps = 0.0
    session.add(ts)
    session.commit()

    # Start pipeline in background
    mgr.start_session(
        session_id=ts.id,
        sct_config=ts.sct_config,
        mct_config=ts.mct_config,
        cameras=cameras_info,
    )

    return {"ok": True, "status": "running", "session_id": ts.id}


@router.post("/{session_id}/stop")
def stop_session(
    session_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(require_operator),
):
    """Stop a running tracking session."""
    ts = session.get(TrackingSession, session_id)
    if not ts:
        raise HTTPException(status_code=404, detail="Session not found")

    mgr = get_session_manager()
    if not mgr.is_running(session_id):
        raise HTTPException(status_code=400, detail="Session is not running")

    mgr.stop_session(session_id)
    return {"ok": True, "status": "stopping"}


@router.get("/{session_id}/status")
def get_session_status(
    session_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Get real-time status of a running session."""
    ts = session.get(TrackingSession, session_id)
    if not ts:
        raise HTTPException(status_code=404, detail="Session not found")

    mgr = get_session_manager()
    runtime_status = mgr.get_status(session_id)

    return {
        "session_id": session_id,
        "db_status": ts.status,
        "is_running": mgr.is_running(session_id),
        "runtime": runtime_status,
        "started_at": ts.started_at,
        "stopped_at": ts.stopped_at,
        "total_frames": ts.total_frames,
        "total_global_ids": ts.total_global_ids,
    }


@router.get("/{session_id}/output")
def get_session_output(
    session_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Get presigned URL to download session output video."""
    ts = session.get(TrackingSession, session_id)
    if not ts:
        raise HTTPException(status_code=404, detail="Session not found")

    if not ts.output_video_path:
        raise HTTPException(status_code=404, detail="No output available")

    from core.minio_client import get_minio

    minio = get_minio()
    url = minio.get_presigned_url(ts.output_video_path)
    return {"url": url, "path": ts.output_video_path}


@router.delete("/{session_id}")
def delete_session(
    session_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(require_admin),
):
    ts = session.get(TrackingSession, session_id)
    if not ts:
        raise HTTPException(status_code=404, detail="Session not found")

    mgr = get_session_manager()
    if mgr.is_running(session_id):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running session. Stop it first.",
        )

    # Clean up MinIO
    from core.minio_client import get_minio

    minio = get_minio()
    try:
        for obj in minio.list_files(prefix=f"recordings/{session_id}/"):
            minio.delete_file(obj)
    except Exception as e:
        logger.warning("Failed to clean MinIO for session %d: %s", session_id, e)

    session.delete(ts)
    session.commit()
    return {"ok": True}
