"""Camera management endpoints with calibration and video upload."""

import io
import json
import logging
import os
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlmodel import Session, select

from api.v1.deps import get_current_user, require_admin, require_operator
from core.minio_client import get_minio, MinIOService
from database.session import get_session
from models.camera import Camera
from models.user import User
from schemas.camera import CameraCreate, CameraRead, CameraUpdate

router = APIRouter()
logger = logging.getLogger(__name__)


# ── CRUD ──

@router.post("/", response_model=CameraRead, status_code=201)
def create_camera(
    body: CameraCreate,
    session: Session = Depends(get_session),
    user: User = Depends(require_admin),
):
    camera = Camera(
        **body.model_dump(),
        created_by=user.id,
    )
    session.add(camera)
    session.commit()
    session.refresh(camera)
    return camera


@router.get("/", response_model=List[CameraRead])
def list_cameras(
    *,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
    offset: int = 0,
    limit: int = Query(default=100, le=100),
    active_only: bool = False,
):
    query = select(Camera)
    if active_only:
        query = query.where(Camera.is_active == True)
    cameras = session.exec(query.offset(offset).limit(limit)).all()
    return cameras


@router.get("/{camera_id}", response_model=CameraRead)
def get_camera(
    camera_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.patch("/{camera_id}", response_model=CameraRead)
def update_camera(
    camera_id: int,
    body: CameraUpdate,
    session: Session = Depends(get_session),
    user: User = Depends(require_operator),
):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(camera, key, value)
    camera.updated_at = datetime.utcnow()

    session.add(camera)
    session.commit()
    session.refresh(camera)
    return camera


@router.delete("/{camera_id}")
def delete_camera(
    camera_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(require_admin),
    minio: MinIOService = Depends(get_minio),
):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Clean up MinIO files
    try:
        for obj in minio.list_files(prefix=f"cameras/{camera_id}/"):
            minio.delete_file(obj)
    except Exception as e:
        logger.warning("Failed to clean MinIO files for camera %d: %s", camera_id, e)

    # Cascade delete related DB records
    from sqlalchemy import delete as sa_delete
    from models.track import Track
    from models.video_segment import VideoSegment

    session.exec(sa_delete(Track).where(Track.camera_id == camera_id))
    session.exec(sa_delete(VideoSegment).where(VideoSegment.camera_id == camera_id))
    session.delete(camera)
    session.commit()
    return {"ok": True}


# ── Calibration ──

@router.post("/{camera_id}/calibration", response_model=CameraRead)
async def upload_calibration(
    camera_id: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    user: User = Depends(require_operator),
    minio: MinIOService = Depends(get_minio),
):
    """Upload a camera calibration JSON file."""
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Validate JSON
    contents = await file.read()
    try:
        calib_data = json.loads(contents)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    # Validate calibration structure
    required_keys = {"camera projection matrix", "homography matrix"}
    if not required_keys.issubset(calib_data.keys()):
        raise HTTPException(
            status_code=400,
            detail=f"Calibration JSON must contain keys: {required_keys}",
        )

    # Upload to MinIO
    object_name = f"calibrations/{camera_id}/calibration.json"
    minio.upload_bytes(object_name, contents, content_type="application/json")

    # Update camera record
    camera.has_calibration = True
    camera.calibration_path = object_name
    camera.updated_at = datetime.utcnow()
    session.add(camera)
    session.commit()
    session.refresh(camera)
    return camera


@router.get("/{camera_id}/calibration")
def get_calibration(
    camera_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
    minio: MinIOService = Depends(get_minio),
):
    """Download the calibration JSON for a camera."""
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not camera.has_calibration or not camera.calibration_path:
        raise HTTPException(status_code=404, detail="No calibration uploaded")

    data = minio.get_file(camera.calibration_path)
    return json.loads(data)


@router.delete("/{camera_id}/calibration", response_model=CameraRead)
def delete_calibration(
    camera_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(require_admin),
    minio: MinIOService = Depends(get_minio),
):
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    if camera.calibration_path:
        try:
            minio.delete_file(camera.calibration_path)
        except Exception:
            pass

    camera.has_calibration = False
    camera.calibration_path = None
    camera.updated_at = datetime.utcnow()
    session.add(camera)
    session.commit()
    session.refresh(camera)
    return camera


# ── Video Upload ──

@router.post("/{camera_id}/video")
async def upload_video(
    camera_id: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    user: User = Depends(require_operator),
    minio: MinIOService = Depends(get_minio),
):
    """Upload a video file for a camera → stored in MinIO."""
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Read file
    contents = await file.read()
    object_name = f"videos/{camera_id}/{file.filename}"

    minio.upload_bytes(
        object_name,
        contents,
        content_type=file.content_type or "video/mp4",
    )

    # Update camera source to point to MinIO
    camera.source_uri = object_name
    camera.source_type = "video"
    camera.updated_at = datetime.utcnow()
    session.add(camera)
    session.commit()
    session.refresh(camera)

    return {
        "ok": True,
        "object_name": object_name,
        "size": len(contents),
        "camera": CameraRead.model_validate(camera).model_dump(),
    }
