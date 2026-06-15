import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
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
        
    # Clear cached output video if it exists so that next fetch retrieves the new one
    import os
    cache_path = f"data/cache/output_videos/network_{network_id}_output.mp4"
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except Exception as e:
            logger.warning(f"Could not remove cached output video on start: {e}")

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
        
    db_network.status = "stopping"
    session.add(db_network)
    session.commit()
    session.refresh(db_network)
    
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
        
    # Delete all dependent tracks and video segments referencing cameras in this network
    camera_ids = [cam.id for cam in db_network.cameras] if db_network.cameras else []
    if camera_ids:
        from models.track import Track
        from models.video_segment import VideoSegment
        from sqlmodel import delete
        session.exec(delete(Track).where(Track.camera_id.in_(camera_ids)))
        session.exec(delete(VideoSegment).where(VideoSegment.camera_id.in_(camera_ids)))
        
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
        
    minio = get_minio_client()
    object_name = db_network.output_video_path.replace("recordings/", "", 1)
    presigned_url = minio.get_presigned_url("recordings", object_name)
    return {"url": presigned_url}

@router.get("/{network_id}/output/file")
def get_network_output_file(
    network_id: int,
    session: Session = Depends(get_session)
):
    import os
    db_network = session.get(CameraNetwork, network_id)
    if not db_network:
        raise HTTPException(status_code=404, detail="Camera network not found")
        
    if not db_network.output_video_path:
        raise HTTPException(status_code=404, detail="Output video not available")
        
    minio = get_minio_client()
    try:
        object_name = db_network.output_video_path.replace("recordings/", "", 1)
        
        # Download to a local cache file so we can serve via FileResponse (supports Range/seek)
        cache_dir = "data/cache/output_videos"
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, f"network_{network_id}_output.mp4")
        
        # Only download if not already cached
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            minio.download_file("recordings", object_name, local_path)
        
        from fastapi.responses import FileResponse
        return FileResponse(
            path=local_path,
            media_type="video/mp4",
            filename=f"tracking_output_{network_id}.mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache",
            }
        )
    except Exception as e:
        logger.warning(f"Could not load output video for network {network_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

from pydantic import BaseModel
from models.video_segment import VideoSegment
from models.track import Track
from models.camera import Camera

class NetworkAnalyzeRoiRequest(BaseModel):
    batch_number: int

def is_point_in_polygon(x: float, y: float, polygon: list) -> bool:
    num = len(polygon)
    j = num - 1
    c = False
    for i in range(num):
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
                (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            c = not c
        j = i
    return c

@router.post("/{network_id}/analyze-roi")
def analyze_network_roi(
    network_id: int,
    payload: NetworkAnalyzeRoiRequest,
    session: Session = Depends(get_session),
    current_user=any_user
):
    db_network = session.get(CameraNetwork, network_id)
    if not db_network:
        raise HTTPException(status_code=404, detail="Camera network not found")
        
    results = {}
    
    # Process each camera in the network
    for camera in db_network.cameras:
        # Find VideoSegment for this camera and batch_number
        segment = session.exec(
            select(VideoSegment)
            .where(VideoSegment.camera_id == camera.id)
            .where(VideoSegment.batch_number == payload.batch_number)
        ).first()
        
        if not segment:
            continue
            
        # Get all tracks for this segment with a buffer to handle timezone mismatches
        from datetime import timedelta, datetime
        start_buffer = segment.start_time - timedelta(hours=12)
        end_buffer = segment.end_time + timedelta(hours=12) if segment.end_time else datetime.utcnow() + timedelta(hours=12)
        
        tracks = session.exec(
            select(Track)
            .where(Track.camera_id == segment.camera_id)
            .where(Track.timestamp >= start_buffer)
            .where(Track.timestamp <= end_buffer)
            .order_by(Track.timestamp)
        ).all()
        
        # Sort tracks manually just in case
        tracks.sort(key=lambda t: t.timestamp)
        
        # Group tracks by person
        person_tracks = {}
        for t in tracks:
            # Filter tracks to be inside the batch time range
            if segment.start_time <= t.timestamp <= (segment.end_time or datetime.utcnow()):
                if t.person_id not in person_tracks:
                    person_tracks[t.person_id] = []
                person_tracks[t.person_id].append(t)
                
        rois = camera.rois or []
        camera_roi_results = {}
        
        for roi in rois:
            roi_id = roi.get("id")
            roi_name = roi.get("name", roi_id)
            polygon = roi.get("polygon", [])
            
            if not polygon:
                continue
                
            people_metrics = []
            occupancy_map = {} # timestamp -> set of person_ids
            
            for person_id, t_list in person_tracks.items():
                inside_records = []
                for t in t_list:
                    bbox = t.bbox
                    if len(bbox) >= 4:
                        x_center = (bbox[0] + bbox[2]) / 2.0
                        y_bottom = bbox[3]
                        if is_point_in_polygon(x_center, y_bottom, polygon):
                            inside_records.append(t)
                            
                if not inside_records:
                    continue
                    
                # Compute dwell time and periods
                total_dwell_seconds = 0.0
                first_enter = inside_records[0].timestamp
                last_exit = inside_records[-1].timestamp
                
                current_segment_start = inside_records[0].timestamp
                prev_time = inside_records[0].timestamp
                
                for rec in inside_records[1:]:
                    if (rec.timestamp - prev_time).total_seconds() > 5.0:
                        total_dwell_seconds += max(1.0, (prev_time - current_segment_start).total_seconds())
                        current_segment_start = rec.timestamp
                    prev_time = rec.timestamp
                total_dwell_seconds += max(1.0, (prev_time - current_segment_start).total_seconds())
                
                people_metrics.append({
                    "person_id": person_id,
                    "dwell_time_seconds": round(total_dwell_seconds, 1),
                    "entered_at": first_enter.isoformat(),
                    "exited_at": last_exit.isoformat()
                })
                
                # Populate occupancy map
                for rec in inside_records:
                    ts_sec = rec.timestamp.replace(microsecond=0)
                    if ts_sec not in occupancy_map:
                        occupancy_map[ts_sec] = set()
                    occupancy_map[ts_sec].add(person_id)
                    
            # Format occupancy over time
            occupancy_over_time = []
            for ts, p_set in sorted(occupancy_map.items()):
                occupancy_over_time.append({
                    "timestamp": ts.isoformat(),
                    "count": len(p_set)
                })
                
            total_people = len(people_metrics)
            avg_dwell = sum(p["dwell_time_seconds"] for p in people_metrics) / total_people if total_people > 0 else 0.0
            max_occupancy = max([len(p_set) for p_set in occupancy_map.values()]) if occupancy_map else 0
            
            camera_roi_results[roi_id] = {
                "roi_id": roi_id,
                "roi_name": roi_name,
                "total_people": total_people,
                "average_dwell_time_seconds": round(avg_dwell, 1),
                "max_occupancy": max_occupancy,
                "people_metrics": people_metrics,
                "occupancy_over_time": occupancy_over_time
            }
            
        results[camera.id] = {
            "camera_id": camera.id,
            "camera_name": camera.name,
            "roi_results": camera_roi_results
        }
        
    return results
