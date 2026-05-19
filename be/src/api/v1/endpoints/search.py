"""Person search by uploaded image — uses detector + embedder + pgvector."""

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlmodel import Session, select
from typing import List

from api.v1.deps import get_current_user
from database.session import get_session
from models.track import Track
from models.user import User
from core.model_loader import get_model_loader, ModelLoader
from utils.box import crop_detections

router = APIRouter()


@router.post("/search")
async def search_person(
    file: UploadFile = File(...),
    limit: int = 20,
    threshold: float = 0.5,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
    loader: ModelLoader = Depends(get_model_loader),
):
    # 1. Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 2. Detect Person
    boxes = loader.detector.detect(img)

    if len(boxes) == 0:
        raise HTTPException(status_code=404, detail="No person detected in the uploaded image")

    # Heuristic: Take the largest bounding box (area) assuming it's the subject
    best_box = None
    max_area = 0

    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_box = box[:4]

    if best_box is None:
        raise HTTPException(status_code=404, detail="No valid person box found")

    # 3. Crop and Embed
    crops = crop_detections(img, [best_box])
    if not crops:
        raise HTTPException(status_code=500, detail="Failed to crop person")

    features = loader.embedder.extract_feature(crops)
    if len(features) == 0:
        raise HTTPException(status_code=500, detail="Failed to extract features")

    query_vector = features[0].tolist()

    # 4. Search in DB using pgvector cosine similarity
    query = (
        select(Track, Track.feature.cosine_distance(query_vector))
        .order_by(Track.feature.cosine_distance(query_vector))
        .limit(200)
    )
    results = session.exec(query).all()

    # Group by (camera_id, person_id)
    grouped_tracks = {}

    for track, dist in results:
        key = (track.camera_id, track.person_id)
        if key not in grouped_tracks:
            grouped_tracks[key] = {
                "camera_id": track.camera_id,
                "person_id": track.person_id,
                "best_score": dist,
                "start_time": track.timestamp,
                "end_time": track.timestamp,
                "count": 0,
                "best_match": None,
                "distance": dist,
            }

        group = grouped_tracks[key]
        group["count"] += 1

        if track.timestamp < group["start_time"]:
            group["start_time"] = track.timestamp
        if track.timestamp > group["end_time"]:
            group["end_time"] = track.timestamp

        if group["best_match"] is None:
            group["best_match"] = {
                "id": track.id,
                "bbox": track.bbox,
                "score": track.score,
                "timestamp": track.timestamp,
                "frame_id": track.frame_id,
                "distance": dist,
            }

    response_list = list(grouped_tracks.values())
    return {"matches": response_list[:limit]}
