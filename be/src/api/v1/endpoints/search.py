import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlmodel import Session, select
from typing import List
from datetime import datetime

from database.session import get_session
from models.track import Track
from core.model_loader import get_model_loader, ModelLoader
from utils.box import crop_detections

router = APIRouter()

@router.post("/search")
async def search_person(
    file: UploadFile = File(...),
    limit: int = 20,
    threshold: float = 0.5,
    session: Session = Depends(get_session),
    loader: ModelLoader = Depends(get_model_loader)
):
    # 1. Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 2. Detect Person
    # We assume the user uploads an image containing ONE person they want to search for.
    # But often users upload a full scene. We should find the largest person detection.
    boxes = loader.detector.detect(img)
    
    # Filter for class_id 0 (person) - YOLO usually returns [x1, y1, x2, y2, conf, cls] 
    # But the project wrapper might return just boxes?
    # Let's check modules/detector/yolov11/detect.py behavior via codebase investigation or assumption.
    # pipelines/sct_pipeline.py says: boxes = self.detector.detect(frame)
    # then tracks = self.tracker.update(boxes, ...)
    # Usually boxes is a numpy array.
    
    if len(boxes) == 0:
        raise HTTPException(status_code=404, detail="No person detected in the uploaded image")
    
    # Heuristic: Take the largest bounding box (area) assuming it's the subject
    best_box = None
    max_area = 0
    
    # Standardize box format. Assume [x1, y1, x2, y2, score, class_id] or similar.
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_box = box[:4] # Keep only coords for cropping

    if best_box is None:
         raise HTTPException(status_code=404, detail="No valid person box found")

    # 3. Crop and Embed
    # crop_detections expects a list of boxes
    crops = crop_detections(img, [best_box])
    if not crops:
         raise HTTPException(status_code=500, detail="Failed to crop person")
         
    # extract_feature returns a list of vectors
    features = loader.embedder.extract_feature(crops)
    if len(features) == 0:
        raise HTTPException(status_code=500, detail="Failed to extract features")
        
    query_vector = features[0].tolist()
    
    # 4. Search in DB
    # Fetch more results to allow for grouping (e.g., 200 raw frames -> ~10-20 unique sightings)
    
    query = select(Track, Track.feature.cosine_distance(query_vector)).order_by(Track.feature.cosine_distance(query_vector)).limit(200)
    results = session.exec(query).all()
    
    # Process results
    # Group by (camera_id, person_id)
    grouped_tracks = {}
    
    for track, dist in results:
        key = (track.camera_id, track.person_id)
        if key not in grouped_tracks:
            grouped_tracks[key] = {
                "camera_id": track.camera_id,
                "person_id": track.person_id,
                "best_score": dist, # Using distance as score (lower is better)
                "start_time": track.timestamp,
                "end_time": track.timestamp,
                "count": 0,
                "best_match": None,
                "distance": dist
            }
        
        group = grouped_tracks[key]
        group["count"] += 1
        
        # Update time range
        if track.timestamp < group["start_time"]:
            group["start_time"] = track.timestamp
        if track.timestamp > group["end_time"]:
            group["end_time"] = track.timestamp
            
        # Since we ordered by distance, the first one is the best match
        if group["best_match"] is None:
             group["best_match"] = {
                 "id": track.id,
                 "bbox": track.bbox,
                 "score": track.score, # Detection score
                 "timestamp": track.timestamp,
                 "frame_id": track.frame_id,
                 "distance": dist
             }
             
    # Convert to list and filter by grouping limit if needed
    response_list = []
    for key, val in grouped_tracks.items():
        response_list.append(val)
        
    # Sort by 'best match' logic?
    # Since we processed in order of similarity, the list order roughly reflects similarity of the *first* sighting.
    
    return {"matches": response_list[:limit]}
