from dataclasses import dataclass
from typing import List, Dict
from torch import Tensor
from enum import Enum
import numpy as np
from utils.box import crop_detections
import os
import cv2

class Detection:
    """
    Detection container.
    Feature is converted to NumPy immediately to avoid torch/numpy mixing.
    """
    def __init__(self, box: List[float], feat: Tensor):
        self.box = box
        # Always store feature as NumPy array
        self.feat = feat.detach().cpu().numpy()


class TrackState(Enum):
    UNCONFIRM = 0
    ACTIVE = 1
    LOST = 2
    DEAD = 3
    CHANGED = 4


from datetime import datetime

class TrackInfo:
    def __init__(self, tracker_id, bbox, score=None, class_id=None, feat=None, frame_info: Dict = None):
        self.frame_info = frame_info
        self.cam_id = self.frame_info["cam_id"]
        self.tracker_id = tracker_id
        self.person_id = None

        self.frame_id = self.frame_info["frame_id"]
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.timestamp = datetime.now()

        # Features are ALWAYS NumPy arrays
        self.features = [feat] if feat is not None else []

        self.state: TrackState = TrackState.UNCONFIRM
        self.lost_age = 0
        self.hits = 1
        

    
    def update_active(self, bbox, score, class_id, feature, frame_info: Dict, smooth_factor: float = 0.1):
        """
        Update track with EMA-smoothed appearance feature.
        """
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.timestamp = datetime.now()
        self.state = TrackState.ACTIVE
        self.lost_age = 0
        self.frame_info = frame_info
        # Safety: ensure NumPy
        if isinstance(feature, Tensor):
            feature = feature.detach().cpu().numpy()

        curr = self.get_representative_feature()

        # First appearance
        if curr is None:
            new_feat = feature
        else:
            # if cosine distance is large, mark as lost and need Re-ID
            cos_dist = 1 - np.dot(curr, feature) / (np.linalg.norm(curr) * np.linalg.norm(feature) + 1e-8)
            if cos_dist > 0.5:  # Threshold can be tuned
                self.state = TrackState.CHANGED
                print(f"Track {self.person_id} appearance changed (cos_dist={cos_dist:.3f}), marking as CHANGED and needs Re-ID")
                return
            new_feat = (1.0 - smooth_factor) * curr + smooth_factor * feature

        # Normalize
        norm = np.linalg.norm(new_feat)
        if norm > 0:
            new_feat = new_feat / norm

        # Keep only latest smoothed feature
        self.features = [new_feat]
        
        self._croped_img = crop_detections(self.frame_info["frame"], [self.bbox])[0] if self.bbox is not None else None
        try:
            if self._croped_img is not None:
                pid = self.person_id if self.person_id is not None else f"tracker_{self.tracker_id}"
                out_dir = os.path.join("debug", str(pid))
                os.makedirs(out_dir, exist_ok=True)
                frame_id = self.frame_info.get("frame_id", "unknown")
                cam_id = self.frame_info.get("cam_id", "cam")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fname = f"{frame_id}.jpg"
                out_path = os.path.join(out_dir, fname)
                cv2.imwrite(out_path, self._croped_img)
        except Exception as e:
            print(f"Failed to save crop for track {self.tracker_id}: {e}")
        
    def get_representative_feature(self):
        """
        Returns L2-normalized average feature or None.
        """
        if not self.features:
            return None

        avg = np.mean(self.features, axis=0)
        norm = np.linalg.norm(avg)

        if norm == 0:
            return avg

        return avg / norm

@dataclass
class MatchResult:
    query_idx: int
    gallery_idx: int
    distance: float
    is_matched: bool
