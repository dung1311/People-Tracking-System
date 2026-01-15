from dataclasses import dataclass
from typing import List, Dict
from torch import Tensor
from enum import Enum
import numpy as np


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
            new_feat = (1.0 - smooth_factor) * curr + smooth_factor * feature

        # Normalize
        norm = np.linalg.norm(new_feat)
        if norm > 0:
            new_feat = new_feat / norm

        # Keep only latest smoothed feature
        self.features = [new_feat]

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
