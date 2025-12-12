from dataclasses import dataclass
from typing import List, Dict
from torch import Tensor
from enum import Enum
import numpy as np

class Detection:
    def __init__(self, box: List[float], feat: Tensor):
        self.box = box
        self.feat = feat

class TrackState(Enum):
    UNCONFIRM = 0
    ACTIVE = 1
    LOST = 2
    DEAD = 3

class TrackInfo:
    def __init__(self, tracker_id, bbox, feat=None, frame_info: Dict = None):
        self.frame_info = frame_info
        self.cam_id = self.frame_info["cam_id"]
        self.tracker_id = tracker_id
        self.person_id = None
        
        self.frame_id = self.frame_info["frame_id"]
        self.bbox = bbox
        self.features = [feat] if feat is not None else []
        self.state: TrackState = TrackState.UNCONFIRM

        self.lost_age = 0
        self.hits = 1

    def update_active(self, bbox, feature, smooth_factor=0.1):
        self.bbox = bbox
        self.state = TrackState.ACTIVE
        self.lost_age = 0
        # EMA Feature update
        curr = self.get_representative_feature()
        new_feat = (1 - smooth_factor) * curr + smooth_factor * feature
        new_feat /= np.linalg.norm(new_feat)
        self.features = [new_feat]

    def get_representative_feature(self):
        if not self.features: 
            return None
        avg = np.mean(self.features, axis=0)
        return avg / np.linalg.norm(avg)

@dataclass
class MatchResult:
    match_id: int
    match_distance: float
    match_frequency: int