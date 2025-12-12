from dataclasses import dataclass
from typing import List
from torch import Tensor

class Detection:
    def __init__(self, box: List[float], feat: Tensor):
        self.box = box
        self.feat = feat

class TrackInfo:
    def __init__(self, track_id, frame_id, timestamp):
        self.track_id = track_id
        self.cropped_person = []
        self.embeddings = []
        self.boxes = []
        
        self.start_frame = frame_id
        self.end_frame = frame_id
        
        self.start_time = timestamp
        self.end_time = timestamp
        
        self.is_inited = False
        self.is_dead = False
        self.update_time = 0

@dataclass
class MatchResult:
    match_id: int
    match_distance: float
    match_frequency: int