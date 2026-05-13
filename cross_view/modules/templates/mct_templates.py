import torch
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass

class STrackInfo:
    def __init__(self, cam_id, track_id, frame_id, timestamp):
        self.cam_id = cam_id
        self.track_id = track_id
        self.global_id = -1
        
        self.start_frame = frame_id
        self.end_frame = frame_id
        
        self.start_time = timestamp
        self.end_time = timestamp
        self.update_time = 0
        
        self.bboxes = []
        self.poses = []
        self.embeddings = torch.empty((0, 512))
        
        self.is_inited = False
        self.is_dead = False

class Pose3DResult:
    def __init__(self, pose_3d, mask, pose_2ds):
        self.pose_3d = pose_3d
        self.mask = mask
        self.pose_2ds: Dict[int, Pose2DResult] = pose_2ds
    
class Pose2DResult:
    def __init__(self, cam_id: int, kpts, embed, box, box_id, iou):
        self.cam_id = cam_id
        self.kpts = kpts
        self.embed = embed.reshape((-1, 512))
        self.box = box
        self.box_id = box_id
        self.iou = iou

class GTrackInfo:
    def __init__(self, track_id, frame_id, timestamp):
        self.track_id = track_id
        
        self.start_frame = frame_id
        self.end_frame = frame_id
        
        self.start_time = timestamp
        self.end_time = timestamp
        self.update_time = 0
        
        # self.bboxes[cam_id] = List[bboxes]
        self.bboxes: Dict[int, List] = defaultdict(list)
        self.pose_2ds: Dict[int, List] = defaultdict(list)
        self.embeddings = torch.empty((0, 512))
        self.pose_3ds: List = [Pose3DResult]
        
        self.is_inited = False
        self.is_dead = False
        
        # local_tracks[cam_id] = local_track_id 
        self.local_tracks: Dict[int, List] = defaultdict(list)

@dataclass
class MatchResult:
    match_id: int
    match_distance: float
    match_frequency: float