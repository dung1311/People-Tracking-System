from dataclasses import dataclass
from datetime import datetime
from typing import List
import torch

@dataclass
class STrack:
    track_id: int
    
    start_time: datetime
    end_time: datetime

    start_frame: int
    end_frame: int

    bboxes: List[List[int]]  # List of bounding boxes [x1, y1, x2, y2, score]
    embeddings: torch.Tensor = torch.empty((0, 512))  # Assuming 512-dim embeddings