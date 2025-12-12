from typing import Dict, Union, List, Set
import logging

import numpy as np

from modules.embedder.base import BaseEmbedder
from modules.tracker_2D.base import BaseTracker
from utils.box import crop_detections
from modules.data_templates.sct_template import TrackInfo, MatchResult, Detection

logger = logging.getLogger(__name__)

class SingleTrackManager:
    def __init__(self, config: Dict, tracker: BaseTracker, embedder: Union[BaseEmbedder]):
        self.config = config
        self.tracker = tracker
        self.embedder = embedder
        if self.embedder:
            logger.debug("Track Manager using embedder")

        self.max_lost_time = config.get("max_age", 30)
        self.reid_threshold = config.get("reid_threshold", 0.7)

        self.lost_tracks: Dict[int, TrackInfo] = {}
        self.live_tracks: Dict[int, TrackInfo] = {}
                
    def process_frame(self, frame: np.ndarray, boxes: List[List[float]]):
        croped_dets = crop_detections(frame, boxes)
        embeddings = self.embedder.extract_feature(croped_dets)
        dets = [Detection(box, feat) for box, feat in zip(boxes, embeddings)]
        
        active_tracks = self.tracker.update(dets)
        


        