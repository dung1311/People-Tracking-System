from typing import Dict, Union, List, Set
import logging

import numpy as np

from modules.embedder.base import BaseEmbedder
from modules.matching.inter_camera_matching import InterCameraMatching
from modules.gallery.base import BaseGallery
from modules.gallery.in_mem.in_mem_gallery import InMemGallery
from utils.box import crop_detections
from modules.data_templates.sct_template import TrackInfo, MatchResult, Detection

logger = logging.getLogger(__name__)

class SingleTrackManager:
    def __init__(self, config: Dict):
        self.is_join_track = config["is_join_track"]
        if self.is_join_track:
            self.embedder = None
            self.matching = None
            self.gallery = None
    
    def process(self, tracks, frame_info):
        if not self.is_join_track:
            return [
                TrackInfo(
                    tracker_id=trk[4],
                    bbox=trk[:4],
                    frame_info=frame_info
                ) for trk in tracks
            ]






        


        