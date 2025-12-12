from typing import Dict, Union, List, Set

import numpy as np
from torch import Tensor

from ..base import BaseGallery
from modules.data_templates.sct_template import TrackInfo, TrackState

class InMemGallery(BaseGallery):
    def __init__(self, config: Dict):
        self.max_age = config["max_age"]
        self.min_hits = config["min_hits"]

        self.next_id = 1

        self.tracks: Dict[int, TrackInfo] = {} # person_id -> TrackInfo
        self.unconfirmed: Dict[int, TrackInfo] = {} # tracker_id -> TrackInfo
        self.map_id: Dict[int, int] = {} # tracker_id -> person_id

    def get_track_by_tracker_id(self, tracker_id: int) -> Union[TrackInfo]:
        if tracker_id in self.map_id:
            person_id = self.map_id[tracker_id]
            return self.tracks[person_id]
        
        return None

    def add_or_update_unconfirmed(self, tracker_id: int, bbox: Union[List[List[float]], np.ndarray], feature: Union[Tensor, np.ndarray], frame_info: Union[Dict]):
        if tracker_id not in self.unconfirmed:
            self.unconfirmed[tracker_id] = TrackInfo(
                tracker_id=tracker_id,
                bbox=bbox,
                feat=feature,
                frame_info=frame_info
            )
        else:
            trk = self.unconfirmed[tracker_id]
            trk.bbox = bbox
            trk.features.append(feature)
            trk.hits += 1
            trk.frame_info = frame_info

        if self.unconfirmed[tracker_id].hits >= self.min_hits:
            return self.unconfirmed[tracker_id]
        return None

    def get_loss_tracks(self):
        return [(pid, trk) for pid, trk in self.tracks.items() if trk.state == TrackState.LOST]
    
    def promote_to_active(self, track: TrackInfo, person_id: int):
        # Assign new person_id if not re-identified
        if person_id is None:
            person_id = self.next_id
            self.next_id += 1
        
        # Re-ID case: person_id already exists, update with new tracker_id
        if person_id in self.tracks:
            old_track = self.tracks[person_id]
            # Remove old tracker_id mapping
            if old_track.tracker_id in self.map_id:
                del self.map_id[old_track.tracker_id]
            
            # Update existing track with new tracker_id and data
            old_track.tracker_id = track.tracker_id
            old_track.bbox = track.bbox
            old_track.features.extend(track.features)
            old_track.state = TrackState.ACTIVE
            old_track.lost_age = 0
            
            # Map new tracker_id to existing person_id
            self.map_id[track.tracker_id] = person_id
        else:
            # New person case
            track.person_id = person_id
            track.state = TrackState.ACTIVE
            
            self.map_id[track.tracker_id] = person_id
            self.tracks[person_id] = track

        # Remove from unconfirmed
        if track.tracker_id in self.unconfirmed:
            del self.unconfirmed[track.tracker_id]
        
        return person_id

    def clean_up(self, current_tracker_ids: Set, current_person_ids: Set):
        # Remove unconfirmed tracks that are no longer detected
        for uid in list(self.unconfirmed.keys()):
            if uid not in current_tracker_ids:
                del self.unconfirmed[uid]
                
        # Update track states based on person_ids
        for pid in list(self.tracks.keys()):
            track = self.tracks[pid]
            
            if pid in current_person_ids:
                # Track is present in current frame
                if track.state == TrackState.LOST:
                    # Reactivate lost track
                    track.state = TrackState.ACTIVE
                    track.lost_age = 0
            else:
                # Track is missing in current frame
                if track.state == TrackState.ACTIVE:
                    # Mark as lost
                    track.state = TrackState.LOST
                    track.lost_age = 0
                elif track.state == TrackState.LOST:
                    # Increment lost age
                    track.lost_age += 1
                    if track.lost_age > self.max_age:
                        # Delete dead track and its mapping
                        if track.tracker_id in self.map_id:
                            del self.map_id[track.tracker_id]
                        del self.tracks[pid]