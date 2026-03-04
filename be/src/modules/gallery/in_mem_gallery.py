from typing import Dict, Optional, List, Set, Tuple
import logging

import numpy as np

from modules.data_templates.sct_template import TrackInfo, TrackState

logger = logging.getLogger(__name__)

class InMemGallery:
    """Gallery managing all tracks with Re-ID capability"""
    
    def __init__(self, config: Dict):
        """
        Args:
            max_age: Maximum frames a LOST track can survive
            min_hits: Minimum hits before confirming a track
            max_features: Maximum number of features to keep per track
        """
        self.max_age = config["max_live_time"]
        self.min_hits = config["min_hits"]
        self.max_features = config["max_features"]
        
        self.tracks: Dict[int, TrackInfo] = {} # person_id -> TrackInfo
        self.unconfirmed: Dict[int, TrackInfo] = {} # tracker_id -> TrackInfo
        self.map_id: Dict[int, int] = {} # tracker_id -> person_id
        self.next_id = 1
        
    def get_track_by_tracker_id(self, tracker_id: int) -> Optional[TrackInfo]:
        """Get track by tracker_id"""
        if tracker_id in self.map_id:
            person_id = self.map_id[tracker_id]
            return self.tracks.get(person_id)
        return None
    
    def remove_track_by_tracker_id(self, tracker_id: int):
        """Remove track by tracker_id (used for cleanup)"""
        if tracker_id in self.map_id:
            person_id = self.map_id[tracker_id]
            if person_id in self.tracks:
                del self.tracks[person_id]
            del self.map_id[tracker_id]
    
    def mark_person_id_lost(self, person_id: int):
        """Mark a track's person_id as changed (e.g., due to large appearance change)"""
        if person_id in self.tracks:
            track = self.tracks[person_id]
            track.state = TrackState.LOST
            logger.debug(f"Marked person_id={person_id} as CHANGED due to appearance change")
    
    def remove_mapped_tracker_id_and_person_id(self, tracker_id: int):
        """Remove mapping for a tracker_id and its associated person_id (used when track is lost)"""
        if tracker_id in self.map_id:
            del self.map_id[tracker_id]
    
    def get_track_by_person_id(self, person_id: int) -> Optional[TrackInfo]:
        """Get track by person_id"""
        return self.tracks.get(person_id)
    
    def add_or_update_unconfirmed(
        self, 
        tracker_id: int, 
        bbox: List[float], 
        feature: np.ndarray, 
        frame_info: Dict
    ) -> Optional[TrackInfo]:
        """
        Add or update unconfirmed track
        
        Returns:
            TrackInfo if track reaches min_hits, None otherwise
        """
        if tracker_id not in self.unconfirmed:
            # Create new unconfirmed track
            self.unconfirmed[tracker_id] = TrackInfo(
                tracker_id=tracker_id,
                bbox=bbox,
                feat=feature,
                frame_info=frame_info
            )
        else:
            # Update existing unconfirmed track
            trk = self.unconfirmed[tracker_id]
            trk.bbox = bbox
            trk.frame_info = frame_info
            trk.frame_id = frame_info["frame_id"]
            
            # Add feature with limit
            if len(trk.features) >= self.max_features:
                trk.features.pop(0)
            trk.features.append(feature)
            trk.hits += 1
        
        # Check if confirmed
        if self.unconfirmed[tracker_id].hits >= self.min_hits:
            return self.unconfirmed[tracker_id]
        
        return None
    
    def get_lost_tracks(self) -> List[Tuple[int, TrackInfo]]:
        """Get all LOST tracks for Re-ID"""
        return [
            (pid, trk) 
            for pid, trk in self.tracks.items() 
            if trk.state == TrackState.LOST
        ]
    
    def promote_to_active(self, track: TrackInfo, person_id: Optional[int] = None) -> int:
        """
        Promote unconfirmed track to active
        
        Args:
            track: Track to promote
            person_id: Existing person_id for Re-ID, or None for new person
            
        Returns:
            Assigned person_id
        """
        # Assign new person_id if not re-identified
        if person_id is None:
            person_id = self.next_id
            self.next_id += 1
        
        # Re-ID case: merge with existing track
        if person_id in self.tracks:
            old_track = self.tracks[person_id]
            
            # Remove old tracker_id mapping (if different)
            if old_track.tracker_id != track.tracker_id and old_track.tracker_id in self.map_id:
                del self.map_id[old_track.tracker_id]
            
            # Update track with new data
            old_track.tracker_id = track.tracker_id
            old_track.bbox = track.bbox
            old_track.frame_info = track.frame_info
            old_track.frame_id = track.frame_id
            
            # Merge features (keep most recent)
            if len(track.features) > 0:
                old_track.features.extend(track.features)
                if len(old_track.features) > self.max_features:
                    old_track.features = old_track.features[-self.max_features:]
            
            old_track.state = TrackState.ACTIVE
            old_track.lost_age = 0
            
            # Map new tracker_id
            self.map_id[track.tracker_id] = person_id
        else:
            # New person case
            track.person_id = person_id
            track.state = TrackState.ACTIVE
            track.lost_age = 0
            
            self.map_id[track.tracker_id] = person_id
            self.tracks[person_id] = track
        
        # Remove from unconfirmed
        if track.tracker_id in self.unconfirmed:
            del self.unconfirmed[track.tracker_id]
        
        return person_id
    
    def clean_up(self, current_tracker_ids: Set[int], current_person_ids: Set[int]):
        """
        Clean up gallery: remove dead tracks, update states
        
        Args:
            current_tracker_ids: tracker_ids detected in current frame
            current_person_ids: person_ids active in current frame
        """
        # Remove unconfirmed tracks no longer detected
        for uid in list(self.unconfirmed.keys()):
            if uid not in current_tracker_ids:
                del self.unconfirmed[uid]
        
        # Update track states
        for pid in list(self.tracks.keys()):
            track = self.tracks[pid]
            
            if pid in current_person_ids:
                # Track detected in current frame
                if track.state == TrackState.LOST:
                    # Reactivate (should not happen if logic is correct)
                    track.state = TrackState.ACTIVE
                    track.lost_age = 0
            else:
                # Track missing in current frame
                if track.state == TrackState.ACTIVE:
                    # Mark as LOST
                    track.state = TrackState.LOST
                    track.lost_age = 1
                elif track.state == TrackState.LOST:
                    # Increment lost age
                    track.lost_age += 1
                    
                    # Delete if too old
                    if track.lost_age >= self.max_age:
                        track.state = TrackState.DEAD
                        if track.tracker_id in self.map_id:
                            del self.map_id[track.tracker_id]
                        del self.tracks[pid]
                        logger.debug(f"Deleted track person_id={pid}, lost for {track.lost_age} frames")