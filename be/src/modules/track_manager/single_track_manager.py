from typing import List, Dict, Set
import logging

import numpy as np

from modules.matching.inter_camera_matching import FeatureMatcher
from modules.gallery.in_mem_gallery import InMemGallery
from modules.embedder.factory import EmbedderFactory
from modules.data_templates.sct_template import TrackInfo, TrackState
from utils.box import crop_detections


logger = logging.getLogger(__name__)

class SingleTrackManager:
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dict with keys:
                - is_join_track: Enable Re-ID
                - max_age: Max frames for LOST track
                - min_hits: Min detections before confirmation
                - max_features: Max features per track
                - reid_threshold: Similarity threshold for Re-ID
                - smooth_factor: EMA smoothing factor for features
        """
        self.is_join_track = config.get("is_join_track", True)
        self.smooth_factor = config.get("smooth_factor", 0.1)
        
        if self.is_join_track:
            self.gallery = InMemGallery(config["GALLERY"])
            self.matcher = FeatureMatcher(config["MATCHING"])
            self.embedder = EmbedderFactory(config["EMBEDDING"]).get_embedder()
        logger.debug(f"TrackManager initialized with join_track={self.is_join_track}")
    
    def process(
        self, 
        tracks: List[List], 
        frame_info: Dict
    ) -> List[TrackInfo]:
        """
        Process tracks for current frame
        
        Args:
            tracks: List of tracks, each is [x1, y1, x2, y2, tracker_id]
            features: List of feature vectors corresponding to tracks
            frame_info: Dict with keys: frame_id, cam_id, frame (image)
            
        Returns:
            List of active TrackInfo objects with assigned person_id
        """
        # Validation
        if tracks is None or len(tracks) == 0:
            if self.is_join_track:
                self.gallery.clean_up(set(), set())
            return []
        
        # Simple mode: no Re-ID
        if not self.is_join_track:
            active_tracks =  [
                TrackInfo(
                    tracker_id=int(trk[4]),
                    bbox=trk[:4].tolist() if isinstance(trk, np.ndarray) else list(trk[:4]),
                    score=float(trk[5]) if len(trk) > 5 else 0.0,
                    class_id=int(trk[6]) if len(trk) > 6 else 0,
                    frame_info=frame_info
                ) for trk in tracks
            ]
            
            for trk in active_tracks:
                trk.person_id = trk.tracker_id
            
            return active_tracks
            
        # Re-ID mode
        need_reid_tracks: List[TrackInfo] = []
        active_tracks: List[TrackInfo] = []
        current_tracker_ids: Set[int] = set()
        current_person_ids: Set[int] = set()
        
        croped_imgs = crop_detections(frame_info['frame'], [trk[:4] for trk in tracks])
        features = self.embedder.extract_feature(croped_imgs)
        # Process each track
        for trk, feat in zip(tracks, features):
            bbox = trk[:4].tolist() if isinstance(trk, np.ndarray) else list(trk[:4])
            tracker_id = int(trk[4])
            score = float(trk[5]) if len(trk) > 5 else 0.0
            class_id = int(trk[6]) if len(trk) > 6 else 0
            
            current_tracker_ids.add(tracker_id)
            
            # Normalize feature
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            
            # Case 1: New or unconfirmed track
            if tracker_id not in self.gallery.map_id:
                confirmed_track = self.gallery.add_or_update_unconfirmed(
                    tracker_id, bbox, feat, frame_info
                )
                
                if confirmed_track is None:
                    # Still unconfirmed
                    continue
                
                # Update confirmed track info
                confirmed_track.score = score
                confirmed_track.class_id = class_id

                # Track confirmed, need Re-ID
                need_reid_tracks.append(confirmed_track)
            
            # Case 2: Existing confirmed track
            else:
                track_info = self.gallery.get_track_by_tracker_id(tracker_id)
                
                if track_info is None:
                    logger.warning(f"Track {tracker_id} in map but not found in gallery")
                    continue
                
                # Update track
                track_info.update_active(bbox, score, class_id, feat, frame_info, self.smooth_factor)
                
                current_person_ids.add(track_info.person_id)
                active_tracks.append(track_info)
        
        # Re-ID for newly confirmed tracks
        if need_reid_tracks:
            lost_tracks = self.gallery.get_lost_tracks()
            
            if lost_tracks:
                # Match against lost tracks
                match_results = self.matcher.match(
                    need_reid_tracks,
                    [trk for _, trk in lost_tracks]
                )
                
                for result in match_results:
                    if result.query_idx >= len(need_reid_tracks):
                        continue
                    
                    new_track = need_reid_tracks[result.query_idx]
                    
                    if result.is_matched and result.gallery_idx < len(lost_tracks):
                        # Re-ID successful
                        old_person_id, _ = lost_tracks[result.gallery_idx]
                        person_id = self.gallery.promote_to_active(new_track, old_person_id)
                        logger.debug(
                            f"frame: {frame_info['frame_id']}, "
                            f"Re-ID: tracker={new_track.tracker_id} -> person={person_id}, "
                            f"distance={result.distance:.3f}"
                        )
                    else:
                        # New person
                        person_id = self.gallery.promote_to_active(new_track, None)
                        logger.debug(f"frame: {frame_info['frame_id']}. New person: tracker={new_track.tracker_id} -> person={person_id}")
                    
                    current_person_ids.add(person_id)
                    promoted_track = self.gallery.tracks.get(person_id)
                    if promoted_track:
                        active_tracks.append(promoted_track)
            else:
                # No lost tracks, all are new persons
                for new_track in need_reid_tracks:
                    person_id = self.gallery.promote_to_active(new_track, None)
                    current_person_ids.add(person_id)
                    active_tracks.append(self.gallery.tracks[person_id])
        
        # Cleanup
        self.gallery.clean_up(current_tracker_ids, current_person_ids)
        
        logger.debug(
            f"Frame {frame_info['frame_id']}: "
            f"{len(active_tracks)} active, "
            f"{len(self.gallery.unconfirmed)} unconfirmed, "
            f"{len([t for t in self.gallery.tracks.values() if t.state == TrackState.LOST])} lost"
        )
        
        return active_tracks
    
    def get_statistics(self) -> Dict:
        """Get gallery statistics"""
        if not self.is_join_track:
            return {}
        
        return {
            "total_tracks": len(self.gallery.tracks),
            "active_tracks": len([t for t in self.gallery.tracks.values() if t.state == TrackState.ACTIVE]),
            "lost_tracks": len([t for t in self.gallery.tracks.values() if t.state == TrackState.LOST]),
            "unconfirmed_tracks": len(self.gallery.unconfirmed),
            "next_person_id": self.gallery.next_id
        }

