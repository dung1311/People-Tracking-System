from typing import Dict, Optional, List, Set, Tuple
import logging

import numpy as np

from modules.data_templates.sct_template import TrackInfo, TrackState

logger = logging.getLogger(__name__)


class InMemGallery:
    """In-memory gallery that manages confirmed / unconfirmed / lost tracks."""

    def __init__(self, config: Dict):
        self.max_age: int = config["max_live_time"]
        self.min_hits: int = config["min_hits"]
        self.max_features: int = config["max_features"]

        self.tracks: Dict[int, TrackInfo] = {}        # person_id  -> TrackInfo
        self.unconfirmed: Dict[int, TrackInfo] = {}    # tracker_id -> TrackInfo
        self.map_id: Dict[int, int] = {}               # tracker_id -> person_id
        self.next_id = 1

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def get_track_by_tracker_id(self, tracker_id: int) -> Optional[TrackInfo]:
        person_id = self.map_id.get(tracker_id)
        if person_id is not None:
            return self.tracks.get(person_id)
        return None

    def get_track_by_person_id(self, person_id: int) -> Optional[TrackInfo]:
        return self.tracks.get(person_id)

    def get_lost_tracks(self) -> List[Tuple[int, TrackInfo]]:
        return [
            (pid, trk)
            for pid, trk in self.tracks.items()
            if trk.state == TrackState.LOST
        ]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def remove_track_by_tracker_id(self, tracker_id: int):
        person_id = self.map_id.pop(tracker_id, None)
        if person_id is not None:
            self.tracks.pop(person_id, None)

    def mark_person_id_lost(self, person_id: int):
        track = self.tracks.get(person_id)
        if track is not None:
            track.state = TrackState.LOST
            logger.debug("Marked person_id=%s as LOST", person_id)

    def remove_mapped_tracker_id(self, tracker_id: int):
        self.map_id.pop(tracker_id, None)

    def add_or_update_unconfirmed(
        self,
        tracker_id: int,
        bbox: List[float],
        feature: np.ndarray,
        cam_id,
        frame_id: int,
    ) -> Optional[TrackInfo]:
        """Add or update an unconfirmed track.

        Returns the TrackInfo once it accumulates *min_hits* observations,
        otherwise returns None.
        """
        if tracker_id not in self.unconfirmed:
            self.unconfirmed[tracker_id] = TrackInfo(
                tracker_id=tracker_id,
                bbox=bbox,
                feat=feature,
                cam_id=cam_id,
                frame_id=frame_id,
            )
        else:
            trk = self.unconfirmed[tracker_id]
            trk.bbox = bbox
            trk.cam_id = cam_id
            trk.frame_id = frame_id
            if len(trk.features) >= self.max_features:
                trk.features.pop(0)
            trk.features.append(feature)
            trk.hits += 1

        if self.unconfirmed[tracker_id].hits >= self.min_hits:
            return self.unconfirmed[tracker_id]
        return None

    def promote_to_active(
        self, track: TrackInfo, person_id: Optional[int] = None
    ) -> int:
        """Promote a track to ACTIVE with a given (re-identified) or new person_id."""
        if person_id is None:
            person_id = self.next_id
            self.next_id += 1

        if person_id in self.tracks:
            old = self.tracks[person_id]
            if old.tracker_id != track.tracker_id and old.tracker_id in self.map_id:
                del self.map_id[old.tracker_id]

            old.tracker_id = track.tracker_id
            old.bbox = track.bbox
            old.cam_id = track.cam_id
            old.frame_id = track.frame_id

            if track.features:
                old.features.extend(track.features)
                if len(old.features) > self.max_features:
                    old.features = old.features[-self.max_features:]

            old.state = TrackState.ACTIVE
            old.lost_age = 0
        else:
            track.person_id = person_id
            track.state = TrackState.ACTIVE
            track.lost_age = 0
            self.tracks[person_id] = track

        self.map_id[track.tracker_id] = person_id
        self.unconfirmed.pop(track.tracker_id, None)
        return person_id

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def clean_up(self, current_tracker_ids: Set[int], current_person_ids: Set[int]):
        """Remove vanished unconfirmed tracks, age lost tracks, delete dead ones."""
        # Remove unconfirmed tracks that are no longer detected
        for uid in list(self.unconfirmed):
            if uid not in current_tracker_ids:
                del self.unconfirmed[uid]

        for pid in list(self.tracks):
            track = self.tracks[pid]

            if pid in current_person_ids:
                if track.state == TrackState.LOST:
                    track.state = TrackState.ACTIVE
                    track.lost_age = 0
            else:
                if track.state == TrackState.ACTIVE:
                    track.state = TrackState.LOST
                    track.lost_age = 1
                elif track.state == TrackState.LOST:
                    track.lost_age += 1
                    if track.lost_age >= self.max_age:
                        track.state = TrackState.DEAD
                        self.map_id.pop(track.tracker_id, None)
                        del self.tracks[pid]
                        logger.debug(
                            "Removed dead track person_id=%s (lost for %s frames)",
                            pid,
                            track.lost_age,
                        )
