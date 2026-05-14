from datetime import datetime
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy import delete
from sqlmodel import Session, select

from database.session import engine
from models.gallery_track import GalleryTrack
from modules.data_templates.sct_template import TrackInfo, TrackState
from modules.gallery.in_mem_gallery import InMemGallery

logger = logging.getLogger(__name__)


class DbGallery(InMemGallery):
    """Database-backed gallery for persistent person tracking state."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self._load_from_db()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_from_db(self) -> None:
        try:
            with Session(engine) as session:
                rows = session.exec(select(GalleryTrack)).all()

            self.tracks.clear()
            self.map_id.clear()

            max_person_id = 0
            for row in rows:
                initial_feat = None
                if row.features:
                    initial_feat = np.array(row.features[-1], dtype=np.float32)

                track = TrackInfo(
                    tracker_id=row.tracker_id,
                    bbox=row.bbox,
                    score=row.score,
                    class_id=row.class_id,
                    feat=initial_feat,
                    cam_id=row.cam_id,
                    frame_id=row.frame_id,
                )

                if row.features:
                    track.features = [
                        np.array(f, dtype=np.float32) for f in row.features
                    ]

                track.person_id = row.person_id
                track.state = (
                    TrackState[row.state]
                    if row.state in TrackState.__members__
                    else TrackState.UNCONFIRM
                )
                track.lost_age = row.lost_age
                track.hits = row.hits

                self.tracks[row.person_id] = track
                if row.is_mapped:
                    self.map_id[row.tracker_id] = row.person_id
                max_person_id = max(max_person_id, row.person_id)

            self.next_id = max_person_id + 1 if max_person_id > 0 else 1
            logger.info("Loaded %s gallery track(s) from database", len(self.tracks))
        except Exception:
            logger.exception("Failed to load gallery state from database")

    def _persist_all_tracks(self) -> None:
        try:
            with Session(engine) as session:
                session.exec(delete(GalleryTrack))

                for person_id, track in self.tracks.items():
                    features = [
                        f.tolist() if isinstance(f, np.ndarray) else list(f)
                        for f in track.features
                    ]

                    row = GalleryTrack(
                        person_id=person_id,
                        tracker_id=int(track.tracker_id),
                        is_mapped=self.map_id.get(int(track.tracker_id)) == person_id,
                        cam_id=int(track.cam_id) if track.cam_id is not None else 0,
                        frame_id=int(track.frame_id) if track.frame_id is not None else 0,
                        bbox=list(track.bbox) if track.bbox is not None else [],
                        score=float(track.score) if track.score is not None else 0.0,
                        class_id=int(track.class_id) if track.class_id is not None else 0,
                        state=track.state.name,
                        lost_age=int(track.lost_age),
                        hits=int(track.hits),
                        features=features,
                        updated_at=datetime.now(),
                    )
                    session.add(row)

                session.commit()
        except Exception:
            logger.exception("Failed to persist gallery state to database")

    # ------------------------------------------------------------------
    # Overrides that persist after mutation
    # ------------------------------------------------------------------
    def remove_track_by_tracker_id(self, tracker_id: int):
        super().remove_track_by_tracker_id(tracker_id)
        self._persist_all_tracks()

    def mark_person_id_lost(self, person_id: int):
        super().mark_person_id_lost(person_id)
        self._persist_all_tracks()

    def remove_mapped_tracker_id(self, tracker_id: int):
        super().remove_mapped_tracker_id(tracker_id)
        self._persist_all_tracks()

    def add_or_update_unconfirmed(
        self,
        tracker_id: int,
        bbox: List[float],
        feature: np.ndarray,
        cam_id,
        frame_id: int,
        is_beautiful: bool = False,
    ) -> Optional[TrackInfo]:
        return super().add_or_update_unconfirmed(
            tracker_id, bbox, feature, cam_id, frame_id, is_beautiful
        )

    def get_lost_tracks(self) -> List[Tuple[int, TrackInfo]]:
        return super().get_lost_tracks()

    def promote_to_active(
        self, track: TrackInfo, person_id: Optional[int] = None
    ) -> int:
        pid = super().promote_to_active(track, person_id)
        self._persist_all_tracks()
        return pid

    def clean_up(self, current_tracker_ids: Set[int], current_person_ids: Set[int]):
        super().clean_up(current_tracker_ids, current_person_ids)
        self._persist_all_tracks()
