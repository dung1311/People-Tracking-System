from typing import List, Dict, Set, Tuple, Optional
import logging

import numpy as np

from modules.matching.inter_camera_matching import FeatureMatcher
from modules.gallery.in_mem_gallery import InMemGallery
from modules.embedder.factory import EmbedderFactory
from modules.data_templates.sct_template import TrackInfo, TrackState
from utils.box import crop_detections

logger = logging.getLogger(__name__)


class SingleTrackManager:
    """Manages the lifecycle of tracks for a single camera.

    Handles confirmation, Re-ID against lost tracks, and feature bookkeeping.
    """

    def __init__(
        self,
        config: Dict,
        gallery: Optional[InMemGallery] = None,
        matcher: Optional[FeatureMatcher] = None,
        embedder=None,
    ):
        self.is_join_track: bool = config.get("is_join_track", True)
        self.smooth_factor: float = config.get("smooth_factor", 0.1)
        self.appearance_threshold: float = config.get("appearance_threshold", 0.5)

        if self.is_join_track:
            self.gallery = gallery or InMemGallery(config["GALLERY"])
            self.matcher = matcher or FeatureMatcher(config["MATCHING"])
            self.embedder = embedder or EmbedderFactory(config["EMBEDDING"]).get_embedder()

        logger.debug("TrackManager initialised (join_track=%s)", self.is_join_track)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(
        self,
        tracks: List[List],
        frame_info: Dict,
    ) -> List[TrackInfo]:
        """Process raw tracker output for the current frame.

        Args:
            tracks: rows of ``[x1, y1, x2, y2, tracker_id, ...]`` from the 2-D tracker.
            frame_info: dict with at least ``cam_id``, ``frame_id``, ``frame`` (image).

        Returns:
            Active TrackInfo objects with assigned ``person_id``.
        """
        if tracks is None or len(tracks) == 0:
            if self.is_join_track:
                self.gallery.clean_up(set(), set())
            return []

        if not self.is_join_track:
            return self._process_simple(tracks, frame_info)

        return self._process_with_reid(tracks, frame_info)

    def get_statistics(self) -> Dict:
        if not self.is_join_track:
            return {}
        return {
            "total_tracks": len(self.gallery.tracks),
            "active_tracks": sum(
                1 for t in self.gallery.tracks.values() if t.state == TrackState.ACTIVE
            ),
            "lost_tracks": sum(
                1 for t in self.gallery.tracks.values() if t.state == TrackState.LOST
            ),
            "unconfirmed_tracks": len(self.gallery.unconfirmed),
            "next_person_id": self.gallery.next_id,
        }

    # ------------------------------------------------------------------
    # Simple mode (no Re-ID)
    # ------------------------------------------------------------------
    def _process_simple(
        self, tracks: List[List], frame_info: Dict
    ) -> List[TrackInfo]:
        result = []
        for trk in tracks:
            bbox, tracker_id, score, class_id = self._parse_track(trk)
            t = TrackInfo(
                tracker_id=tracker_id,
                bbox=bbox,
                score=score,
                class_id=class_id,
                cam_id=frame_info["cam_id"],
                frame_id=frame_info["frame_id"],
            )
            t.person_id = tracker_id
            result.append(t)
        return result

    # ------------------------------------------------------------------
    # Re-ID mode
    # ------------------------------------------------------------------
    def _process_with_reid(
        self, tracks: List[List], frame_info: Dict
    ) -> List[TrackInfo]:
        features = self._extract_features(tracks, frame_info["frame"])

        active_tracks: List[TrackInfo] = []
        need_reid_tracks: List[TrackInfo] = []
        current_tracker_ids: Set[int] = set()
        current_person_ids: Set[int] = set()

        cam_id = frame_info["cam_id"]
        frame_id = frame_info["frame_id"]

        for trk, feat in zip(tracks, features):
            bbox, tracker_id, score, class_id = self._parse_track(trk)
            current_tracker_ids.add(tracker_id)
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            
            is_beautiful = frame_info.get("is_full_body", [])
            is_beautiful_for_track = is_beautiful[tracker_id] if isinstance(is_beautiful, dict) and tracker_id in is_beautiful else False

            if tracker_id in self.gallery.map_id:
                result = self._handle_confirmed_track(
                    tracker_id, bbox, score, class_id, feat, cam_id, frame_id
                )
                if result is None:
                    continue
                track_info, is_changed = result
                if is_changed:
                    need_reid_tracks.append(track_info)
                else:
                    current_person_ids.add(track_info.person_id)
                    active_tracks.append(track_info)
            else:
                confirmed = self._handle_unconfirmed_track(
                    tracker_id, bbox, score, class_id, feat, cam_id, frame_id, is_beautiful_for_track
                )
                if confirmed is not None:
                    need_reid_tracks.append(confirmed)

        if need_reid_tracks:
            for track_info, person_id in self._match_lost_tracks(need_reid_tracks):
                current_person_ids.add(person_id)
                promoted = self.gallery.tracks.get(person_id)
                if promoted is not None:
                    active_tracks.append(promoted)

        self.gallery.clean_up(current_tracker_ids, current_person_ids)

        logger.debug(
            "Frame %s: %d active, %d unconfirmed, %d lost",
            frame_id,
            len(active_tracks),
            len(self.gallery.unconfirmed),
            sum(1 for t in self.gallery.tracks.values() if t.state == TrackState.LOST),
        )
        return active_tracks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_track(trk) -> Tuple[List[float], int, float, int]:
        bbox = trk[:4].tolist() if isinstance(trk, np.ndarray) else list(trk[:4])
        tracker_id = int(trk[4])
        score = float(trk[5]) if len(trk) > 5 else 0.0
        class_id = int(trk[6]) if len(trk) > 6 else 0
        return bbox, tracker_id, score, class_id

    def _extract_features(
        self, tracks: List[List], frame: np.ndarray
    ) -> List[np.ndarray]:
        cropped_imgs = crop_detections(frame, [trk[:4] for trk in tracks])
        return self.embedder.extract_feature(cropped_imgs)

    def _handle_confirmed_track(
        self,
        tracker_id: int,
        bbox: List[float],
        score: float,
        class_id: int,
        feat: np.ndarray,
        cam_id,
        frame_id: int,
    ) -> Optional[Tuple[TrackInfo, bool]]:
        """Update an already-confirmed track.

        Returns ``(track_info, is_changed)`` or *None* if the track
        disappeared from the gallery unexpectedly.
        """
        track_info = self.gallery.get_track_by_tracker_id(tracker_id)
        if track_info is None:
            logger.warning("Tracker %s mapped but not found in gallery", tracker_id)
            return None

        track_info.update(
            bbox, score, class_id, feat, frame_id,
            smooth_factor=self.smooth_factor,
            appearance_threshold=self.appearance_threshold,
        )

        if track_info.state == TrackState.CHANGED:
            logger.info(
                "Tracker %s (person %s) appearance changed, splitting for Re-ID",
                tracker_id,
                track_info.person_id,
            )
            self.gallery.mark_person_id_lost(track_info.person_id)
            self.gallery.remove_mapped_tracker_id(tracker_id)

            new_track = TrackInfo(
                tracker_id=tracker_id,
                bbox=bbox,
                score=score,
                class_id=class_id,
                cam_id=cam_id,
                frame_id=frame_id,
            )
            new_track.features = [feat]
            return new_track, True

        return track_info, False

    def _handle_unconfirmed_track(
        self,
        tracker_id: int,
        bbox: List[float],
        score: float,
        class_id: int,
        feat: np.ndarray,
        cam_id,
        frame_id: int,
        is_beautiful: bool = False,
    ) -> Optional[TrackInfo]:
        """Buffer an unconfirmed track; return it when it reaches *min_hits*."""
        confirmed = self.gallery.add_or_update_unconfirmed(
            tracker_id, bbox, feat, cam_id, frame_id, is_beautiful
        )
        if confirmed is None:
            return None
        confirmed.score = score
        confirmed.class_id = class_id
        return confirmed

    def _match_lost_tracks(
        self, need_reid_tracks: List[TrackInfo]
    ) -> List[Tuple[TrackInfo, int]]:
        """Match newly-confirmed tracks against LOST gallery entries.

        Returns list of ``(track, assigned_person_id)`` pairs.
        """
        results: List[Tuple[TrackInfo, int]] = []
        lost_tracks = self.gallery.get_lost_tracks()

        if lost_tracks:
            match_results = self.matcher.match(
                need_reid_tracks, [trk for _, trk in lost_tracks]
            )
            for result in match_results:
                if result.query_idx >= len(need_reid_tracks):
                    continue
                new_track = need_reid_tracks[result.query_idx]

                if result.is_matched and result.gallery_idx < len(lost_tracks):
                    old_person_id, _ = lost_tracks[result.gallery_idx]
                    person_id = self.gallery.promote_to_active(new_track, old_person_id)
                    logger.info(
                        "Re-ID: tracker %s -> person %s (dist=%.3f, min_dist=%.3f)",
                        new_track.tracker_id,
                        person_id,
                        result.distance,
                        result.min_distance,
                    )
                else:
                    person_id = self.gallery.promote_to_active(new_track, None)
                    logger.info(
                        "New person: tracker %s -> person %s (min_dist=%.3f)",
                        new_track.tracker_id,
                        person_id,
                        result.min_distance,
                    )
                results.append((new_track, person_id))
        else:
            for new_track in need_reid_tracks:
                person_id = self.gallery.promote_to_active(new_track, None)
                results.append((new_track, person_id))

        return results
