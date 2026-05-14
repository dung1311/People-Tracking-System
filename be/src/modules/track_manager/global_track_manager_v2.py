"""Lightweight global track manager for MCT Pipeline v2.

Responsibilities:
  - Re-ID cluster representatives against existing global tracks.
  - Allocate new global IDs for unmatched clusters.
  - Maintain per-global-track EMA features and lost-age bookkeeping.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from modules.matching.cross_camera_clustering import CrossCameraClusterer

logger = logging.getLogger(__name__)


class GlobalTrack:
    __slots__ = ("global_id", "feature", "last_seen_frame", "lost_age", "cam_tracks")

    def __init__(self, global_id: int, feature: np.ndarray, frame_id: int):
        self.global_id = global_id
        self.feature = feature.copy()
        self.last_seen_frame = frame_id
        self.lost_age = 0
        self.cam_tracks: Dict[int, int] = {}

    def update_feature(self, feat: np.ndarray, smooth: float = 0.1):
        self.feature = (1.0 - smooth) * self.feature + smooth * feat
        norm = np.linalg.norm(self.feature)
        if norm > 0:
            self.feature /= norm


class GlobalTrackManagerV2:
    """Manages global track pool: ReID, allocation, aging."""

    def __init__(self, config: Dict):
        self.th_reid: float = config.get("reid", 0.4)
        self.max_lost_age: int = config.get("max_lost_age", 300)
        self.feat_smooth: float = config.get("feature_smooth", 0.1)

        self.tracks: Dict[int, GlobalTrack] = {}
        self._next_gid = 1

    @property
    def num_globals(self) -> int:
        return len(self.tracks)

    def assign(
        self,
        clusters: List[List[int]],
        features: np.ndarray,
        cam_ids: np.ndarray,
        person_ids: np.ndarray,
        frame_id: int,
    ) -> Dict[Tuple[int, int], int]:
        """Given clusters from the clusterer, assign each track a global ID.

        Args:
            clusters: list of groups, each group is a list of flat indices.
            features: ``(N, D)`` L2-normed per-track features.
            cam_ids: ``(N,)`` camera ID per track.
            person_ids: ``(N,)`` local person ID per track.
            frame_id: current frame number.

        Returns:
            ``{(cam_id, local_pid): global_id}`` mapping.
        """
        K = len(clusters)
        if K == 0:
            self._age_unseen(set())
            return {}

        cluster_feats = self._compute_cluster_features(clusters, features)
        assigned = self._reid_clusters(cluster_feats)

        for k in range(K):
            if assigned[k] is None:
                assigned[k] = self._next_gid
                self._next_gid += 1

        mapping = self._update_tracks(
            clusters, assigned, cluster_feats, cam_ids, person_ids, frame_id,
        )
        return mapping

    def age_all(self):
        """Age all tracks by 1 frame and prune dead ones (call when no tracks present)."""
        self._age_unseen(set())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_cluster_features(
        self, clusters: List[List[int]], features: np.ndarray,
    ) -> np.ndarray:
        K = len(clusters)
        D = features.shape[1]
        cluster_feats = np.empty((K, D), dtype=np.float64)
        for k, idxs in enumerate(clusters):
            cluster_feats[k] = np.mean(features[idxs], axis=0)
        return CrossCameraClusterer.l2_normalise(cluster_feats)

    def _reid_clusters(self, cluster_feats: np.ndarray) -> List[Optional[int]]:
        K = len(cluster_feats)
        assigned: List[Optional[int]] = [None] * K

        if not self.tracks:
            return assigned

        gids = list(self.tracks.keys())
        gf = np.stack([self.tracks[g].feature for g in gids])
        reid_cost = CrossCameraClusterer.cosine_dist(cluster_feats, gf)

        row, col = linear_sum_assignment(reid_cost)
        for r, c in zip(row, col):
            if reid_cost[r, c] < self.th_reid:
                assigned[r] = gids[c]

        return assigned

    def _update_tracks(
        self,
        clusters: List[List[int]],
        assigned: List[Optional[int]],
        cluster_feats: np.ndarray,
        cam_ids: np.ndarray,
        person_ids: np.ndarray,
        frame_id: int,
    ) -> Dict[Tuple[int, int], int]:
        mapping: Dict[Tuple[int, int], int] = {}
        seen: set = set()

        for k, idxs in enumerate(clusters):
            gid = assigned[k]
            seen.add(gid)

            if gid in self.tracks:
                gt = self.tracks[gid]
                gt.update_feature(cluster_feats[k], self.feat_smooth)
                gt.last_seen_frame = frame_id
                gt.lost_age = 0
            else:
                gt = GlobalTrack(gid, cluster_feats[k], frame_id)
                self.tracks[gid] = gt

            gt.cam_tracks.clear()
            for idx in idxs:
                cid = int(cam_ids[idx])
                pid = int(person_ids[idx])
                gt.cam_tracks[cid] = pid
                mapping[(cid, pid)] = gid

        self._age_unseen(seen)
        return mapping

    def _age_unseen(self, seen: set):
        for gid in list(self.tracks):
            if gid not in seen:
                self.tracks[gid].lost_age += 1
                if self.tracks[gid].lost_age > self.max_lost_age:
                    del self.tracks[gid]
