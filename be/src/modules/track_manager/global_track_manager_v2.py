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
        log_file=None,
        cluster_homo_dists: List[float] = None,
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
        assigned, distances, best_gids = self._reid_clusters(cluster_feats)

        for k in range(K):
            dist = distances[k]
            old_gid = assigned[k]
            best_gid = best_gids[k]
            
            size = len(clusters[k])
            cam_lids = [f"C{int(cam_ids[idx])}:L{int(person_ids[idx])}" for idx in clusters[k]]
            cluster_info = f" | Size: {size} ({', '.join(cam_lids)})"
            
            if cluster_homo_dists and len(cluster_homo_dists) > k:
                cluster_info += f" | Max Homo Dist: {cluster_homo_dists[k]:.3f}m"
            
            if old_gid is None:
                if len(clusters[k]) >= 2:
                    assigned[k] = self._next_gid
                    self._next_gid += 1
                    if log_file:
                        log_file.write(f"Frame {frame_id} | PRIMARY | Cluster {k}{cluster_info} | Min Dist to GID {best_gid}: {dist if dist is not None else 'N/A'} | Status: NEW_CREATED | Assigned GID: {assigned[k]}\n")
                else:
                    if log_file:
                        log_file.write(f"Frame {frame_id} | PRIMARY | Cluster {k}{cluster_info} | Min Dist to GID {best_gid}: {dist if dist is not None else 'N/A'} | Status: REJECTED_SIZE | Assigned GID: None\n")
            else:
                if log_file:
                    log_file.write(f"Frame {frame_id} | PRIMARY | Cluster {k}{cluster_info} | Min Dist to GID {best_gid}: {dist:.4f} | Status: REID_MATCHED | Assigned GID: {old_gid}\n")

        mapping = self._update_tracks(
            clusters, assigned, cluster_feats, cam_ids, person_ids, frame_id,
        )
        return mapping

    def match_only(self, features: np.ndarray, threshold: float = None, ignore_gids: set = None) -> Tuple[List[Optional[int]], List[Optional[float]], List[Optional[int]]]:
        """Match features against existing global tracks using ReID only.
        
        Args:
            features: ``(N, D)`` L2-normed per-track features.
            threshold: Optional custom ReID threshold. Defaults to self.th_reid.
            ignore_gids: Optional set of global IDs to ignore during matching.
            
        Returns:
            A tuple containing:
            - A list of assigned Global IDs or None for each feature.
            - A list of best matching distances (float) or None.
            - A list of the Global IDs that gave the best distance.
        """
        K = len(features)
        assigned: List[Optional[int]] = [None] * K
        distances: List[Optional[float]] = [None] * K
        best_gids: List[Optional[int]] = [None] * K

        if not self.tracks:
            return assigned, distances, best_gids

        gids = list(self.tracks.keys())
        gf = np.stack([self.tracks[g].feature for g in gids])
        reid_cost = CrossCameraClusterer.cosine_dist(features, gf)
        
        if ignore_gids:
            for j, gid in enumerate(gids):
                if gid in ignore_gids:
                    reid_cost[:, j] = float('inf')

        thresh = threshold if threshold is not None else self.th_reid

        # Iterate over each feature to find the best match
        for i in range(K):
            min_cost_idx = np.argmin(reid_cost[i])
            min_cost = float(reid_cost[i, min_cost_idx])
            if min_cost == float('inf'):
                continue
            distances[i] = min_cost
            best_gids[i] = gids[min_cost_idx]
            if min_cost < thresh:
                assigned[i] = gids[min_cost_idx]

        return assigned, distances, best_gids

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

    def _reid_clusters(self, cluster_feats: np.ndarray) -> Tuple[List[Optional[int]], List[Optional[float]], List[Optional[int]]]:
        K = len(cluster_feats)
        assigned: List[Optional[int]] = [None] * K
        distances: List[Optional[float]] = [None] * K
        best_gids: List[Optional[int]] = [None] * K

        if not self.tracks:
            return assigned, distances, best_gids

        gids = list(self.tracks.keys())
        gf = np.stack([self.tracks[g].feature for g in gids])
        reid_cost = CrossCameraClusterer.cosine_dist(cluster_feats, gf)

        # Populate minimum distances for all clusters for logging purposes
        # (even those that will be left out by Hungarian assignment)
        for r in range(K):
            min_cost_idx = np.argmin(reid_cost[r])
            distances[r] = float(reid_cost[r, min_cost_idx])
            best_gids[r] = gids[min_cost_idx]

        row, col = linear_sum_assignment(reid_cost)
        for r, c in zip(row, col):
            dist = float(reid_cost[r, c])
            distances[r] = dist
            best_gids[r] = gids[c]
            if dist < self.th_reid:
                assigned[r] = gids[c]

        return assigned, distances, best_gids

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
            if gid is None:
                continue

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
