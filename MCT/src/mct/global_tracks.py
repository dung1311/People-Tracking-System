"""Lightweight global track manager: ReID clusters, allocate GIDs, aging."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from mct.distance import cosine_distance_matrix, l2_normalise_rows

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
    """Manages global track pool: Hungarian ReID, allocation, lost aging."""

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
        gids: Optional[List[Optional[int]]] = None,
    ) -> Dict[Tuple[int, int], int]:
        K = len(clusters)
        if K == 0:
            self._age_unseen(set())
            return {}

        cluster_feats = self._compute_cluster_features(clusters, features)

        # 1. Collect static pre-existing GID assignments from the current frame's active tracks
        static_mapping: Dict[Tuple[int, int], int] = {}
        if gids is not None:
            for idx, gid in enumerate(gids):
                if gid is not None:
                    cid = int(cam_ids[idx])
                    pid = int(person_ids[idx])
                    static_mapping[(cid, pid)] = gid

        # Fallback to last frame memory if needed (for safety/backward compatibility)
        track_to_prev_gid = {}
        for gid, gt in self.tracks.items():
            for cid, pid in gt.cam_tracks.items():
                track_to_prev_gid[(cid, pid)] = gid

        # Combine static mapping with fallback for all active tracks in clusters
        locked_mapping: Dict[Tuple[int, int], int] = {}
        for k, idxs in enumerate(clusters):
            for idx in idxs:
                cid = int(cam_ids[idx])
                pid = int(person_ids[idx])
                if (cid, pid) in static_mapping:
                    locked_mapping[(cid, pid)] = static_mapping[(cid, pid)]
                elif (cid, pid) in track_to_prev_gid:
                    locked_mapping[(cid, pid)] = track_to_prev_gid[(cid, pid)]

        # Map cid -> set of gids already locked on this camera in the current frame
        locked_gids_per_cam: Dict[int, Set[int]] = {}
        # Map gid -> Tuple[int, int] identifying which track locked this gid
        gid_locked_by: Dict[int, Tuple[int, int]] = {}
        for (cid, pid), gid in locked_mapping.items():
            if cid not in locked_gids_per_cam:
                locked_gids_per_cam[cid] = set()
            locked_gids_per_cam[cid].add(gid)
            gid_locked_by[gid] = (cid, pid)

        # Permissibility check: A GID is permissible for a cluster if it's not locked on any of the cluster's cameras
        # by a different track.
        def is_gid_permissible_for_cluster(gid_val: int, idxs_list: List[int]) -> bool:
            for idx_in in idxs_list:
                cid_in = int(cam_ids[idx_in])
                pid_in = int(person_ids[idx_in])
                if gid_val in locked_gids_per_cam.get(cid_in, set()):
                    if gid_locked_by[gid_val] != (cid_in, pid_in):
                        return False
            return True

        # 2. For each cluster, determine its representative GID (if any)
        cluster_gids: List[Optional[int]] = [None] * K
        for k, idxs in enumerate(clusters):
            gids_in_cluster = []
            for idx in idxs:
                cid = int(cam_ids[idx])
                pid = int(person_ids[idx])
                if (cid, pid) in locked_mapping:
                    gids_in_cluster.append(locked_mapping[(cid, pid)])
            
            if gids_in_cluster:
                # Sort unique candidates by frequency descending
                candidates = sorted(set(gids_in_cluster), key=lambda g: gids_in_cluster.count(g), reverse=True)
                for cand in candidates:
                    if is_gid_permissible_for_cluster(cand, idxs):
                        cluster_gids[k] = cand
                        break

        # 3. Match remaining clusters via Hungarian ReID against unused GIDs
        unassigned_cluster_indices = [k for k in range(K) if cluster_gids[k] is None]
        used_gids = set(g for g in cluster_gids if g is not None)
        available_gids = [g for g in self.tracks.keys() if g not in used_gids]

        if unassigned_cluster_indices and available_gids:
            sub_cluster_feats = cluster_feats[unassigned_cluster_indices]
            gf = np.stack([self.tracks[g].feature for g in available_gids])
            reid_cost = cosine_distance_matrix(sub_cluster_feats, gf)

            # Enforce permissibility constraints in ReID
            for r_idx, k in enumerate(unassigned_cluster_indices):
                for c_idx, gid in enumerate(available_gids):
                    if not is_gid_permissible_for_cluster(gid, clusters[k]):
                        reid_cost[r_idx, c_idx] = 1e6

            row, col = linear_sum_assignment(reid_cost)
            for r, c in zip(row, col):
                if reid_cost[r, c] < self.th_reid:
                    k_idx = unassigned_cluster_indices[r]
                    gid = available_gids[c]
                    cluster_gids[k_idx] = gid

        # 4. Allocate new GIDs for clusters that still have no GID
        for k in range(K):
            if cluster_gids[k] is None:
                cluster_gids[k] = self._next_gid
                self._next_gid += 1

        # 5. Build final mapping:
        # - Locked tracks keep their pre-existing GID exactly to guarantee absolute static GID stability
        # - New/unassigned tracks get the GID of their cluster (which is guaranteed conflict-free)
        mapping: Dict[Tuple[int, int], int] = {}
        seen_gids = set()

        for k, idxs in enumerate(clusters):
            c_gid = cluster_gids[k]
            for idx in idxs:
                cid = int(cam_ids[idx])
                pid = int(person_ids[idx])
                if (cid, pid) in locked_mapping:
                    gid = locked_mapping[(cid, pid)]
                else:
                    gid = c_gid
                mapping[(cid, pid)] = gid
                seen_gids.add(gid)

        # 6. Update global track database
        # Clear cam_tracks for all active/seen global tracks once per frame
        for gid in seen_gids:
            if gid in self.tracks:
                self.tracks[gid].cam_tracks.clear()

        # Update global tracks
        for k, idxs in enumerate(clusters):
            c_gid = cluster_gids[k]
            
            # Update feature for the representative cluster GID
            if c_gid in self.tracks:
                gt = self.tracks[c_gid]
                gt.update_feature(cluster_feats[k], self.feat_smooth)
                gt.last_seen_frame = frame_id
                gt.lost_age = 0
            else:
                gt = GlobalTrack(c_gid, cluster_feats[k], frame_id)
                self.tracks[c_gid] = gt

            # Populate cam_tracks based on final mapping
            for idx in idxs:
                cid = int(cam_ids[idx])
                pid = int(person_ids[idx])
                gid = mapping[(cid, pid)]
                if gid in self.tracks:
                    self.tracks[gid].cam_tracks[cid] = pid
                    self.tracks[gid].last_seen_frame = frame_id
                    self.tracks[gid].lost_age = 0

        self._age_unseen(seen_gids)
        return mapping

    def age_all(self):
        self._age_unseen(set())

    def _compute_cluster_features(
        self, clusters: List[List[int]], features: np.ndarray,
    ) -> np.ndarray:
        K = len(clusters)
        D = features.shape[1]
        cluster_feats = np.empty((K, D), dtype=np.float64)
        for k, idxs in enumerate(clusters):
            cluster_feats[k] = np.mean(features[idxs], axis=0)
        return l2_normalise_rows(cluster_feats)

    def _reid_clusters(self, cluster_feats: np.ndarray) -> List[Optional[int]]:
        K = len(cluster_feats)
        assigned: List[Optional[int]] = [None] * K

        if not self.tracks:
            return assigned

        gids = list(self.tracks.keys())
        gf = np.stack([self.tracks[g].feature for g in gids])
        reid_cost = cosine_distance_matrix(cluster_feats, gf)

        row, col = linear_sum_assignment(reid_cost)
        for r, c in zip(row, col):
            if reid_cost[r, c] < self.th_reid:
                assigned[r] = gids[c]

        return assigned

    def _age_unseen(self, seen: set):
        for gid in list(self.tracks):
            if gid not in seen:
                self.tracks[gid].lost_age += 1
                if self.tracks[gid].lost_age > self.max_lost_age:
                    del self.tracks[gid]


class GlobalTrackV4:
    __slots__ = ("global_id", "cam_features", "last_seen_frame", "lost_age", "cam_tracks")

    def __init__(self, global_id: int, frame_id: int):
        self.global_id = global_id
        self.cam_features: Dict[int, np.ndarray] = {}
        self.last_seen_frame = frame_id
        self.lost_age = 0
        self.cam_tracks: Dict[int, int] = {}

    def update_feature(self, cam_id: int, feat: np.ndarray, smooth: float = 0.1):
        if cam_id not in self.cam_features:
            self.cam_features[cam_id] = feat.copy()
        else:
            self.cam_features[cam_id] = (1.0 - smooth) * self.cam_features[cam_id] + smooth * feat
            norm = np.linalg.norm(self.cam_features[cam_id])
            if norm > 0:
                self.cam_features[cam_id] /= norm


class GlobalTrackManagerV4:
    """Manages global track pool: Hungarian ReID, allocation, lost aging.
    Matches features per camera rather than a single global feature.
    """

    def __init__(self, config: Dict):
        self.th_reid: float = config.get("reid", 0.4)
        self.max_lost_age: int = config.get("max_lost_age", 300)
        self.feat_smooth: float = config.get("feature_smooth", 0.1)

        self.tracks: Dict[int, GlobalTrackV4] = {}
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
        gids: Optional[List[Optional[int]]] = None,
    ) -> Dict[Tuple[int, int], int]:
        K = len(clusters)
        if K == 0:
            self._age_unseen(set())
            return {}

        cluster_cam_feats = self._compute_cluster_cam_features(clusters, features, cam_ids)

        # 1. Collect static pre-existing GID assignments from the current frame's active tracks
        static_mapping: Dict[Tuple[int, int], int] = {}
        if gids is not None:
            for idx, gid in enumerate(gids):
                if gid is not None:
                    cid = int(cam_ids[idx])
                    pid = int(person_ids[idx])
                    static_mapping[(cid, pid)] = gid

        # Fallback to last frame memory if needed (for safety/backward compatibility)
        track_to_prev_gid = {}
        for gid, gt in self.tracks.items():
            for cid, pid in gt.cam_tracks.items():
                track_to_prev_gid[(cid, pid)] = gid

        # Combine static mapping with fallback for all active tracks in clusters
        locked_mapping: Dict[Tuple[int, int], int] = {}
        for k, idxs in enumerate(clusters):
            for idx in idxs:
                cid = int(cam_ids[idx])
                pid = int(person_ids[idx])
                if (cid, pid) in static_mapping:
                    locked_mapping[(cid, pid)] = static_mapping[(cid, pid)]
                elif (cid, pid) in track_to_prev_gid:
                    locked_mapping[(cid, pid)] = track_to_prev_gid[(cid, pid)]

        # Map cid -> set of gids already locked on this camera in the current frame
        locked_gids_per_cam: Dict[int, Set[int]] = {}
        # Map gid -> Tuple[int, int] identifying which track locked this gid
        gid_locked_by: Dict[int, Tuple[int, int]] = {}
        for (cid, pid), gid in locked_mapping.items():
            if cid not in locked_gids_per_cam:
                locked_gids_per_cam[cid] = set()
            locked_gids_per_cam[cid].add(gid)
            gid_locked_by[gid] = (cid, pid)

        # Permissibility check: A GID is permissible for a cluster if it's not locked on any of the cluster's cameras
        # by a different track.
        def is_gid_permissible_for_cluster(gid_val: int, idxs_list: List[int]) -> bool:
            for idx_in in idxs_list:
                cid_in = int(cam_ids[idx_in])
                pid_in = int(person_ids[idx_in])
                if gid_val in locked_gids_per_cam.get(cid_in, set()):
                    if gid_locked_by[gid_val] != (cid_in, pid_in):
                        return False
            return True

        # 2. For each cluster, determine its representative GID (if any)
        cluster_gids: List[Optional[int]] = [None] * K
        for k, idxs in enumerate(clusters):
            gids_in_cluster = []
            for idx in idxs:
                cid = int(cam_ids[idx])
                pid = int(person_ids[idx])
                if (cid, pid) in locked_mapping:
                    gids_in_cluster.append(locked_mapping[(cid, pid)])
            
            if gids_in_cluster:
                # Sort unique candidates by frequency descending
                candidates = sorted(set(gids_in_cluster), key=lambda g: gids_in_cluster.count(g), reverse=True)
                for cand in candidates:
                    if is_gid_permissible_for_cluster(cand, idxs):
                        cluster_gids[k] = cand
                        break

        # 3. Match remaining clusters via Hungarian ReID against unused GIDs
        unassigned_cluster_indices = [k for k in range(K) if cluster_gids[k] is None]
        used_gids = set(g for g in cluster_gids if g is not None)
        available_gids = [g for g in self.tracks.keys() if g not in used_gids]

        if unassigned_cluster_indices and available_gids:
            N_sub = len(available_gids)
            K_sub = len(unassigned_cluster_indices)
            reid_cost = np.ones((K_sub, N_sub), dtype=np.float64) * 1e6

            for r_idx, k in enumerate(unassigned_cluster_indices):
                c_feats = cluster_cam_feats[k]
                for c_idx, gid in enumerate(available_gids):
                    # Enforce permissibility constraints in ReID
                    if not is_gid_permissible_for_cluster(gid, clusters[k]):
                        continue

                    gt = self.tracks[gid]
                    min_dist = 1e6
                    for cid_c, f_c in c_feats.items():
                        for cid_gt, f_gt in gt.cam_features.items():
                            sim = np.clip(np.dot(f_c, f_gt), -1.0, 1.0)
                            dist = 1.0 - sim
                            if dist < min_dist:
                                min_dist = dist
                    if min_dist < 1e6:
                        reid_cost[r_idx, c_idx] = min_dist

            row, col = linear_sum_assignment(reid_cost)
            for r, c in zip(row, col):
                if reid_cost[r, c] < self.th_reid:
                    k_idx = unassigned_cluster_indices[r]
                    gid = available_gids[c]
                    cluster_gids[k_idx] = gid

        # 4. Allocate new GIDs for clusters that still have no GID
        for k in range(K):
            if cluster_gids[k] is None:
                cluster_gids[k] = self._next_gid
                self._next_gid += 1

        # 5. Build final mapping:
        # - Locked tracks keep their pre-existing GID exactly to guarantee absolute static GID stability
        # - New/unassigned tracks get the GID of their cluster (which is guaranteed conflict-free)
        mapping: Dict[Tuple[int, int], int] = {}
        seen_gids = set()

        for k, idxs in enumerate(clusters):
            c_gid = cluster_gids[k]
            for idx in idxs:
                cid = int(cam_ids[idx])
                pid = int(person_ids[idx])
                if (cid, pid) in locked_mapping:
                    gid = locked_mapping[(cid, pid)]
                else:
                    gid = c_gid
                mapping[(cid, pid)] = gid
                seen_gids.add(gid)

        # 6. Update global track database
        # Clear cam_tracks for all active/seen global tracks once per frame
        for gid in seen_gids:
            if gid in self.tracks:
                self.tracks[gid].cam_tracks.clear()

        # Update global tracks
        for k, idxs in enumerate(clusters):
            c_gid = cluster_gids[k]
            
            # Update feature for the representative cluster GID
            if c_gid in self.tracks:
                gt = self.tracks[c_gid]
                gt.last_seen_frame = frame_id
                gt.lost_age = 0
            else:
                gt = GlobalTrackV4(c_gid, frame_id)
                self.tracks[c_gid] = gt

            for cid, f_c in cluster_cam_feats[k].items():
                gt.update_feature(cid, f_c, self.feat_smooth)

            # Populate cam_tracks based on final mapping
            for idx in idxs:
                cid = int(cam_ids[idx])
                pid = int(person_ids[idx])
                gid = mapping[(cid, pid)]
                if gid in self.tracks:
                    self.tracks[gid].cam_tracks[cid] = pid
                    self.tracks[gid].last_seen_frame = frame_id
                    self.tracks[gid].lost_age = 0

        self._age_unseen(seen_gids)
        return mapping

    def age_all(self):
        self._age_unseen(set())

    def _compute_cluster_cam_features(
        self, clusters: List[List[int]], features: np.ndarray, cam_ids: np.ndarray
    ) -> List[Dict[int, np.ndarray]]:
        cluster_cam_feats = []
        for idxs in clusters:
            c_feats = {}
            for idx in idxs:
                cid = int(cam_ids[idx])
                if cid not in c_feats:
                    c_feats[cid] = [features[idx]]
                else:
                    c_feats[cid].append(features[idx])
            
            cam_dict = {}
            for cid, f_list in c_feats.items():
                f_mean = np.mean(f_list, axis=0)
                norm = np.linalg.norm(f_mean)
                if norm > 0:
                    f_mean /= norm
                cam_dict[cid] = f_mean
            cluster_cam_feats.append(cam_dict)
        return cluster_cam_feats

    def _reid_clusters(self, cluster_cam_feats: List[Dict[int, np.ndarray]]) -> List[Optional[int]]:
        K = len(cluster_cam_feats)
        assigned: List[Optional[int]] = [None] * K

        if not self.tracks:
            return assigned

        gids = list(self.tracks.keys())
        N = len(gids)

        reid_cost = np.ones((K, N), dtype=np.float64) * 1e6

        for k, c_feats in enumerate(cluster_cam_feats):
            for j, gid in enumerate(gids):
                gt = self.tracks[gid]
                
                min_dist = 1e6
                for cid_c, f_c in c_feats.items():
                    for cid_gt, f_gt in gt.cam_features.items():
                        sim = np.clip(np.dot(f_c, f_gt), -1.0, 1.0)
                        dist = 1.0 - sim
                        if dist < min_dist:
                                min_dist = dist
                
                if min_dist < 1e6:
                    reid_cost[k, j] = min_dist

        row, col = linear_sum_assignment(reid_cost)
        for r, c in zip(row, col):
            if reid_cost[r, c] < self.th_reid:
                assigned[r] = gids[c]

        return assigned

    def _age_unseen(self, seen: set):
        for gid in list(self.tracks):
            if gid not in seen:
                self.tracks[gid].lost_age += 1
                if self.tracks[gid].lost_age > self.max_lost_age:
                    del self.tracks[gid]
