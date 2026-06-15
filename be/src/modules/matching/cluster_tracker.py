"""Cluster-level multi-camera tracker.

Adapted from AIC2024_Track1_Nota/trackers/multicam_tracker/cluster_track.py.
Tracks groups (clusters) of cross-camera matched tracks over time, maintaining
persistent global IDs with feature history and self-refinement.

Key concepts:
- **MTrack**: A cluster track representing a person across cameras.
  Stores a deque of features and uses Procrustes analysis to avoid
  adding duplicate observations.
- **ClusterTracker**: Manages MTrack lifecycle (tracked / lost / removed)
  with multi-step association and periodic self-refinement.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes

from modules.matching.cross_camera_clustering_v2 import (
    embedding_distance,
    euclidean_distance,
    group_distances,
    linear_assignment,
)

logger = logging.getLogger(__name__)

# Try importing sklearn for refinement; gracefully degrade if not available.
try:
    from sklearn.cluster import AgglomerativeClustering

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    logger.warning("sklearn not available; cluster self-refinement disabled")


# -----------------------------------------------------------------------
# Track state
# -----------------------------------------------------------------------

class _TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4


class _BaseTrack:
    """Minimal base class for cluster-level tracks."""

    _count = 0

    track_id = 0
    is_activated = False
    state = _TrackState.New
    frame_id = 0
    start_frame = 0

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        _BaseTrack._count += 1
        return _BaseTrack._count

    def mark_lost(self):
        self.state = _TrackState.Lost

    def mark_removed(self):
        self.state = _TrackState.Removed

    @staticmethod
    def clear_count():
        _BaseTrack._count = 0


# -----------------------------------------------------------------------
# MTrack (cluster track)
# -----------------------------------------------------------------------

class MTrack(_BaseTrack):
    """A cluster-level track representing a person across cameras.

    Stores a feature gallery (deque) with pose-quality filtering
    and Procrustes-based duplicate rejection.
    """

    def __init__(
        self,
        global_id: int,
        centroid: np.ndarray,
        feat_list: Optional[List[np.ndarray]],
        keypoints_list: Optional[List[np.ndarray]],
        coords: Optional[List[np.ndarray]],
        min_hits: int = 30,
        feat_history: int = 10,
        pose_thresh: int = 10,
    ):
        self.is_activated = False
        self.centroid = centroid
        self.global_id = global_id

        self.smooth_feat = None
        self.curr_feat = None
        self.curr_keypoints = None
        self.curr_coords = coords

        self.features: deque = deque([], maxlen=feat_history)
        self.keypoints_history: deque = deque([], maxlen=feat_history)
        self.pose_thresh = pose_thresh

        self.tracklet_len = 0
        self.alpha = 0.9
        self.min_hits = min_hits

        if feat_list is not None:
            self.update_features(feat_list, keypoints_list, coords)

    def update_features(
        self,
        features: List[np.ndarray],
        keypoints_list: Optional[List[np.ndarray]],
        coords: Optional[List[np.ndarray]],
    ):
        """Add new features with quality filtering.

        Uses keypoint count and Procrustes analysis to avoid adding
        duplicate or low-quality observations.
        """
        self.curr_feat = features
        self.curr_keypoints = keypoints_list
        self.curr_coords = coords

        if keypoints_list is None:
            keypoints_list = [None] * len(features)

        for feat, kpts in zip(features, keypoints_list):
            if kpts is None:
                continue
            if len(self.features) >= self.features.maxlen:
                return

            # Check keypoint quality
            num_visible = int(np.sum(kpts[:, 2] > 0.5)) if kpts is not None and kpts.shape[0] >= 17 else 0
            if num_visible >= self.pose_thresh:
                # Procrustes duplicate check
                if self._is_duplicate_pose(kpts):
                    continue
                self.features.append(feat)
                self.keypoints_history.append(kpts)

        # Fallback: if no features added yet, use best available
        if len(self.features) == 0:
            max_num, max_feat, max_kpts = 0, None, None
            for feat, kpts in zip(features, keypoints_list):
                if kpts is None:
                    continue
                n = int(np.sum(kpts[:, 2] > 0.5))
                if n > max_num:
                    max_num = n
                    max_feat = feat
                    max_kpts = kpts

            if max_feat is not None and max_num > 0:
                self.features.append(max_feat)
                self.keypoints_history.append(max_kpts)
            elif features:
                # Last resort: add all features
                for feat in features:
                    self.features.append(feat)

    def _is_duplicate_pose(self, new_kpts: np.ndarray) -> bool:
        """Check if new_kpts is similar to any existing keypoints via Procrustes."""
        for old_kpts in self.keypoints_history:
            if old_kpts is None:
                continue
            try:
                p1 = old_kpts[:, :2].copy()
                p2 = new_kpts[:, :2].copy()
                _, _, distance = procrustes(p1, p2)
                if distance < 1e-8:
                    return True
            except (ValueError, np.linalg.LinAlgError):
                continue
        return False

    def activate(self, frame_id: int):
        """Start a new tracklet."""
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = _TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "MTrack", frame_id: int, new_id: bool = False):
        """Re-activate a lost track with new observations."""
        self.centroid = new_track.centroid
        self.global_id = new_track.global_id

        if new_track.curr_feat is not None:
            self.update_features(
                new_track.curr_feat,
                new_track.curr_keypoints,
                new_track.curr_coords,
            )
        self.tracklet_len = 0
        self.state = _TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track: "MTrack", frame_id: int):
        """Update a matched track with new observations."""
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.centroid = new_track.centroid
        self.global_id = new_track.global_id

        if new_track.curr_feat is not None:
            self.update_features(
                new_track.curr_feat,
                new_track.curr_keypoints,
                new_track.curr_coords,
            )

        self.state = _TrackState.Tracked
        if self.tracklet_len > self.min_hits:
            self.is_activated = True


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

def _joint_mtracks(lista: List[MTrack], listb: List[MTrack]) -> List[MTrack]:
    """Merge two lists of MTrack without duplicates."""
    exists = {}
    res = []
    for t in lista:
        exists[t.track_id] = 1
        res.append(t)
    for t in listb:
        if not exists.get(t.track_id, 0):
            exists[t.track_id] = 1
            res.append(t)
    return res


def _sub_mtracks(lista: List[MTrack], listb: List[MTrack]) -> List[MTrack]:
    """Remove tracks in listb from lista."""
    ids_to_remove = {t.track_id for t in listb}
    return [t for t in lista if t.track_id not in ids_to_remove]


# -----------------------------------------------------------------------
# ClusterTracker
# -----------------------------------------------------------------------

class ClusterTracker:
    """Tracks clusters over time with multi-step association.

    Adapted from AIC2024's ``MCTracker``.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.tracked_mtracks: List[MTrack] = []
        self.lost_mtracks: List[MTrack] = []
        self.removed_mtracks: List[MTrack] = []
        _BaseTrack.clear_count()

        self.frame_id = 0
        self.max_time_lost: int = config.get("max_time_lost", 18000)
        self.min_hits: int = config.get("min_hits", 10)
        self.match_thresh: float = config.get("match_thresh", 0.999)
        self.emb_thresh: float = config.get("emb_thresh", 0.30)
        self.euc_thresh: float = config.get("euc_thresh", 1.5)
        self.refinement_interval: int = config.get("refinement_interval", 5)

        # Refinement thresholds
        self.refine_1st_emb: float = config.get("refine_1st_emb", 0.325)
        self.refine_2nd_emb: float = config.get("refine_2nd_emb", 0.325)
        self.refine_2nd_euc: float = config.get("refine_2nd_euc", 1.0)
        self.refine_3rd_emb: float = config.get("refine_3rd_emb", 0.30)

        self._clustering = None
        if _HAS_SKLEARN:
            self._clustering = AgglomerativeClustering(
                n_clusters=2, metric="cosine", linkage="average",
            )

    def update(self, groups: np.ndarray):
        """Update cluster tracks with new groups.

        Args:
            groups: numpy object array from CrossCameraClusterer.update().
                    Each row: ``[global_id, features, centroid, keypoints, coords]``
        """
        self.frame_id += 1
        activated_mtracks = []
        refind_mtracks = []
        lost_mtracks = []
        removed_mtracks = []

        if len(groups) > 0:
            global_ids = groups[:, 0]
            features = groups[:, 1]
            centroids = groups[:, 2]
            keypoints_list = groups[:, 3]
            coords = groups[:, 4]
        else:
            global_ids = []
            features = []
            centroids = []
            keypoints_list = []
            coords = []

        # Create MTrack objects from new groups
        if len(centroids) > 0:
            new_groups = [
                MTrack(g, c, f, k, cd, self.min_hits)
                for g, c, f, k, cd in zip(
                    global_ids, centroids, features, keypoints_list, coords,
                )
            ]
        else:
            new_groups = []

        # --- Step 1: Separate tracked vs unconfirmed ---
        unconfirmed = []
        tracked_mtracks = []
        for track in self.tracked_mtracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_mtracks.append(track)

        # --- Step 2: First association with tracked mtracks ---
        exist_features = [f for m in tracked_mtracks for f in list(m.features)]
        lengths_exists = [len(m.features) for m in tracked_mtracks]
        new_features = [f for g in new_groups for f in list(g.features)]
        lengths_new = [len(g.features) for g in new_groups]
        exist_centroids = [m.centroid for m in tracked_mtracks]
        new_centroids = [g.centroid for g in new_groups]

        shape = (len(lengths_exists), len(lengths_new))
        if 0 in shape:
            dists = np.empty(shape)
        else:
            rerank_dists = embedding_distance(exist_features, new_features) / 2.0
            emb_dists = group_distances(rerank_dists, lengths_exists, lengths_new, shape)

            if self.frame_id % 10 == 0:
                dists = emb_dists
            else:
                euc_dists = euclidean_distance(exist_centroids, new_centroids)

                emb_min, emb_max = np.min(emb_dists), np.max(emb_dists)
                euc_min, euc_max = np.min(euc_dists), np.max(euc_dists)
                emb_range = emb_max - emb_min if emb_max > emb_min else 1.0
                euc_range = euc_max - euc_min if euc_max > euc_min else 1.0

                norm_emb = (emb_dists - emb_min) / emb_range
                norm_euc = (euc_dists - euc_min) / euc_range
                dists = 0.5 * norm_euc + 0.5 * norm_emb
                dists[euc_dists > 1.0] = 1.0

        matches, u_exist, u_new = linear_assignment(dists, thresh=self.match_thresh)

        for iexist, inew in matches:
            exist = tracked_mtracks[iexist]
            new = new_groups[inew]
            if exist.state == _TrackState.Tracked:
                exist.update(new, self.frame_id)
                activated_mtracks.append(exist)
            else:
                exist.re_activate(new, self.frame_id, new_id=False)
                refind_mtracks.append(exist)

        for it in u_exist:
            track = tracked_mtracks[it]
            if track.state != _TrackState.Lost and track.state != _TrackState.Removed:
                track.mark_lost()
                lost_mtracks.append(track)

        # --- Step 3: Second association with lost mtracks ---
        new_groups = [new_groups[i] for i in u_new]

        lost_features = [f for m in self.lost_mtracks for f in list(m.features)]
        lengths_lost = [len(m.features) for m in self.lost_mtracks]
        new_features = [f for g in new_groups for f in list(g.features)]
        lengths_new = [len(g.features) for g in new_groups]

        shape = (len(lengths_lost), len(lengths_new))
        if 0 in shape:
            emb_dists = np.empty(shape)
        else:
            rerank_dists = embedding_distance(lost_features, new_features) / 2.0
            emb_dists = group_distances(rerank_dists, lengths_lost, lengths_new, shape)

        dists = emb_dists
        matches, u_lost, u_new = linear_assignment(dists, thresh=self.emb_thresh)

        for ilost, inew in matches:
            lost = self.lost_mtracks[ilost]
            new = new_groups[inew]
            lost.re_activate(new, self.frame_id, new_id=False)
            refind_mtracks.append(lost)

        # --- Step 4: Unconfirmed tracks ---
        new_groups = [new_groups[i] for i in u_new]

        exist_centroids = [m.centroid for m in unconfirmed]
        new_centroids = [g.centroid for g in new_groups]
        exist_features = [f for m in unconfirmed for f in list(m.features)]
        lengths_exists = [len(m.features) for m in unconfirmed]
        new_features = [f for g in new_groups for f in list(g.features)]
        lengths_new = [len(g.features) for g in new_groups]

        shape = (len(lengths_exists), len(lengths_new))
        if 0 in shape:
            dists = np.empty(shape)
        else:
            rerank_dists = embedding_distance(exist_features, new_features) / 2.0
            emb_dists = group_distances(rerank_dists, lengths_exists, lengths_new, shape)
            euc_dists = euclidean_distance(exist_centroids, new_centroids)

            emb_min, emb_max = np.min(emb_dists), np.max(emb_dists)
            euc_min, euc_max = np.min(euc_dists), np.max(euc_dists)
            emb_range = emb_max - emb_min if emb_max > emb_min else 1.0
            euc_range = euc_max - euc_min if euc_max > euc_min else 1.0

            norm_emb = (emb_dists - emb_min) / emb_range
            norm_euc = (euc_dists - euc_min) / euc_range
            dists = 0.5 * norm_euc + 0.5 * norm_emb

            if shape == (1, 1):
                dists = emb_dists
            dists[euc_dists > self.euc_thresh] = 1.0

        matches, u_unconfirmed, u_new = linear_assignment(dists, thresh=self.match_thresh)
        for iexist, inew in matches:
            unconfirmed[iexist].update(new_groups[inew], self.frame_id)
            activated_mtracks.append(unconfirmed[iexist])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_mtracks.append(track)

        # --- Step 5: Init new mtracks ---
        for inew in u_new:
            track = new_groups[inew]
            track.activate(self.frame_id)
            activated_mtracks.append(track)

        # --- Step 6: Remove stale lost tracks ---
        for track in self.lost_mtracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_mtracks.append(track)

        # --- Merge ---
        self.tracked_mtracks = [t for t in self.tracked_mtracks if t.state == _TrackState.Tracked]
        self.tracked_mtracks = _joint_mtracks(self.tracked_mtracks, activated_mtracks)
        self.tracked_mtracks = _joint_mtracks(self.tracked_mtracks, refind_mtracks)
        self.lost_mtracks = _sub_mtracks(self.lost_mtracks, self.tracked_mtracks)
        self.lost_mtracks.extend(lost_mtracks)
        self.lost_mtracks = _sub_mtracks(self.lost_mtracks, removed_mtracks)

        # Update start_frame for stable tracks
        for mtrack in self.tracked_mtracks:
            if mtrack.start_frame == 1 or not mtrack.is_activated:
                pass
            elif mtrack.frame_id - mtrack.start_frame >= 30:
                mtrack.start_frame = 1

        output = [t for t in self.tracked_mtracks if t.is_activated]
        logger.debug(
            "MCTracker frame %d: %d tracked, %d unconfirmed, %d lost",
            self.frame_id,
            len(output),
            len([t for t in self.tracked_mtracks if not t.is_activated]),
            len(self.lost_mtracks),
        )

    def refinement_clusters(self):
        """Self-refinement of tracked clusters.

        Step 1: Detect clusters containing two different people (feature split).
        Step 2: Merge clusters that represent the same person.
        Step 3: Compare tracked clusters with lost clusters for re-identification.
        """
        if not _HAS_SKLEARN:
            return

        tracked_mtracks = [t for t in self.tracked_mtracks if t.is_activated]
        unconfirmed_mtracks = [t for t in self.tracked_mtracks if not t.is_activated]
        lost_mtracks = self.lost_mtracks
        all_mtracks = tracked_mtracks + unconfirmed_mtracks + lost_mtracks

        # --- Step 1: Detect cluster with two people ---
        for mtrack in all_mtracks:
            features_arr = np.array(list(mtrack.features))
            if features_arr.shape[0] <= 1:
                continue

            try:
                self._clustering.fit(features_arr)
                labels = self._clustering.labels_
            except Exception:
                continue

            a_feats = features_arr[labels == 0]
            b_feats = features_arr[labels == 1]

            # Compare max-area features from each sub-cluster
            a_rep = a_feats[0:1]  # Use first as representative
            b_rep = b_feats[0:1]
            emb_dist = embedding_distance(a_rep.tolist(), b_rep.tolist()) / 2.0
            dist_val = emb_dist[0, 0] if emb_dist.size > 0 else 0

            if dist_val > self.refine_1st_emb:
                logger.info(
                    "Refine step1: cluster %d has two people (%d + %d) dist=%.4f",
                    mtrack.track_id, int(np.sum(labels == 0)),
                    int(np.sum(labels == 1)), dist_val,
                )
                # Keep the earlier-added sub-cluster
                a_mean_idx = float(np.mean(np.where(labels == 0)[0]))
                b_mean_idx = float(np.mean(np.where(labels == 1)[0]))

                mtrack.features = deque([], maxlen=mtrack.features.maxlen)
                if a_mean_idx <= b_mean_idx:
                    mtrack.features.extend(a_feats)
                else:
                    mtrack.features.extend(b_feats)

        # --- Step 2: Merge duplicate tracked clusters ---
        tracked_mtracks = [t for t in self.tracked_mtracks if t.is_activated]
        removed_mtracks = []

        from itertools import combinations
        for a_cluster, b_cluster in combinations(tracked_mtracks, 2):
            if a_cluster.state == _TrackState.Removed or b_cluster.state == _TrackState.Removed:
                continue

            a_feats = list(a_cluster.features)
            b_feats = list(b_cluster.features)
            if not a_feats or not b_feats:
                continue

            emb_dists = embedding_distance(a_feats, b_feats) / 2.0
            emb_dist = float(np.mean(emb_dists))

            a_cent = [a_cluster.centroid]
            b_cent = [b_cluster.centroid]
            euc_dist = euclidean_distance(a_cent, b_cent)[0, 0]

            if emb_dist < self.refine_2nd_emb and euc_dist < self.refine_2nd_euc:
                if (a_cluster.start_frame == 1 and b_cluster.start_frame == 1
                        and self.frame_id > 10):
                    continue
                logger.info(
                    "Refine step2: cluster %d and %d are same person",
                    a_cluster.track_id, b_cluster.track_id,
                )
                if a_cluster.track_id <= b_cluster.track_id:
                    b_cluster.mark_removed()
                    removed_mtracks.append(b_cluster)
                else:
                    a_cluster.mark_removed()
                    removed_mtracks.append(a_cluster)

        self.tracked_mtracks = [t for t in tracked_mtracks if t.state == _TrackState.Tracked]

        # --- Step 3: Compare tracked vs lost for re-identification ---
        tracked_mtracks = [t for t in self.tracked_mtracks if t.is_activated]
        refind_mtracks = []

        for tracked in tracked_mtracks:
            if tracked.state == _TrackState.Removed:
                continue
            for lost in lost_mtracks:
                if lost.state == _TrackState.Tracked:
                    continue

                t_feats = list(tracked.features)
                l_feats = list(lost.features)
                if not t_feats or not l_feats:
                    continue

                emb_dists = embedding_distance(t_feats, l_feats) / 2.0
                emb_dist = float(np.mean(emb_dists))

                if (emb_dist < self.refine_3rd_emb
                        and lost.track_id < tracked.track_id
                        and tracked.start_frame != 1):
                    logger.info(
                        "Refine step3: cluster %d removed (matched lost %d)",
                        tracked.track_id, lost.track_id,
                    )
                    lost.re_activate(tracked, self.frame_id, new_id=False)
                    refind_mtracks.append(lost)
                    tracked.mark_removed()
                    removed_mtracks.append(tracked)
                    break

        # --- Final merge ---
        self.tracked_mtracks = [t for t in self.tracked_mtracks if t.state == _TrackState.Tracked]
        self.tracked_mtracks = _joint_mtracks(self.tracked_mtracks, refind_mtracks)
        self.tracked_mtracks = _joint_mtracks(self.tracked_mtracks, unconfirmed_mtracks)
        self.lost_mtracks = _sub_mtracks(self.lost_mtracks, self.tracked_mtracks)
        self.lost_mtracks = _sub_mtracks(self.lost_mtracks, removed_mtracks)
