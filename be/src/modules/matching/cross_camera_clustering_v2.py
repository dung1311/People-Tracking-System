"""Cross-camera clustering via pairwise matching.

Adapted from AIC2024_Track1_Nota/trackers/multicam_tracker/clustering.py.
Replaces the Union-Find based approach in the existing MCT pipeline with
a pairwise camera matching strategy that:

1. For each camera pair, filters tracks by pose quality.
2. Computes embedding + Euclidean distances with pose-weighted fusion.
3. Runs Hungarian matching to assign shared ``t_global_id`` values.
4. Groups matched tracks into clusters for the MCTracker.

Uses ``scipy`` for distance computation and assignment (no ``lap``/``cython_bbox``).
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from modules.data_templates.sct_template import TrackInfo

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# COCO keypoint indices (for pose quality checks)
# -----------------------------------------------------------------------
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_ELBOW = 7
_RIGHT_ELBOW = 8
_LEFT_WRIST = 9
_RIGHT_WRIST = 10
_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_KNEE = 13
_RIGHT_KNEE = 14
_LEFT_ANKLE = 15
_RIGHT_ANKLE = 16
_NOSE = 0


# -----------------------------------------------------------------------
# ID Distributor
# -----------------------------------------------------------------------

class IDDistributor:
    """Assigns unique temporary global IDs for cross-camera clustering."""

    def __init__(self, init_id: int = 0):
        self.cur_id = init_id

    def assign_id(self) -> int:
        self.cur_id += 1
        return self.cur_id

    def reset(self):
        self.cur_id = 0


# -----------------------------------------------------------------------
# Pose quality checks (adapted from CrowdPose → COCO)
# -----------------------------------------------------------------------

def _pose_check(kpts: np.ndarray, thr: float = 0.3) -> np.ndarray:
    """Check visibility of 5 body regions: head, neck, shoulder, hip, knee.

    Args:
        kpts: ``(17, 3)`` COCO keypoints.
        thr: confidence threshold.

    Returns:
        ``(5,)`` boolean array.
    """
    # head → nose
    head = kpts[_NOSE, 2]
    # neck → average of shoulders
    neck = (kpts[_LEFT_SHOULDER, 2] + kpts[_RIGHT_SHOULDER, 2]) / 2
    # shoulder
    shoulder = max(kpts[_LEFT_SHOULDER, 2], kpts[_RIGHT_SHOULDER, 2])
    # hip
    hip = (kpts[_LEFT_HIP, 2] + kpts[_RIGHT_HIP, 2]) / 2
    # knee
    knee = (kpts[_LEFT_KNEE, 2] + kpts[_RIGHT_KNEE, 2]) / 2

    return np.array([head, neck, shoulder, hip, knee]) > thr


def _pose_check_all(kpts: np.ndarray, thr: float = 0.5):
    """Full pose visibility check returning (heads, points, parts).

    Args:
        kpts: ``(17, 3)`` COCO keypoints.
        thr: confidence threshold.

    Returns:
        Tuple of 3 boolean arrays: (has_heads, has_points, has_parts).
    """
    head = kpts[_NOSE, 2]
    neck = (kpts[_LEFT_SHOULDER, 2] + kpts[_RIGHT_SHOULDER, 2]) / 2
    l_shoulder = kpts[_LEFT_SHOULDER, 2]
    r_shoulder = kpts[_RIGHT_SHOULDER, 2]

    l_elbow = kpts[_LEFT_ELBOW, 2]
    r_elbow = kpts[_RIGHT_ELBOW, 2]
    l_wrist = kpts[_LEFT_WRIST, 2]
    r_wrist = kpts[_RIGHT_WRIST, 2]
    l_hip = kpts[_LEFT_HIP, 2]
    r_hip = kpts[_RIGHT_HIP, 2]
    l_knee = kpts[_LEFT_KNEE, 2]
    r_knee = kpts[_RIGHT_KNEE, 2]
    l_ankle = kpts[_LEFT_ANKLE, 2]
    r_ankle = kpts[_RIGHT_ANKLE, 2]

    has_heads = np.array([head, neck, l_shoulder, r_shoulder]) > thr
    has_points = np.array([
        l_elbow, r_elbow, l_wrist, r_wrist,
        l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle,
    ]) > thr
    has_parts = np.array([
        max(l_shoulder, r_shoulder),
        max(l_elbow, r_elbow),
        max(l_wrist, r_wrist),
        max(l_hip, r_hip),
        max(l_knee, r_knee),
        max(l_ankle, r_ankle),
    ]) > thr

    return has_heads, has_points, has_parts


def _count_nearby_keypoints(bbox, all_keypoints_list: List[np.ndarray]) -> int:
    """Count total keypoints from all persons that fall inside this bbox."""
    x1, y1, x2, y2 = bbox[:4]
    total = 0
    for kpts in all_keypoints_list:
        if kpts is None:
            continue
        inside = (
            (kpts[:, 0] >= x1) & (kpts[:, 0] <= x2) &
            (kpts[:, 1] >= y1) & (kpts[:, 1] <= y2)
        )
        total += int(np.sum(inside))
    return total


# -----------------------------------------------------------------------
# Distance functions
# -----------------------------------------------------------------------

def embedding_distance(a_features: List[np.ndarray], b_features: List[np.ndarray]) -> np.ndarray:
    """Cosine distance matrix between two feature lists.

    Returns:
        ``(len(a), len(b))`` distance matrix.
    """
    if not a_features or not b_features:
        return np.empty((len(a_features), len(b_features)), dtype=np.float64)

    a = np.asarray(a_features, dtype=np.float64)
    b = np.asarray(b_features, dtype=np.float64)
    return np.maximum(0.0, cdist(a, b, metric="cosine"))


def euclidean_distance(a_locations: List[np.ndarray], b_locations: List[np.ndarray]) -> np.ndarray:
    """Euclidean distance matrix between two location lists.

    Returns:
        ``(len(a), len(b))`` distance matrix.
    """
    if not a_locations or not b_locations:
        return np.empty((len(a_locations), len(b_locations)), dtype=np.float64)

    a = np.asarray(a_locations, dtype=np.float64)
    b = np.asarray(b_locations, dtype=np.float64)
    return cdist(a, b, metric="euclidean")


def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """Run scipy linear_sum_assignment with threshold gating.

    Returns:
        (matches, unmatched_a, unmatched_b) similar to lap.lapjv interface.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1]),
        )

    # Mask out entries above threshold
    masked = cost_matrix.copy()
    masked[masked > thresh] = thresh + 1e6

    row_idx, col_idx = linear_sum_assignment(masked)

    matches = []
    unmatched_a = set(range(cost_matrix.shape[0]))
    unmatched_b = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
            unmatched_a.discard(r)
            unmatched_b.discard(c)

    matches = np.array(matches, dtype=int).reshape(-1, 2) if matches else np.empty((0, 2), dtype=int)
    return matches, np.array(sorted(unmatched_a)), np.array(sorted(unmatched_b))


# -----------------------------------------------------------------------
# Grouping helper (adapted from AIC2024 grouping_rerank)
# -----------------------------------------------------------------------

def group_distances(
    flat_dists: np.ndarray,
    lengths_a: List[int],
    lengths_b: List[int],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Aggregate flat per-feature distances into per-cluster distances.

    Each cluster may have multiple features. This computes the mean distance
    between all feature pairs of two clusters.

    Args:
        flat_dists: ``(sum(lengths_a), sum(lengths_b))`` distance matrix.
        lengths_a: number of features per cluster in group A.
        lengths_b: number of features per cluster in group B.
        shape: ``(len(lengths_a), len(lengths_b))``.

    Returns:
        ``shape`` distance matrix.
    """
    result = np.zeros(shape, dtype=np.float64)

    for i, len_a in enumerate(lengths_a):
        for j, len_b in enumerate(lengths_b):
            start_x = sum(lengths_a[:i])
            end_x = start_x + len_a
            start_y = sum(lengths_b[:j])
            end_y = start_y + len_b
            result[i, j] = np.mean(flat_dists[start_x:end_x, start_y:end_y])

    return result


# -----------------------------------------------------------------------
# CrossCameraClusterer
# -----------------------------------------------------------------------

class CrossCameraClusterer:
    """Clusters tracks across cameras using pairwise matching.

    Adapted from AIC2024's ``Clustering`` class.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.emb_thresh: float = config.get("emb_thresh", 0.30)
        self.euc_thresh: float = config.get("euc_thresh", 1.0)

        pose_cfg = config.get("pose_quality", {})
        self.hw_thresh: float = pose_cfg.get("hw_thresh", 1.25)
        self.min_visible_keypoints: int = pose_cfg.get("min_visible_keypoints", 6)
        self.min_head_keypoints: int = pose_cfg.get("min_head_keypoints", -1)
        self.min_num_kpts: int = pose_cfg.get("min_num_kpts", 16)

    def update(
        self,
        per_cam_tracks: Dict[int, List[TrackInfo]],
        id_distributor: IDDistributor,
    ) -> np.ndarray:
        """Run cross-camera pairwise matching and return grouped clusters.

        1. Assign temp global IDs to all tracks.
        2. For each camera pair, match tracks by embedding + euclidean.
        3. Unify matched tracks under the same t_global_id.
        4. Group by t_global_id.

        Args:
            per_cam_tracks: ``{cam_id: [TrackInfo, ...]}``
            id_distributor: assigns unique temp IDs.

        Returns:
            numpy object array of groups. Each row:
            ``[t_global_id, features_list, centroid, keypoints_list, coords_list]``
            Empty array if no groups.
        """
        # --- Step 0: Assign temp global IDs ---
        all_cam_ids = sorted(per_cam_tracks.keys())
        for cam_id in all_cam_ids:
            for track in per_cam_tracks[cam_id]:
                track.t_global_id = id_distributor.assign_id()
                # Reset matched_dist for this frame
                if not hasattr(track, "_matched_dist"):
                    track._matched_dist = None
                else:
                    track._matched_dist = None

        # --- Step 1: Pairwise matching ---
        cam_pairs = list(combinations(all_cam_ids, 2))
        for cam_a, cam_b in cam_pairs:
            self._match_pair(
                per_cam_tracks[cam_a],
                per_cam_tracks[cam_b],
            )

        # --- Step 2: Collect and group by t_global_id ---
        all_tracks = []
        for cam_id in all_cam_ids:
            for track in per_cam_tracks[cam_id]:
                feat = track.get_representative_feature()
                if feat is None:
                    continue
                if track.location is None:
                    continue
                all_tracks.append(track)

        if not all_tracks:
            return np.array([], dtype=object)

        # Group by t_global_id
        id_to_tracks: Dict[int, List[TrackInfo]] = {}
        for track in all_tracks:
            gid = track.t_global_id
            if gid not in id_to_tracks:
                id_to_tracks[gid] = []
            id_to_tracks[gid].append(track)

        groups = []
        for gid in sorted(id_to_tracks.keys()):
            tracks_in_group = id_to_tracks[gid]
            features = [t.get_representative_feature() for t in tracks_in_group]
            centroid = np.mean([t.location for t in tracks_in_group], axis=0)
            keypoints_list = [t.keypoints for t in tracks_in_group]
            coords = [t.location for t in tracks_in_group]

            groups.append([gid, features, centroid, keypoints_list, coords])

        return np.array(groups, dtype=object)

    def _match_pair(
        self,
        tracks_a: List[TrackInfo],
        tracks_b: List[TrackInfo],
    ):
        """Match tracks from two cameras and unify t_global_ids.

        Modifies ``track.t_global_id`` in-place.
        """
        # Filter by pose quality
        a_features, a_locations, a_pose = [], [], []
        a_map = {}  # filtered index → original index
        for i, track in enumerate(tracks_a):
            if not self._passes_quality(track):
                continue
            feat = track.get_representative_feature()
            if feat is None or track.location is None:
                continue
            idx = len(a_features)
            a_features.append(feat)
            a_locations.append(track.location)
            a_pose.append(
                _pose_check(track.keypoints) if track.keypoints is not None
                else np.zeros(5, dtype=bool)
            )
            a_map[idx] = i

        b_features, b_locations, b_pose = [], [], []
        b_map = {}
        for i, track in enumerate(tracks_b):
            if not self._passes_quality(track):
                continue
            feat = track.get_representative_feature()
            if feat is None or track.location is None:
                continue
            idx = len(b_features)
            b_features.append(feat)
            b_locations.append(track.location)
            b_pose.append(
                _pose_check(track.keypoints) if track.keypoints is not None
                else np.zeros(5, dtype=bool)
            )
            b_map[idx] = i

        if not a_features or not b_features:
            return

        euc_dists = euclidean_distance(a_locations, b_locations)
        emb_dists = embedding_distance(a_features, b_features) / 2.0

        if emb_dists.size == 0:
            return

        # Combined cost with pose-weighted ratio
        n_a, n_b = len(a_features), len(b_features)
        dists = np.zeros((n_a, n_b), dtype=np.float64)

        emb_min, emb_max = np.min(emb_dists), np.max(emb_dists)
        euc_min, euc_max = np.min(euc_dists), np.max(euc_dists)

        emb_range = emb_max - emb_min if emb_max > emb_min else 1.0
        euc_range = euc_max - euc_min if euc_max > euc_min else 1.0

        norm_emb = (emb_dists - emb_min) / emb_range
        norm_euc = (euc_dists - euc_min) / euc_range

        for i in range(n_a):
            for j in range(n_b):
                # Pose overlap ratio (how many body parts are visible in both)
                ratio = float(np.sum(a_pose[i].astype(float) * b_pose[j].astype(float))) / 10.0
                dists[i, j] = (1 - ratio) * norm_euc[i, j] + ratio * norm_emb[i, j]

        # Hard gates
        dists[emb_dists > self.emb_thresh] = 1.0
        dists[euc_dists > self.euc_thresh] = 1.0

        matches, _, _ = linear_assignment(dists, thresh=0.999)

        # Unify t_global_ids for matched pairs
        for id_a, id_b in matches:
            orig_a = a_map[id_a]
            orig_b = b_map[id_b]
            track_a = tracks_a[orig_a]
            track_b = tracks_b[orig_b]

            a_gid = track_a.t_global_id
            b_gid = track_b.t_global_id
            matched_id = min(a_gid, b_gid)

            # Check if the matched_id is already assigned to someone else
            # in the same camera (constraint: one ID per camera)
            all_a_gids = [t.t_global_id for t in tracks_a]
            all_b_gids = [t.t_global_id for t in tracks_b]
            all_gids = all_a_gids + all_b_gids
            if all_gids.count(matched_id) > 1:
                continue

            # Handle re-match with better distance
            cur_dist = euc_dists[id_a, id_b] * emb_dists[id_a, id_b]
            if track_a._matched_dist is not None and a_gid > b_gid:
                if cur_dist > track_a._matched_dist:
                    continue
            elif track_b._matched_dist is not None and a_gid < b_gid:
                if cur_dist > track_b._matched_dist:
                    continue

            track_a.t_global_id = matched_id
            track_b.t_global_id = matched_id
            track_a._matched_dist = cur_dist
            track_b._matched_dist = cur_dist

    def _passes_quality(self, track: TrackInfo) -> bool:
        """Check if track has sufficient pose quality for cross-camera matching."""
        kpts = track.keypoints
        if kpts is None:
            return False

        # Height/width ratio check
        bbox = track.bbox
        x1, y1, x2, y2 = bbox[:4]
        w, h = x2 - x1, y2 - y1
        if w <= 0:
            return False
        if h / w < self.hw_thresh:
            return False

        # Keypoint visibility check
        has_heads, has_points, _ = _pose_check_all(kpts)
        if np.sum(has_points) <= self.min_visible_keypoints:
            return False
        if self.min_head_keypoints >= 0 and np.sum(has_heads) <= self.min_head_keypoints:
            return False

        return True

    def update_using_cluster_tracker(
        self,
        per_cam_tracks: Dict[int, List[TrackInfo]],
        cluster_tracker,
    ):
        """Push global IDs from MCTracker back to per-camera tracks.

        Adapted from AIC2024's ``Clustering.update_using_mctracker()``.

        For each camera's tracks:
        1. First: match high-quality tracks against cluster tracker's
           tracked + recently-lost clusters using embedding distance.
        2. Second: match remaining tracks using euclidean distance only.
        """
        # Collect active + recently-lost clusters
        mtrack_pool = []
        for track in cluster_tracker.tracked_mtracks:
            if track.is_activated:
                mtrack_pool.append(track)
        for track in cluster_tracker.lost_mtracks:
            if cluster_tracker.frame_id - track.end_frame <= 15:
                mtrack_pool.append(track)

        if not mtrack_pool:
            return

        for cam_id, tracks in per_cam_tracks.items():
            # Split into high/low pose quality
            high_tracks = []
            low_tracks = []
            for track in tracks:
                if track.keypoints is not None and float(np.sum(_pose_check(track.keypoints))) >= 4:
                    high_tracks.append(track)
                else:
                    low_tracks.append(track)

            # --- First association: embedding distance ---
            sct_features = [t.get_representative_feature() for t in high_tracks
                            if t.get_representative_feature() is not None]
            sct_locations = [t.location for t in high_tracks
                             if t.location is not None]

            valid_high = [t for t in high_tracks
                          if t.get_representative_feature() is not None
                          and t.location is not None]

            mct_features_flat = []
            length_mcts = []
            mct_centroids = []
            for mt in mtrack_pool:
                feats = list(mt.features)
                mct_features_flat.extend(feats)
                length_mcts.append(len(feats))
                mct_centroids.append(mt.centroid)

            sct_feats_list = [t.get_representative_feature() for t in valid_high]

            shape = (len(sct_feats_list), len(length_mcts))
            u_scts = np.arange(len(sct_feats_list))

            if shape[0] > 0 and shape[1] > 0 and mct_features_flat:
                emb_dists_flat = embedding_distance(sct_feats_list, mct_features_flat) / 2.0
                emb_dists = group_distances(
                    emb_dists_flat,
                    [1] * len(sct_feats_list),
                    length_mcts,
                    shape,
                )
                euc_dists = euclidean_distance(
                    [t.location for t in valid_high],
                    mct_centroids,
                )

                emb_min, emb_max = np.min(emb_dists), np.max(emb_dists)
                euc_min, euc_max = np.min(euc_dists), np.max(euc_dists)

                emb_range = emb_max - emb_min if emb_max > emb_min else 1.0
                euc_range = euc_max - euc_min if euc_max > euc_min else 1.0

                norm_emb = (emb_dists - emb_min) / emb_range
                norm_euc = (euc_dists - euc_min) / euc_range

                dists = 0.5 * norm_euc + 0.5 * norm_emb

                matches, u_scts, u_mcts = linear_assignment(dists, thresh=0.999)
                for isct, imct in matches:
                    valid_high[isct].person_id = mtrack_pool[imct].track_id

            # --- Second association: euclidean only for remaining ---
            left_tracks = low_tracks + [valid_high[i] for i in u_scts]
            left_mtracks = [mtrack_pool[i] for i in (u_mcts if 'u_mcts' in dir() else range(len(mtrack_pool)))]

            if left_tracks and left_mtracks:
                sct_locs = [t.location for t in left_tracks if t.location is not None]
                valid_left = [t for t in left_tracks if t.location is not None]
                mct_locs = [mt.centroid for mt in left_mtracks]

                if sct_locs and mct_locs:
                    euc_dists2 = euclidean_distance(sct_locs, mct_locs)
                    matches2, u_left, _ = linear_assignment(euc_dists2, thresh=2.0)
                    for isct, imct in matches2:
                        valid_left[isct].person_id = left_mtracks[imct].track_id
                    # Unmatched get -2 (no global ID)
                    for i in u_left:
                        if valid_left[i].person_id is None or valid_left[i].person_id < 0:
                            valid_left[i].person_id = -2
