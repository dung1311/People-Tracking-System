"""Cross-camera matching using epipolar, homography, visual, and Fréchet cues."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from modules.data_templates.mct_template import (
    CameraCalibration,
    CameraPairCalibration,
    CrossCameraMatchResult,
    GlobalTrackInfo,
)
from modules.data_templates.sct_template import TrackInfo

logger = logging.getLogger(__name__)

# Default weights / thresholds -- will be overridden from MCT_CONFIG
_DEFAULT_WEIGHTS = {
    "epipolar": 0.25,
    "homography": 0.25,
    "visual": 0.35,
    "frechet": 0.15,
}
_DEFAULT_THRESHOLDS = {
    "epipolar": 100.0,
    "homography": 200.0,
    "visual": 0.5,
    "frechet": 300.0,
    "combined": 0.7,
}


# -----------------------------------------------------------------------
# Individual cost functions
# -----------------------------------------------------------------------

def _bbox_foot_point(bbox: List[float]) -> np.ndarray:
    """Bottom-centre of a bbox ``[x1, y1, x2, y2]``."""
    return np.array([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=np.float64)


def compute_epipolar_cost(
    F: np.ndarray,
    bbox_i: List[float],
    bbox_j: List[float],
) -> float:
    """Mean symmetric epipolar distance between bbox foot points."""
    pt_i = _bbox_foot_point(bbox_i).reshape(1, 1, 2).astype(np.float64)
    pt_j = _bbox_foot_point(bbox_j).reshape(1, 1, 2).astype(np.float64)

    lines_i_to_j = cv2.computeCorrespondEpilines(pt_i, 1, F).reshape(-1, 3)
    lines_j_to_i = cv2.computeCorrespondEpilines(pt_j, 2, F).reshape(-1, 3)

    pt_j_flat = pt_j.reshape(2)
    pt_i_flat = pt_i.reshape(2)

    a, b, c = lines_i_to_j[0]
    d1 = abs(a * pt_j_flat[0] + b * pt_j_flat[1] + c) / (np.sqrt(a**2 + b**2) + 1e-8)

    a, b, c = lines_j_to_i[0]
    d2 = abs(a * pt_i_flat[0] + b * pt_i_flat[1] + c) / (np.sqrt(a**2 + b**2) + 1e-8)

    return float((d1 + d2) / 2.0)


def compute_homography_cost(
    cal_i: CameraCalibration,
    cal_j: CameraCalibration,
    bbox_i: List[float],
    bbox_j: List[float],
) -> float:
    """Euclidean distance on the world ground plane between two projected foot points."""
    pt_i_world = cal_i.project_to_world(_bbox_foot_point(bbox_i))
    pt_j_world = cal_j.project_to_world(_bbox_foot_point(bbox_j))
    return float(np.linalg.norm(pt_i_world - pt_j_world))


def compute_visual_cost(
    feat_i: Optional[np.ndarray],
    feat_j: Optional[np.ndarray],
) -> float:
    """Cosine distance between two L2-normalised feature vectors."""
    if feat_i is None or feat_j is None:
        return 1.0
    sim = np.dot(feat_i, feat_j) / (np.linalg.norm(feat_i) * np.linalg.norm(feat_j) + 1e-8)
    return float(1.0 - sim)


def _discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
    """Discrete Fréchet distance between two 2-D trajectories (DP).

    Args:
        P: shape ``(n, 2)``
        Q: shape ``(m, 2)``
    """
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return 0.0

    ca = np.full((n, m), -1.0)

    def _d(i: int, j: int) -> float:
        return float(np.linalg.norm(P[i] - Q[j]))

    def _recurse(i: int, j: int) -> float:
        if ca[i, j] >= 0:
            return ca[i, j]
        d = _d(i, j)
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(_recurse(0, j - 1), d)
        elif j == 0:
            ca[i, j] = max(_recurse(i - 1, 0), d)
        else:
            ca[i, j] = max(
                min(_recurse(i - 1, j), _recurse(i - 1, j - 1), _recurse(i, j - 1)),
                d,
            )
        return ca[i, j]

    # Iterative version to avoid recursion limit on long trajectories
    for i in range(n):
        for j in range(m):
            d = _d(i, j)
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[0, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)

    return float(ca[n - 1, m - 1])


def compute_frechet_cost(
    traj_i: List[np.ndarray],
    traj_j: List[np.ndarray],
) -> float:
    """Discrete Fréchet distance between two world-plane trajectories."""
    if len(traj_i) < 2 or len(traj_j) < 2:
        return 0.0
    P = np.array(traj_i)
    Q = np.array(traj_j)
    return _discrete_frechet(P, Q)


# -----------------------------------------------------------------------
# Normalisation helpers
# -----------------------------------------------------------------------

def _normalise(value: float, threshold: float) -> float:
    """Map *value* to [0, 1] range clamped by *threshold*."""
    if threshold <= 0:
        return 0.0
    return min(value / threshold, 1.0)


# -----------------------------------------------------------------------
# Main matcher
# -----------------------------------------------------------------------

class CrossCameraMatcher:
    """Matches tracks across a pair of cameras."""

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        thresholds: Dict[str, float] | None = None,
    ):
        self.weights = weights or dict(_DEFAULT_WEIGHTS)
        self.thresholds = thresholds or dict(_DEFAULT_THRESHOLDS)

    def match(
        self,
        tracks_i: List[TrackInfo],
        tracks_j: List[TrackInfo],
        pair_cal: CameraPairCalibration,
        global_tracks: Dict[int, GlobalTrackInfo] | None = None,
    ) -> List[CrossCameraMatchResult]:
        """Build cost matrix and run Hungarian matching.

        Args:
            tracks_i: active tracks from camera *i*.
            tracks_j: active tracks from camera *j*.
            pair_cal: pre-computed calibration for the camera pair.
            global_tracks: optional existing global tracks (used for Fréchet).

        Returns:
            List of ``CrossCameraMatchResult`` for every matched pair (and
            unmatched entries marked with ``is_matched=False``).
        """
        if not tracks_i or not tracks_j:
            return []

        n_i, n_j = len(tracks_i), len(tracks_j)
        cost_matrix = np.full((n_i, n_j), 1e9)
        detail = {}  # (i, j) -> individual costs

        w = self.weights
        th = self.thresholds

        for i, ti in enumerate(tracks_i):
            feat_i = ti.get_representative_feature()
            traj_i = self._get_trajectory(ti, global_tracks)

            for j, tj in enumerate(tracks_j):
                feat_j = tj.get_representative_feature()
                traj_j = self._get_trajectory(tj, global_tracks)

                epi = compute_epipolar_cost(pair_cal.F, ti.bbox, tj.bbox)
                homo = compute_homography_cost(
                    pair_cal.cal_i, pair_cal.cal_j, ti.bbox, tj.bbox
                )
                vis = compute_visual_cost(feat_i, feat_j)
                fre = compute_frechet_cost(traj_i, traj_j)

                # Normalise each cost to [0, 1]
                epi_n = _normalise(epi, th["epipolar"])
                homo_n = _normalise(homo, th["homography"])
                vis_n = _normalise(vis, 1.0)  # already in [0, 1]
                fre_n = _normalise(fre, th["frechet"])

                combined = (
                    w["epipolar"] * epi_n
                    + w["homography"] * homo_n
                    + w["visual"] * vis_n
                    + w["frechet"] * fre_n
                )

                cost_matrix[i, j] = combined
                detail[(i, j)] = (epi, homo, vis, fre, combined)

        # Hungarian assignment
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        results: List[CrossCameraMatchResult] = []
        matched_i, matched_j = set(), set()

        for ri, ci in zip(row_idx, col_idx):
            epi, homo, vis, fre, combined = detail[(ri, ci)]
            matched = combined < th["combined"]
            results.append(
                CrossCameraMatchResult(
                    cam_i=pair_cal.cal_i.cam_id,
                    cam_j=pair_cal.cal_j.cam_id,
                    local_pid_i=tracks_i[ri].person_id,
                    local_pid_j=tracks_j[ci].person_id,
                    epipolar_cost=epi,
                    homography_cost=homo,
                    visual_cost=vis,
                    frechet_cost=fre,
                    combined_cost=combined,
                    is_matched=matched,
                )
            )
            if matched:
                matched_i.add(ri)
                matched_j.add(ci)

        return results

    # ------------------------------------------------------------------

    @staticmethod
    def _get_trajectory(
        track: TrackInfo,
        global_tracks: Dict[int, GlobalTrackInfo] | None,
    ) -> List[np.ndarray]:
        """Retrieve the world-plane trajectory for a local track."""
        if global_tracks is None:
            return []
        for gt in global_tracks.values():
            if track.cam_id in gt.cam_tracks and gt.cam_tracks[track.cam_id] == track.person_id:
                return gt.world_trajectory
        return []
