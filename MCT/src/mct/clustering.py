"""Cross-camera track clustering: homography + visual cost, Union-Find."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from mct.distance import (
    cosine_distance_matrix,
    euclidean_distance_matrix,
    l2_normalise_rows,
    project_feet_to_world,
)
from mct.union_find import UnionFind

logger = logging.getLogger(__name__)


class CrossCameraClusterer:
    """Stateless: build cost matrix and cluster with Union-Find + same-cam constraint."""

    def __init__(self, config: Dict):
        weights = config["weights"]
        thresholds = config["thresholds"]

        self.w_homo: float = weights["homography"]
        self.w_vis: float = weights["visual"]

        self.th_homo: float = thresholds["homography"]
        self.th_vis_gate: float = thresholds.get("visual_gate", 1.0)
        self.th_homo_gate: float = thresholds.get("homo_gate", float("inf"))
        self.th_combined: float = thresholds["combined"]

    def cluster(
        self,
        cam_ids: np.ndarray,
        foot_points: np.ndarray,
        features: np.ndarray,
        H_invs: Dict[int, np.ndarray],
    ) -> List[List[int]]:
        N = len(cam_ids)
        if N <= 1:
            return [[i] for i in range(N)]

        world = project_feet_to_world(foot_points, cam_ids, H_invs)
        cost = self._build_cost_matrix(cam_ids, world, features)

        return self._union_find_cluster(cost, cam_ids, N)

    @staticmethod
    def cosine_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return cosine_distance_matrix(A, B)

    @staticmethod
    def euclidean_dist(pts: np.ndarray) -> np.ndarray:
        return euclidean_distance_matrix(pts)

    @staticmethod
    def l2_normalise(feats: np.ndarray) -> np.ndarray:
        return l2_normalise_rows(feats)

    @staticmethod
    def _project_feet(
        feet: np.ndarray, cam_ids: np.ndarray, H_invs: Dict[int, np.ndarray],
    ) -> np.ndarray:
        return project_feet_to_world(feet, cam_ids, H_invs)

    def _build_cost_matrix(
        self,
        cam_ids: np.ndarray,
        world: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        homo_dist = euclidean_distance_matrix(world)
        vis_dist = cosine_distance_matrix(features, features)
        homo_norm = np.minimum(homo_dist / self.th_homo, 1.0)

        cost = self.w_homo * homo_norm + self.w_vis * vis_dist

        cost[vis_dist > self.th_vis_gate] = np.inf
        cost[homo_dist > self.th_homo_gate] = np.inf
        cost[cam_ids[:, None] == cam_ids[None, :]] = np.inf

        return cost

    def _union_find_cluster(
        self, cost: np.ndarray, cam_ids: np.ndarray, N: int,
    ) -> List[List[int]]:
        ui, uj = np.triu_indices(N, k=1)
        ec = cost[ui, uj]
        valid = ec < self.th_combined
        if not valid.any():
            return [[i] for i in range(N)]

        ei, ej = ui[valid], uj[valid]
        order = np.argsort(ec[valid])
        ei, ej = ei[order], ej[order]

        uf = UnionFind(N)
        comp_cams: List[set] = [{int(cam_ids[i])} for i in range(N)]

        for a, b in zip(ei, ej):
            ra, rb = uf.find(int(a)), uf.find(int(b))
            if ra == rb:
                continue
            if comp_cams[ra] & comp_cams[rb]:
                continue
            merged = comp_cams[ra] | comp_cams[rb]
            uf.union(ra, rb)
            comp_cams[uf.find(ra)] = merged

        return uf.groups()
