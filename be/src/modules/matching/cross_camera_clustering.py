"""Cross-camera track clustering using homography + visual cost with Union-Find.

Build an (N x N) cost matrix across all tracks from all cameras.
Same-camera pairs are excluded. Per-metric hard gates filter obviously
wrong matches before the combined threshold is applied. Union-Find with
a same-camera constraint provides transitive closure of matched pairs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from modules.matching.union_find import UnionFind

logger = logging.getLogger(__name__)


class CrossCameraClusterer:
    """Stateless component: given tracks + calibrations, returns clusters."""

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
        """Cluster tracks across cameras.

        Args:
            cam_ids: ``(N,)`` camera ID per track.
            foot_points: ``(N, 2)`` bbox bottom-centre in pixel coords.
            features: ``(N, D)`` L2-normalised appearance features.
            H_invs: ``{cam_id: 3x3 H_inv}`` for world projection.

        Returns:
            List of clusters, each a list of track indices.
        """
        N = len(cam_ids)
        if N <= 1:
            return [[i] for i in range(N)]

        world = self._project_feet(foot_points, cam_ids, H_invs)
        cost = self._build_cost_matrix(cam_ids, world, features)

        return self._union_find_cluster(cost, cam_ids, N)

    # ------------------------------------------------------------------
    # Vectorised math (static, reusable)
    # ------------------------------------------------------------------

    @staticmethod
    def _project_feet(
        feet: np.ndarray, cam_ids: np.ndarray, H_invs: Dict[int, np.ndarray],
    ) -> np.ndarray:
        N = len(feet)
        pts_h = np.empty((N, 3), dtype=np.float64)
        pts_h[:, :2] = feet
        pts_h[:, 2] = 1.0
        world = np.empty((N, 2), dtype=np.float64)
        for cid, H_inv in H_invs.items():
            mask = cam_ids == cid
            if not mask.any():
                continue
            w = (H_inv @ pts_h[mask].T).T
            w /= w[:, 2:3]
            world[mask] = w[:, :2]
        return world

    @staticmethod
    def cosine_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """``(N, D), (M, D)`` both L2-normed → ``(N, M)`` cosine distance."""
        sim = A @ B.T
        np.clip(sim, -1.0, 1.0, out=sim)
        return 1.0 - sim

    @staticmethod
    def euclidean_dist(pts: np.ndarray) -> np.ndarray:
        """``(N, 2)`` → ``(N, N)`` Euclidean distance."""
        sq = np.sum(pts ** 2, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2)

    @staticmethod
    def l2_normalise(feats: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_cost_matrix(
        self,
        cam_ids: np.ndarray,
        world: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        homo_dist = self.euclidean_dist(world)
        vis_dist = self.cosine_dist(features, features)
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
