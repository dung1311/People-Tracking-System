"""Vectorised distance and homography helpers for cross-camera matching."""

from __future__ import annotations

from typing import Dict

import numpy as np


def project_feet_to_world(
    feet: np.ndarray,
    cam_ids: np.ndarray,
    H_invs: Dict[int, np.ndarray],
) -> np.ndarray:
    """Map pixel foot points to world plane using per-camera ``H_inv``.

    Args:
        feet: ``(N, 2)`` pixel coords (e.g. bbox bottom-centre).
        cam_ids: ``(N,)`` integer camera id per row.
        H_invs: ``{cam_id: (3, 3)}`` inverse homography (image → world plane).

    Returns:
        ``(N, 2)`` world coordinates.
    """
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


def cosine_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``(N, D), (M, D)`` L2-normalised rows → ``(N, M)`` cosine distance ``1 - cos``."""
    sim = A @ B.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def euclidean_distance_matrix(pts: np.ndarray) -> np.ndarray:
    """``(N, 2)`` → ``(N, N)`` pairwise Euclidean distances."""
    sq = np.sum(pts ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def l2_normalise_rows(feats: np.ndarray) -> np.ndarray:
    """L2-normalise each row of ``(N, D)`` features."""
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return feats / norms
