from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera calibration
# ---------------------------------------------------------------------------

class CameraCalibration:
    """Lightweight wrapper around a single camera's projection / homography."""

    def __init__(self, cam_id: int, P: np.ndarray, H: np.ndarray):
        self.cam_id = cam_id
        self.P = np.asarray(P, dtype=np.float64)        # 3x4
        self.H = np.asarray(H, dtype=np.float64)         # 3x3  (world -> image)
        self.H_inv = np.linalg.inv(self.H)               # 3x3  (image -> world)

    @classmethod
    def load_from_json(cls, path: str, cam_id: int) -> "CameraCalibration":
        with open(path) as f:
            data = json.load(f)
        P = np.array(data["camera projection matrix"], dtype=np.float64)
        H = np.array(data["homography matrix"], dtype=np.float64)
        return cls(cam_id=cam_id, P=P, H=H)

    def project_to_world(self, image_point: np.ndarray) -> np.ndarray:
        """Project a 2-D image point to the world ground plane via H_inv.

        Args:
            image_point: shape ``(2,)`` -- ``(u, v)`` in pixel coords.

        Returns:
            ``(x, y)`` on the world ground plane.
        """
        pt_h = np.array([image_point[0], image_point[1], 1.0])
        world_h = self.H_inv @ pt_h
        world_h /= world_h[2]
        return world_h[:2]


class CameraPairCalibration:
    """Pre-computed geometric relations between two cameras."""

    def __init__(self, cal_i: CameraCalibration, cal_j: CameraCalibration):
        from modules.pose_3d.geometry.stereo import get_fundamental_matrix

        self.cal_i = cal_i
        self.cal_j = cal_j
        self.F = get_fundamental_matrix(cal_i.P, cal_j.P)

    @property
    def key(self) -> Tuple[int, int]:
        return (self.cal_i.cam_id, self.cal_j.cam_id)


# ---------------------------------------------------------------------------
# Global track state
# ---------------------------------------------------------------------------

class GlobalTrackState(Enum):
    ACTIVE = 0
    LOST = 1
    DEAD = 2


# ---------------------------------------------------------------------------
# Global track info
# ---------------------------------------------------------------------------

class GlobalTrackInfo:
    """A person tracked across multiple cameras."""

    def __init__(self, global_id: int, frame_id: int):
        self.global_id = global_id
        self.state = GlobalTrackState.ACTIVE
        self.start_frame = frame_id
        self.last_seen_frame = frame_id
        self.lost_age = 0

        # cam_id -> local person_id (current mapping)
        self.cam_tracks: Dict[int, int] = {}

        # Accumulated world-plane positions (from homography projection)
        self.world_trajectory: List[np.ndarray] = []

        # Accumulated visual features (L2-normalised)
        self.features: List[np.ndarray] = []

    # -- feature helpers ---------------------------------------------------

    def add_feature(self, feat: np.ndarray, max_features: int = 50):
        if feat is None:
            return
        self.features.append(feat)
        if len(self.features) > max_features:
            self.features = self.features[-max_features:]

    def get_representative_feature(self) -> Optional[np.ndarray]:
        if not self.features:
            return None
        avg = np.mean(self.features, axis=0)
        norm = np.linalg.norm(avg)
        return avg if norm == 0 else avg / norm

    # -- trajectory helpers ------------------------------------------------

    def add_world_point(self, pt: np.ndarray, max_len: int = 100):
        self.world_trajectory.append(pt)
        if len(self.world_trajectory) > max_len:
            self.world_trajectory = self.world_trajectory[-max_len:]


# ---------------------------------------------------------------------------
# Match result for cross-camera matching
# ---------------------------------------------------------------------------

@dataclass
class CrossCameraMatchResult:
    cam_i: int
    cam_j: int
    local_pid_i: int
    local_pid_j: int
    epipolar_cost: float
    homography_cost: float
    visual_cost: float
    frechet_cost: float
    combined_cost: float
    is_matched: bool
