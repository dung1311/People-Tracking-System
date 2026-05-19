from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from mct.calibration import CameraCalibration, CameraPairCalibration

logger = logging.getLogger(__name__)


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
