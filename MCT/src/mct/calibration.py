"""Camera calibration: homography and optional stereo pair (lazy numba)."""

from __future__ import annotations

import json
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CameraCalibration:
    """Single camera: projection matrix P and homography H (world ↔ image)."""

    def __init__(self, cam_id: int, P: np.ndarray, H: np.ndarray):
        self.cam_id = cam_id
        self.P = np.asarray(P, dtype=np.float64)
        self.H = np.asarray(H, dtype=np.float64)
        self.H_inv = np.linalg.inv(self.H)

    @classmethod
    def load_from_json(cls, path: str, cam_id: int) -> "CameraCalibration":
        with open(path) as f:
            data = json.load(f)
        P = np.array(data["camera projection matrix"], dtype=np.float64)
        H = np.array(data["homography matrix"], dtype=np.float64)
        return cls(cam_id=cam_id, P=P, H=H)

    def project_to_world(self, image_point: np.ndarray) -> np.ndarray:
        pt_h = np.array([image_point[0], image_point[1], 1.0])
        world_h = self.H_inv @ pt_h
        world_h /= world_h[2]
        return world_h[:2]


class CameraPairCalibration:
    """Fundamental matrix between two cameras (requires ``be`` on path for numba stereo)."""

    def __init__(self, cal_i: CameraCalibration, cal_j: CameraCalibration):
        from modules.pose_3d.geometry.stereo import get_fundamental_matrix

        self.cal_i = cal_i
        self.cal_j = cal_j
        self.F = get_fundamental_matrix(cal_i.P, cal_j.P)

    @property
    def key(self) -> Tuple[int, int]:
        return (self.cal_i.cam_id, self.cal_j.cam_id)
