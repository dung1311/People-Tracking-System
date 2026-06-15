"""Pose-aware perspective projection to world ground plane.

Adapted from AIC2024_Track1_Nota/perspective_transform/model.py.
Uses COCO 17-keypoint format instead of CrowdPose 14-keypoint format.

For partially visible persons (e.g. only upper body), the foot position
is estimated by extrapolating from visible keypoints using learned ratios
between body-part distances and full body height.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from modules.data_templates.mct_template import CameraCalibration
from modules.data_templates.sct_template import TrackInfo

logger = logging.getLogger(__name__)

# COCO 17-keypoint indices
_NOSE = 0
_LEFT_EYE = 1
_RIGHT_EYE = 2
_LEFT_EAR = 3
_RIGHT_EAR = 4
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

# Mapping from COCO keypoint name → index
_COCO_KEYPOINTS = {
    "nose": _NOSE, "left_eye": _LEFT_EYE, "right_eye": _RIGHT_EYE,
    "left_ear": _LEFT_EAR, "right_ear": _RIGHT_EAR,
    "left_shoulder": _LEFT_SHOULDER, "right_shoulder": _RIGHT_SHOULDER,
    "left_elbow": _LEFT_ELBOW, "right_elbow": _RIGHT_ELBOW,
    "left_wrist": _LEFT_WRIST, "right_wrist": _RIGHT_WRIST,
    "left_hip": _LEFT_HIP, "right_hip": _RIGHT_HIP,
    "left_knee": _LEFT_KNEE, "right_knee": _RIGHT_KNEE,
    "left_ankle": _LEFT_ANKLE, "right_ankle": _RIGHT_ANKLE,
}


class PerspectiveProjector:
    """Projects foot points from image coordinates to the world ground plane.

    Uses pose keypoints to estimate foot position accurately, even when
    the lower body is not visible (e.g. person behind a counter).
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.pose_thr: float = config.get("pose_thr", 0.3)
        self.ratio_smooth: float = config.get("ratio_smooth", 0.9)

        # Height-to-body-part ratios (learned/adapted from AIC2024)
        # [h/w, h/(neck-nose), h/(hip-nose), h/(knee-nose),
        #  h/(neck-nose), h/(shoulder-nose), h/(elbow-nose)]
        self.ratio = np.array([3.5, 7.0, 2.0, 1.6, 7.0, 5.0, 3.0])

    def update_ratio(self, tracks: List[TrackInfo]):
        """Update body-part ratios from tracks with full-body visibility."""
        new_ratio = np.zeros(7)
        count = 0

        for track in tracks:
            kpts = track.keypoints
            if kpts is None or kpts.shape[0] < 17:
                continue

            # Check if all keypoints are visible with high confidence
            if np.sum(kpts[:, 2] > 0.8) < 14:
                continue

            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue

            nose_y = kpts[_NOSE, 1]

            # Compute ratios (similar to AIC2024's all_good_pose_bbox)
            try:
                neck_y = (kpts[_LEFT_SHOULDER, 1] + kpts[_RIGHT_SHOULDER, 1]) / 2
                new_ratio[0] += h / w
                new_ratio[1] += h / max(neck_y - nose_y, 1e-6)
                hip_y = (kpts[_LEFT_HIP, 1] + kpts[_RIGHT_HIP, 1] +
                         kpts[_LEFT_WRIST, 1] + kpts[_RIGHT_WRIST, 1]) / 4
                new_ratio[2] += h / max(hip_y - nose_y, 1e-6)
                knee_y = (kpts[_LEFT_KNEE, 1] + kpts[_RIGHT_KNEE, 1]) / 2
                new_ratio[3] += h / max(knee_y - nose_y, 1e-6)
                new_ratio[4] += h / max(neck_y - nose_y, 1e-6)
                shoulder_y = (kpts[_LEFT_SHOULDER, 1] + kpts[_RIGHT_SHOULDER, 1]) / 2
                new_ratio[5] += h / max(shoulder_y - nose_y, 1e-6)
                elbow_y = (kpts[_LEFT_ELBOW, 1] + kpts[_RIGHT_ELBOW, 1]) / 2
                new_ratio[6] += h / max(elbow_y - nose_y, 1e-6)
                count += 1
            except (IndexError, ZeroDivisionError):
                continue

        if count > 0:
            new_ratio /= count
            # Check for inf/negative
            if not (np.any(np.isinf(new_ratio)) or np.any(new_ratio < 0)):
                self.ratio = self.ratio * self.ratio_smooth + new_ratio * (1 - self.ratio_smooth)

    def estimate_foot_point(
        self,
        bbox: List[float],
        keypoints: Optional[np.ndarray],
    ) -> np.ndarray:
        """Estimate the foot position in image coordinates.

        Args:
            bbox: ``[x1, y1, x2, y2]`` bounding box.
            keypoints: ``(17, 3)`` COCO keypoints with ``(x, y, score)``
                       or ``None`` if pose estimation was not run.

        Returns:
            ``(2,)`` foot point in image coordinates ``(u, v)``.
        """
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1

        if keypoints is None or keypoints.shape[0] < 17:
            # Fallback: bbox bottom center, extrapolated
            foot_x = (x1 + x2) / 2
            foot_y = y1 + w * self.ratio[0]
            return np.array([foot_x, foot_y])

        kpts = keypoints
        keys = {}
        for name, idx in _COCO_KEYPOINTS.items():
            keys[name] = kpts[idx]  # (x, y, score)

        # Find the lowest visible keypoint (closest to bottom of bbox)
        min_dist_to_bottom = float("inf")
        min_key = ""
        max_dist_to_top = float("inf")
        max_key = ""

        for name, kpt in keys.items():
            if kpt[2] > self.pose_thr:
                d_bottom = abs(kpt[1] - y2)
                if d_bottom < min_dist_to_bottom:
                    min_dist_to_bottom = d_bottom
                    min_key = name
                d_top = abs(kpt[1] - y1)
                if d_top < max_dist_to_top:
                    max_dist_to_top = d_top
                    max_key = name

        foot_x = (x1 + x2) / 2

        if min_key == "":
            # No visible keypoint — use bbox-based estimation
            foot_y = y1 + w * self.ratio[0]

        elif "ankle" in min_key:
            # Ankle visible → foot is at bbox bottom
            foot_y = y2

        elif "nose" in min_key or "eye" in min_key or "ear" in min_key:
            # Only head visible → large extrapolation
            foot_y = y2 + h * self.ratio[1]

        elif (keys[min_key][1] + h / 4) < y2:
            # Lowest keypoint is well above bottom — person is cut off
            nose_y = keys.get("nose", kpts[_NOSE])[1]
            if "nose" in max_key or "eye" in max_key or "ear" in max_key:
                key_gap = abs(nose_y - keys[min_key][1])
                if "hip" in min_key or "wrist" in min_key:
                    foot_y = y1 + key_gap * self.ratio[2]
                elif "knee" in min_key:
                    foot_y = y1 + key_gap * self.ratio[3]
                elif "shoulder" in min_key:
                    foot_y = y1 + key_gap * self.ratio[5]
                elif "elbow" in min_key:
                    foot_y = y1 + key_gap * self.ratio[6]
                else:
                    foot_y = y2
            else:
                foot_y = y2

        else:
            # Lowest keypoint near bottom
            if "hip" in min_key or "wrist" in min_key:
                foot_y = y2 + h * (self.ratio[2] - 1.0)
            elif "knee" in min_key:
                foot_y = y2 + h * (self.ratio[3] - 1.0)
            elif "shoulder" in min_key:
                foot_y = y2 + h * (self.ratio[5] - 1.0)
            elif "elbow" in min_key:
                foot_y = y2 + h * (self.ratio[6] - 1.0)
            else:
                foot_y = y2

        return np.array([foot_x, foot_y])

    def project_to_world(
        self,
        calibration: CameraCalibration,
        foot_point: np.ndarray,
    ) -> np.ndarray:
        """Project foot point from image to world ground plane.

        Args:
            calibration: camera calibration with homography.
            foot_point: ``(2,)`` image coordinate.

        Returns:
            ``(2,)`` world-plane coordinate.
        """
        return calibration.project_to_world(foot_point)

    def compute_locations(
        self,
        tracks: List[TrackInfo],
        calibration: CameraCalibration,
    ):
        """Compute and set ``track.location`` for each track in-place.

        Also updates the internal body-part ratios using full-body tracks.

        Args:
            tracks: list of tracks with ``bbox`` and optionally ``keypoints``.
            calibration: camera calibration.
        """
        self.update_ratio(tracks)

        for track in tracks:
            foot = self.estimate_foot_point(track.bbox, track.keypoints)
            track.location = self.project_to_world(calibration, foot)
