from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
from datetime import datetime

import numpy as np


class TrackState(Enum):
    UNCONFIRM = 0
    ACTIVE = 1
    LOST = 2
    DEAD = 3
    CHANGED = 4


class TrackInfo:
    """Pure data container for a single tracked person."""

    def __init__(
        self,
        tracker_id: int,
        bbox: List[float],
        score: float = None,
        class_id: int = None,
        feat: np.ndarray = None,
        cam_id=None,
        frame_id: int = None,
    ):
        self.tracker_id = tracker_id
        self.person_id: Optional[int] = None
        self.cam_id = cam_id
        self.frame_id = frame_id
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.timestamp = datetime.now()

        if feat is not None:
            if hasattr(feat, "detach"):
                feat = feat.detach().cpu().numpy()
            self.features: List[np.ndarray] = [feat]
        else:
            self.features: List[np.ndarray] = []

        self.state = TrackState.UNCONFIRM
        self.lost_age = 0
        self.hits = 1

    def update(
        self,
        bbox: List[float],
        score: float,
        class_id: int,
        feature: np.ndarray,
        frame_id: int,
        smooth_factor: float = 0.1,
        appearance_threshold: float = 0.5,
    ):
        """Update track with new observation.

        If appearance diverges beyond *appearance_threshold* (cosine distance),
        the state is set to CHANGED and features are left untouched so that the
        caller can decide what to do (e.g. trigger Re-ID).

        Otherwise the feature is EMA-smoothed and state becomes ACTIVE.
        """
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.frame_id = frame_id
        self.timestamp = datetime.now()
        self.lost_age = 0

        if hasattr(feature, "detach"):
            feature = feature.detach().cpu().numpy()

        curr = self.get_representative_feature()
        if curr is not None:
            cos_dist = 1 - np.dot(curr, feature) / (
                np.linalg.norm(curr) * np.linalg.norm(feature) + 1e-8
            )
            if cos_dist > appearance_threshold:
                self.state = TrackState.CHANGED
                return

        smoothed = (
            feature
            if curr is None
            else (1.0 - smooth_factor) * curr + smooth_factor * feature
        )
        norm = np.linalg.norm(smoothed)
        if norm > 0:
            smoothed = smoothed / norm
        self.features = [smoothed]
        self.state = TrackState.ACTIVE

    def get_representative_feature(self) -> Optional[np.ndarray]:
        """Return L2-normalised mean of stored features, or None."""
        if not self.features:
            return None
        avg = np.mean(self.features, axis=0)
        norm = np.linalg.norm(avg)
        return avg if norm == 0 else avg / norm


@dataclass
class MatchResult:
    query_idx: int
    gallery_idx: int
    distance: float
    is_matched: bool
    min_distance: float = None
