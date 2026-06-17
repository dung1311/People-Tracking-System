from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class _Identity:
    person_id: int
    features: List[np.ndarray] = field(default_factory=list)
    state: str = "ACTIVE"
    last_frame: int = 0

    def representative(self) -> Optional[np.ndarray]:
        if not self.features:
            return None
        feat = np.mean(self.features, axis=0)
        norm = np.linalg.norm(feat)
        return feat if norm == 0 else feat / norm


@dataclass
class _TrackerState:
    hits: int = 0
    features: List[np.ndarray] = field(default_factory=list)
    person_id: Optional[int] = None

    def representative(self) -> Optional[np.ndarray]:
        if not self.features:
            return None
        feat = np.mean(self.features, axis=0)
        norm = np.linalg.norm(feat)
        return feat if norm == 0 else feat / norm


class TrackerIdentityManager:
    """Assign stable person IDs inside a 2D tracker using ReID features."""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", True) and config.get("EMBEDDING"))
        self.min_hits = int(config.get("min_hits", 3))
        self.max_features = int(config.get("max_features", 50))
        self.reid_threshold = float(config.get("distance_threshold", 0.5))
        self.swap_threshold = float(config.get("appearance_threshold", 0.5))
        self.smooth_factor = float(config.get("smooth_factor", 0.1))

        self.next_person_id = 1
        self.tracker_states: Dict[int, _TrackerState] = {}
        self.identities: Dict[int, _Identity] = {}

        self.embedder = None
        if self.enabled:
            from modules.embedder.factory import EmbedderFactory

            self.embedder = EmbedderFactory(config["EMBEDDING"]).get_embedder()

    def update_tracker(
        self,
        tracker,
        bbox,
        frame,
        frame_id: int,
    ) -> Optional[int]:
        """Update identity state for one matched tracker observation."""
        tracker_key = int(tracker.id)
        if not self.enabled or frame is None:
            pid = self._ensure_passthrough_id(tracker)
            tracker.person_id = pid
            return pid

        feature = self._extract_feature(frame, bbox)
        if feature is None:
            state = self.tracker_states.get(tracker_key)
            return state.person_id if state is not None else None

        state = self.tracker_states.setdefault(tracker_key, _TrackerState())
        state.hits += 1

        if state.person_id is not None:
            ident = self.identities.get(state.person_id)
            dist = self._distance(feature, ident.representative() if ident else None)
            if dist is not None and dist > self.swap_threshold:
                self.mark_lost_tracker(tracker)
                state = self.tracker_states.setdefault(tracker_key, _TrackerState())

        if state.person_id is None:
            self._append_feature(state.features, feature)
            if state.hits >= self.min_hits:
                pid = self._match_lost_or_new(state.representative(), frame_id)
                state.person_id = pid
                self._activate_identity(pid, state.features, frame_id)
        else:
            self._update_identity(state.person_id, feature, frame_id)

        tracker.person_id = state.person_id
        return state.person_id

    def mark_lost_tracker(self, tracker):
        tracker_key = int(tracker.id)
        state = self.tracker_states.pop(tracker_key, None)
        if state is None or state.person_id is None:
            return
        ident = self.identities.get(state.person_id)
        if ident is not None:
            ident.state = "LOST"

    def get_person_id(self, tracker) -> Optional[int]:
        if not self.enabled:
            return int(tracker.id) + 1
        state = self.tracker_states.get(int(tracker.id))
        return state.person_id if state is not None else None

    def get_feature(self, person_id: int) -> Optional[np.ndarray]:
        ident = self.identities.get(int(person_id))
        return ident.representative() if ident is not None else None

    def _ensure_passthrough_id(self, tracker) -> int:
        if not hasattr(tracker, "person_id") or tracker.person_id is None:
            tracker.person_id = int(tracker.id) + 1
        return int(tracker.person_id)

    def _extract_feature(self, frame, bbox) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, bbox[:4])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        feats = self.embedder.extract_feature([frame[y1:y2, x1:x2]])
        if feats is None:
            return None
        if hasattr(feats, "numel") and feats.numel() == 0:
            return None
        if not hasattr(feats, "numel") and len(feats) == 0:
            return None
        feat = feats[0]
        if hasattr(feat, "detach"):
            feat = feat.detach().cpu().numpy()
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feat)
        return feat if norm == 0 else feat / norm

    def _distance(self, feat_a: Optional[np.ndarray], feat_b: Optional[np.ndarray]) -> Optional[float]:
        if feat_a is None or feat_b is None:
            return None
        return float(1.0 - np.dot(feat_a, feat_b) / (
            np.linalg.norm(feat_a) * np.linalg.norm(feat_b) + 1e-8
        ))

    def _match_lost_or_new(self, feature: Optional[np.ndarray], frame_id: int) -> int:
        lost = [ident for ident in self.identities.values() if ident.state == "LOST"]
        if feature is None or not lost:
            return self._new_identity(feature, frame_id)

        dists = np.array([
            self._distance(feature, ident.representative()) or np.inf
            for ident in lost
        ], dtype=np.float64).reshape(1, -1)
        rows, cols = linear_sum_assignment(dists)
        if len(rows) > 0 and dists[rows[0], cols[0]] < self.reid_threshold:
            return lost[cols[0]].person_id
        return self._new_identity(feature, frame_id)

    def _new_identity(self, feature: Optional[np.ndarray], frame_id: int) -> int:
        pid = self.next_person_id
        self.next_person_id += 1
        self.identities[pid] = _Identity(pid, [], "ACTIVE", frame_id)
        if feature is not None:
            self._append_feature(self.identities[pid].features, feature)
        return pid

    def _activate_identity(self, person_id: int, features: List[np.ndarray], frame_id: int):
        ident = self.identities.get(person_id)
        if ident is None:
            ident = _Identity(person_id)
            self.identities[person_id] = ident
        ident.state = "ACTIVE"
        ident.last_frame = frame_id
        for feature in features:
            self._append_feature(ident.features, feature)

    def _update_identity(self, person_id: int, feature: np.ndarray, frame_id: int):
        ident = self.identities.get(person_id)
        if ident is None:
            ident = _Identity(person_id)
            self.identities[person_id] = ident
        rep = ident.representative()
        smoothed = feature if rep is None else (1 - self.smooth_factor) * rep + self.smooth_factor * feature
        norm = np.linalg.norm(smoothed)
        if norm > 0:
            smoothed = smoothed / norm
        self._append_feature(ident.features, smoothed)
        ident.state = "ACTIVE"
        ident.last_frame = frame_id

    def _append_feature(self, features: List[np.ndarray], feature: np.ndarray):
        features.append(feature)
        if len(features) > self.max_features:
            del features[: len(features) - self.max_features]
