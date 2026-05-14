"""Global Track Manager for multi-camera tracking.

Maintains global person identities across cameras by merging per-camera
SCT results based on cross-camera matching output.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from modules.data_templates.mct_template import (
    CameraCalibration,
    CrossCameraMatchResult,
    GlobalTrackInfo,
    GlobalTrackState,
)
from modules.data_templates.sct_template import TrackInfo

logger = logging.getLogger(__name__)

MCT_CONFIG = {
    "max_lost_age": 300,
    "max_features": 50,
    "max_trajectory_len": 100,
}


class GlobalTrackManager:
    """Manages the lifecycle of global (cross-camera) person tracks."""

    def __init__(self, config: Dict | None = None):
        cfg = config or MCT_CONFIG
        self.max_lost_age: int = cfg.get("max_lost_age", 300)
        self.max_features: int = cfg.get("max_features", 50)
        self.max_trajectory_len: int = cfg.get("max_trajectory_len", 100)

        self.tracks: Dict[int, GlobalTrackInfo] = {}
        self._next_id = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        match_results: List[CrossCameraMatchResult],
        per_cam_tracks: Dict[int, List[TrackInfo]],
        calibrations: Dict[int, CameraCalibration],
        frame_id: int,
    ) -> List[GlobalTrackInfo]:
        """Process one round of cross-camera matching results.

        1. Apply match results to link / create global tracks.
        2. Accumulate trajectories and features for every active local track.
        3. Age and clean up lost/dead global tracks.

        Returns:
            List of currently ACTIVE global tracks.
        """
        seen_global_ids: Set[int] = set()

        # --- Phase 1: process matched pairs --------------------------------
        for mr in match_results:
            if not mr.is_matched:
                continue
            gid = self._link_matched_pair(
                mr.cam_i, mr.local_pid_i,
                mr.cam_j, mr.local_pid_j,
                frame_id,
            )
            if gid is not None:
                seen_global_ids.add(gid)

        # --- Phase 2: ensure every active local track has a global ID ------
        for cam_id, tracks in per_cam_tracks.items():
            cal = calibrations.get(cam_id)
            for t in tracks:
                if t.person_id is None:
                    continue
                gid = self._find_global_id(cam_id, t.person_id)
                if gid is None:
                    gid = self._create_global_track(cam_id, t.person_id, frame_id)
                gt = self.tracks[gid]
                self._accumulate(gt, t, cal)
                gt.last_seen_frame = frame_id
                seen_global_ids.add(gid)

        # --- Phase 3: age management --------------------------------------
        self._age_tracks(seen_global_ids)

        return [t for t in self.tracks.values() if t.state == GlobalTrackState.ACTIVE]

    def get_global_id_for(self, cam_id: int, local_pid: int) -> Optional[int]:
        return self._find_global_id(cam_id, local_pid)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _allocate_id(self) -> int:
        gid = self._next_id
        self._next_id += 1
        return gid

    def _find_global_id(self, cam_id: int, local_pid: int) -> Optional[int]:
        for gid, gt in self.tracks.items():
            if gt.cam_tracks.get(cam_id) == local_pid:
                return gid
        return None

    def _create_global_track(
        self, cam_id: int, local_pid: int, frame_id: int
    ) -> int:
        gid = self._allocate_id()
        gt = GlobalTrackInfo(global_id=gid, frame_id=frame_id)
        gt.cam_tracks[cam_id] = local_pid
        self.tracks[gid] = gt
        logger.debug("New global track %d for cam=%s local_pid=%s", gid, cam_id, local_pid)
        return gid

    def _link_matched_pair(
        self,
        cam_i: int, pid_i: int,
        cam_j: int, pid_j: int,
        frame_id: int,
    ) -> Optional[int]:
        """Link two local tracks into the same global track.

        If one already has a global ID, the other is merged in.
        If both do, the smaller one is merged into the larger.
        If neither does, a new global track is created.
        """
        gid_i = self._find_global_id(cam_i, pid_i)
        gid_j = self._find_global_id(cam_j, pid_j)

        if gid_i is not None and gid_j is not None:
            if gid_i == gid_j:
                return gid_i
            # Merge smaller into larger
            keep, drop = (gid_i, gid_j) if gid_i <= gid_j else (gid_j, gid_i)
            self._merge(keep, drop)
            return keep

        if gid_i is not None:
            self.tracks[gid_i].cam_tracks[cam_j] = pid_j
            return gid_i

        if gid_j is not None:
            self.tracks[gid_j].cam_tracks[cam_i] = pid_i
            return gid_j

        gid = self._allocate_id()
        gt = GlobalTrackInfo(global_id=gid, frame_id=frame_id)
        gt.cam_tracks[cam_i] = pid_i
        gt.cam_tracks[cam_j] = pid_j
        self.tracks[gid] = gt
        logger.debug(
            "New global track %d linking cam%s:pid%s <-> cam%s:pid%s",
            gid, cam_i, pid_i, cam_j, pid_j,
        )
        return gid

    def _merge(self, keep_id: int, drop_id: int):
        """Merge *drop* global track into *keep*."""
        keep = self.tracks[keep_id]
        drop = self.tracks.pop(drop_id)
        for cam_id, pid in drop.cam_tracks.items():
            keep.cam_tracks.setdefault(cam_id, pid)
        keep.features.extend(drop.features)
        if len(keep.features) > self.max_features:
            keep.features = keep.features[-self.max_features:]
        keep.world_trajectory.extend(drop.world_trajectory)
        if len(keep.world_trajectory) > self.max_trajectory_len:
            keep.world_trajectory = keep.world_trajectory[-self.max_trajectory_len:]
        logger.debug("Merged global track %d into %d", drop_id, keep_id)

    def _accumulate(
        self,
        gt: GlobalTrackInfo,
        track: TrackInfo,
        cal: Optional[CameraCalibration],
    ):
        """Accumulate feature and world-plane position from a local track."""
        feat = track.get_representative_feature()
        gt.add_feature(feat, self.max_features)

        if cal is not None and track.bbox is not None:
            foot = np.array(
                [(track.bbox[0] + track.bbox[2]) / 2.0, track.bbox[3]],
                dtype=np.float64,
            )
            world_pt = cal.project_to_world(foot)
            gt.add_world_point(world_pt, self.max_trajectory_len)

    def _age_tracks(self, seen_global_ids: Set[int]):
        """Transition unseen ACTIVE -> LOST, age LOST, and remove DEAD."""
        for gid in list(self.tracks):
            gt = self.tracks[gid]
            if gid in seen_global_ids:
                if gt.state == GlobalTrackState.LOST:
                    gt.state = GlobalTrackState.ACTIVE
                    gt.lost_age = 0
            else:
                if gt.state == GlobalTrackState.ACTIVE:
                    gt.state = GlobalTrackState.LOST
                    gt.lost_age = 1
                elif gt.state == GlobalTrackState.LOST:
                    gt.lost_age += 1
                    if gt.lost_age >= self.max_lost_age:
                        gt.state = GlobalTrackState.DEAD
                        del self.tracks[gid]
                        logger.debug("Removed dead global track %d", gid)
