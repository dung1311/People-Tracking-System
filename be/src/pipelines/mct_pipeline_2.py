"""MCT Pipeline v2 – unified cost matrix + Union-Find clustering.

Design:
  1. Each frame: flatten all tracks from every camera into one list (N tracks).
  2. Build an (N x N) combined cost matrix (homo + visual).
     Same-camera cells = inf.
  3. Threshold + greedy Union-Find with same-camera constraint → clusters.
     Transitive: if t1@cam1 ↔ t2@cam2 and t2@cam2 ↔ t3@cam3
     then {t1, t2, t3} become one cluster automatically.
  4. ReID each cluster against existing global tracks (cosine on features).
     Matched → reuse old global ID.  Unmatched → allocate new.
  5. Write MOT15 txt per camera (global IDs) + annotated grid video.

All heavy computation is numpy-vectorised for speed.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from modules.data_templates.mct_template import CameraCalibration
from modules.data_templates.sct_template import TrackInfo
from modules.detector.factory import DetectorFactory
from modules.track_manager.single_track_manager import SingleTrackManager
from modules.tracker_2D.factory import TrackerFactory
from utils.vis import draw_grid

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

MCT2_CONFIG = {
    "weights": {"homography": 0.5, "visual": 0.5},
    "thresholds": {
        "homography": 200.0,   # normalisation cap (world-plane units)
        "combined": 0.6,       # clustering edge threshold
        "reid": 0.4,           # cosine distance for re-id against globals
    },
    "global_track": {
        "max_lost_age": 300,
        "feature_smooth": 0.1,
    },
}

# -----------------------------------------------------------------------
# Union-Find
# -----------------------------------------------------------------------

class _UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

# -----------------------------------------------------------------------
# Lightweight global track
# -----------------------------------------------------------------------

class _GlobalTrack:
    __slots__ = ("global_id", "feature", "last_seen_frame", "lost_age", "cam_tracks")

    def __init__(self, global_id: int, feature: np.ndarray, frame_id: int):
        self.global_id = global_id
        self.feature = feature.copy()
        self.last_seen_frame = frame_id
        self.lost_age = 0
        self.cam_tracks: Dict[int, int] = {}

    def update_feature(self, feat: np.ndarray, smooth: float = 0.1):
        self.feature = (1.0 - smooth) * self.feature + smooth * feat
        norm = np.linalg.norm(self.feature)
        if norm > 0:
            self.feature /= norm

# -----------------------------------------------------------------------
# Per-camera SCT worker (reuse from pipeline v1 but inlined to stay self-contained)
# -----------------------------------------------------------------------

class _CameraWorker:
    def __init__(self, cam_id: int, video_path: str, sct_config: dict):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(video_path)
        self.detector = DetectorFactory(sct_config["DETECTION"]).get_detector()
        self.tracker = TrackerFactory(sct_config["TRACKING"]).get_tracker()
        self.track_manager = SingleTrackManager(sct_config["TRACK_MANAGER"])
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_tracks: List[TrackInfo] = []
        self._frame_id = 0
        self._stopped = False

    def process_next_frame(self) -> bool:
        ret, frame = self.cap.read()
        if not ret:
            self._stopped = True
            return False
        self._frame_id += 1
        h, w = frame.shape[:2]
        frame_info = {
            "cam_id": self.cam_id,
            "frame_id": self._frame_id,
            "frame": frame,
            "img_info": (h, w),
            "img_size": (h, w),
        }
        bboxes = self.detector.detect(frame)
        tracks = self.tracker.update(bboxes, frame_info)
        self._latest_tracks = self.track_manager.process(tracks, frame_info)
        self._latest_frame = frame
        return True

    @property
    def stopped(self) -> bool:
        return self._stopped

    def release(self):
        self.cap.release()

# -----------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------

class MCTPipeline2:
    """MCT using a single unified cost matrix + Union-Find clustering."""

    def __init__(
        self,
        sct_config: dict,
        camera_video_map: Dict[int, str],
        camera_calib_map: Dict[int, str],
        mct_config: dict | None = None,
    ):
        cfg = mct_config or MCT2_CONFIG
        self.w_homo = cfg["weights"]["homography"]
        self.w_vis = cfg["weights"]["visual"]
        self.th_homo = cfg["thresholds"]["homography"]
        self.th_combined = cfg["thresholds"]["combined"]
        self.th_reid = cfg["thresholds"]["reid"]
        gt_cfg = cfg.get("global_track", {})
        self.max_lost_age = gt_cfg.get("max_lost_age", 300)
        self.feat_smooth = gt_cfg.get("feature_smooth", 0.1)

        # Camera H_inv matrices (image → world)
        self.H_invs: Dict[int, np.ndarray] = {}
        for cam_id, path in camera_calib_map.items():
            cal = CameraCalibration.load_from_json(path, cam_id)
            self.H_invs[cam_id] = cal.H_inv

        # Workers
        self.workers: Dict[int, _CameraWorker] = {
            cid: _CameraWorker(cid, vp, sct_config)
            for cid, vp in camera_video_map.items()
        }

        self.global_tracks: Dict[int, _GlobalTrack] = {}
        self._next_gid = 1
        self.cam_ids_sorted = sorted(self.H_invs)
        self.cam_names = [f"Cam {c}" for c in self.cam_ids_sorted]

    # ---------- vectorised math ----------

    def _project_feet(self, feet: np.ndarray, cids: np.ndarray) -> np.ndarray:
        """(N,2) pixel feet + (N,) cam ids → (N,2) world coordinates."""
        N = len(feet)
        pts_h = np.empty((N, 3), dtype=np.float64)
        pts_h[:, :2] = feet
        pts_h[:, 2] = 1.0
        world = np.empty((N, 2), dtype=np.float64)
        for cid, H_inv in self.H_invs.items():
            mask = cids == cid
            if not mask.any():
                continue
            w = (H_inv @ pts_h[mask].T).T
            w /= w[:, 2:3]
            world[mask] = w[:, :2]
        return world

    @staticmethod
    def _cosine_dist_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """(N,D), (M,D) both L2-normed → (N,M) cosine distance."""
        sim = A @ B.T
        np.clip(sim, -1.0, 1.0, out=sim)
        return 1.0 - sim

    @staticmethod
    def _euclidean_dist_matrix(pts: np.ndarray) -> np.ndarray:
        """(N,2) → (N,N) Euclidean distance."""
        sq = np.sum(pts ** 2, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2)

    @staticmethod
    def _l2_normalise(feats: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms

    # ---------- core matching ----------

    def _cluster_and_assign(
        self,
        tracks: List[TrackInfo],
        cam_ids: np.ndarray,
        feet: np.ndarray,
        feats: np.ndarray,
        frame_id: int,
    ) -> Dict[Tuple[int, int], int]:
        """Cluster + ReID → ``{(cam_id, local_pid): global_id}``."""
        N = len(tracks)

        # --- Step 1: build cost matrix ---
        world = self._project_feet(feet, cam_ids)
        homo_dist = self._euclidean_dist_matrix(world)          # (N, N)
        vis_dist = self._cosine_dist_matrix(feats, feats)       # (N, N)
        homo_norm = np.minimum(homo_dist / self.th_homo, 1.0)
        cost = self.w_homo * homo_norm + self.w_vis * vis_dist
        same_cam = cam_ids[:, None] == cam_ids[None, :]
        cost[same_cam] = np.inf

        # --- Step 2: threshold edges, sort ascending ---
        upper = np.triu_indices(N, k=1)
        edge_costs = cost[upper]
        valid = edge_costs < self.th_combined
        if not valid.any():
            clusters = [[i] for i in range(N)]
        else:
            ei = upper[0][valid]
            ej = upper[1][valid]
            ec = edge_costs[valid]
            order = np.argsort(ec)
            ei, ej = ei[order], ej[order]

            # --- Step 3: Union-Find with same-camera constraint ---
            uf = _UnionFind(N)
            comp_cams: List[set] = [{int(cam_ids[i])} for i in range(N)]
            for a, b in zip(ei, ej):
                ra, rb = uf.find(a), uf.find(b)
                if ra == rb:
                    continue
                if comp_cams[ra] & comp_cams[rb]:
                    continue
                merged = comp_cams[ra] | comp_cams[rb]
                uf.union(ra, rb)
                comp_cams[uf.find(ra)] = merged

            groups: Dict[int, List[int]] = defaultdict(list)
            for i in range(N):
                groups[uf.find(i)].append(i)
            clusters = list(groups.values())

        # --- Step 4: cluster representative features ---
        K = len(clusters)
        cluster_feats = np.empty((K, feats.shape[1]), dtype=np.float64)
        for k, idxs in enumerate(clusters):
            cluster_feats[k] = np.mean(feats[idxs], axis=0)
        cluster_feats = self._l2_normalise(cluster_feats)

        # --- Step 5: ReID against existing global tracks ---
        assigned: List[Optional[int]] = [None] * K

        active_gids: List[int] = []
        active_feats: List[np.ndarray] = []
        for gid, gt in self.global_tracks.items():
            active_gids.append(gid)
            active_feats.append(gt.feature)

        if active_gids:
            gf = np.stack(active_feats)                                # (M, D)
            reid_cost = self._cosine_dist_matrix(cluster_feats, gf)    # (K, M)
            row, col = linear_sum_assignment(reid_cost)
            for r, c in zip(row, col):
                if reid_cost[r, c] < self.th_reid:
                    assigned[r] = active_gids[c]

        # New IDs for unmatched
        for k in range(K):
            if assigned[k] is None:
                assigned[k] = self._next_gid
                self._next_gid += 1

        # --- Step 6: update global tracks & build mapping ---
        mapping: Dict[Tuple[int, int], int] = {}
        seen: set = set()

        for k, idxs in enumerate(clusters):
            gid = assigned[k]
            seen.add(gid)
            if gid in self.global_tracks:
                gt = self.global_tracks[gid]
                gt.update_feature(cluster_feats[k], self.feat_smooth)
                gt.last_seen_frame = frame_id
                gt.lost_age = 0
            else:
                gt = _GlobalTrack(gid, cluster_feats[k], frame_id)
                self.global_tracks[gid] = gt

            gt.cam_tracks.clear()
            for idx in idxs:
                t = tracks[idx]
                cid = int(cam_ids[idx])
                pid = t.person_id if t.person_id is not None else t.tracker_id
                gt.cam_tracks[cid] = pid
                mapping[(cid, pid)] = gid

        # Age unseen
        for gid in list(self.global_tracks):
            if gid not in seen:
                self.global_tracks[gid].lost_age += 1
                if self.global_tracks[gid].lost_age > self.max_lost_age:
                    del self.global_tracks[gid]

        return mapping

    # ---------- main loop ----------

    def run(
        self,
        output_path: str = "outputs/mct2_output.mp4",
        txt_dir: str = "outputs/txt",
    ):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)

        mot_files = {
            cid: open(os.path.join(txt_dir, f"cam{cid}_mct.txt"), "w")
            for cid in self.workers
        }
        writer = None
        frame_count = 0
        t0 = time.time()

        logger.info("MCT Pipeline v2 starting – %d cameras", len(self.workers))

        try:
            while True:
                alive = False
                for w in self.workers.values():
                    if not w.stopped and w.process_next_frame():
                        alive = True
                if not alive:
                    break
                frame_count += 1

                # Collect per-camera results
                per_cam: Dict[int, List[TrackInfo]] = {}
                frames: Dict[int, np.ndarray] = {}
                for cid, w in self.workers.items():
                    if w._latest_frame is not None:
                        frames[cid] = w._latest_frame
                        per_cam[cid] = w._latest_tracks

                # Flatten tracks with valid features
                all_tracks: List[TrackInfo] = []
                all_cids: List[int] = []
                for cid in self.cam_ids_sorted:
                    for t in per_cam.get(cid, []):
                        if t.person_id is None:
                            continue
                        feat = t.get_representative_feature()
                        if feat is None:
                            continue
                        all_tracks.append(t)
                        all_cids.append(cid)

                if all_tracks:
                    cam_arr = np.array(all_cids, dtype=np.int32)
                    feet = np.array(
                        [[(t.bbox[0] + t.bbox[2]) * 0.5, t.bbox[3]] for t in all_tracks],
                        dtype=np.float64,
                    )
                    feats = self._l2_normalise(
                        np.array([t.get_representative_feature() for t in all_tracks],
                                 dtype=np.float64)
                    )
                    mapping = self._cluster_and_assign(
                        all_tracks, cam_arr, feet, feats, frame_count,
                    )
                else:
                    mapping = {}
                    for gid in list(self.global_tracks):
                        self.global_tracks[gid].lost_age += 1
                        if self.global_tracks[gid].lost_age > self.max_lost_age:
                            del self.global_tracks[gid]

                # MOT15 output
                for cid, tracks in per_cam.items():
                    fh = mot_files.get(cid)
                    if fh is None:
                        continue
                    for t in tracks:
                        if t.person_id is None:
                            continue
                        gid = mapping.get((cid, t.person_id))
                        if gid is None:
                            continue
                        x1, y1, x2, y2 = t.bbox
                        fh.write(
                            f"{frame_count},{gid},{x1:.1f},{y1:.1f},"
                            f"{x2 - x1:.1f},{y2 - y1:.1f},1,-1,-1,-1\n"
                        )

                # Grid video
                ann = []
                for cid in self.cam_ids_sorted:
                    if cid not in frames:
                        continue
                    ann.append(self._draw(
                        frames[cid].copy(), per_cam.get(cid, []),
                        cid, frame_count, mapping,
                    ))
                if ann:
                    grid = draw_grid(ann, self.cam_names)
                    elapsed = time.time() - t0
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    cv2.putText(
                        grid,
                        f"Frame: {frame_count} | FPS: {fps:.1f} | Globals: {len(self.global_tracks)}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )
                    if writer is None:
                        h, w = grid.shape[:2]
                        writer = cv2.VideoWriter(
                            output_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h),
                        )
                    writer.write(grid)

                if frame_count % 100 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "Frame %d | %d globals | %.1f FPS",
                        frame_count, len(self.global_tracks), frame_count / elapsed,
                    )

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted")
        finally:
            for w in self.workers.values():
                w.release()
            if writer:
                writer.release()
            for fh in mot_files.values():
                fh.close()
            logger.info("Done: %d frames. MOT15 → %s/", frame_count, txt_dir)

    # ---------- drawing ----------

    def _draw(
        self,
        frame: np.ndarray,
        tracks: List[TrackInfo],
        cam_id: int,
        frame_id: int,
        mapping: Dict[Tuple[int, int], int],
    ) -> np.ndarray:
        np.random.seed(42)
        palette = np.random.randint(0, 255, size=(500, 3), dtype=np.uint8)

        for t in tracks:
            if t.person_id is None:
                continue
            gid = mapping.get((cam_id, t.person_id))
            if gid is not None:
                label = f"G{gid}"
                color = tuple(int(c) for c in palette[gid % len(palette)])
            else:
                label = f"L{t.person_id}"
                color = (128, 128, 128)

            x1, y1, x2, y2 = map(int, t.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        cv2.putText(
            frame, f"Cam {cam_id} | F{frame_id}",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
        return frame
