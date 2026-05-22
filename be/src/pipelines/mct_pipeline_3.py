"""MCT Pipeline v3 – global IDs pushed back into each camera's SCT.

Key difference from v2:
  After cross-camera matching assigns global IDs, each camera's gallery is
  updated so that ``track.person_id == global_id``.  From the next frame
  onward the SCT continues tracking with that global ID, preserving
  identity across the entire system.

Flow per frame:
  1. Each camera runs SCT → tracks carry either a *global ID* (from last
     frame's remap) or a *local ID* (newly confirmed track).
  2. Flatten all tracks, cluster via homo + visual + Union-Find.
  3. For each cluster, determine the global ID:
     a. If any track already carries a known global ID → reuse it.
     b. Otherwise, ReID against recently-lost globals (feature-based).
     c. If still no match → allocate a fresh global ID.
  4. Build per-camera ``{current_pid: global_id}`` mapping.
  5. Call ``track_manager.apply_global_ids(mapping)`` on every camera so
     the gallery stays consistent with the global ID assignment.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from modules.data_templates.mct_template import CameraCalibration
from modules.data_templates.sct_template import TrackInfo
from modules.detector.factory import DetectorFactory
from modules.track_manager.mct_track_manager import MCTTrackManager
from modules.tracker_2D.factory import TrackerFactory
from utils.vis import draw_grid

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

MCT3_CONFIG = {
    "weights": {"homography": 0.5, "visual": 0.5},
    "thresholds": {
        "homography": 10.0,
        "visual_gate": 0.5,
        "homo_gate": 10.0,
        "combined": 0.6,
        "reid_lost": 0.4,
    },
    "global_track": {
        "max_lost_age": 300,
        "feature_smooth": 0.1,
    },
}

# -----------------------------------------------------------------------
# Helpers
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


class _LostGlobalTrack:
    """Feature snapshot of a global track that is no longer visible."""
    __slots__ = ("global_id", "feature", "lost_age")

    def __init__(self, gid: int, feat: np.ndarray):
        self.global_id = gid
        self.feature = feat.copy()
        self.lost_age = 0


# -----------------------------------------------------------------------
# Per-camera worker (uses MCTTrackManager)
# -----------------------------------------------------------------------

class _CameraWorker:
    def __init__(self, cam_id: int, video_path: str, sct_config: dict):
        from modules.pose_estimator.factory import PoseEstimatorFactory
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(video_path)
        self.detector = DetectorFactory(sct_config["DETECTION"]).get_detector()
        self.tracker = TrackerFactory(sct_config["TRACKING"]).get_tracker()
        self.pe = PoseEstimatorFactory(sct_config["POSE_ESTIMATION"]).get_pose_estimator()
        self.track_manager = MCTTrackManager(sct_config["TRACK_MANAGER"])
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
        
        # Check full body on generated tracks
        if len(tracks) > 0:
            from utils.pose import is_full_body
            track_boxes = [trk[:4] for trk in tracks]
            kpts_scores = self.pe.detect(frame, track_boxes)
            is_full_body_dict = {}
            for i, trk in enumerate(tracks):
                tracker_id = int(trk[4])
                is_full_body_dict[tracker_id] = is_full_body(kpts_scores[i], confidence_threshold=0.5) if i < len(kpts_scores) else False
            frame_info["is_full_body"] = is_full_body_dict
            
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

class MCTPipeline3:

    def __init__(self, sct_config: dict, mct_config_path: str):
        with open(mct_config_path) as f:
            cfg = yaml.safe_load(f)
            
        match_cfg = cfg.get("MATCHING", {})
        self.w_homo = match_cfg.get("weights", {}).get("homography", 0.5)
        self.w_vis = match_cfg.get("weights", {}).get("visual", 0.5)
        self.th_homo = match_cfg.get("thresholds", {}).get("homography", 50.0)
        self.th_vis_gate = match_cfg.get("thresholds", {}).get("visual_gate", 1.0)
        self.th_homo_gate = match_cfg.get("thresholds", {}).get("homo_gate", float("inf"))
        self.th_combined = match_cfg.get("thresholds", {}).get("combined", 0.5)
        self.th_reid_lost = match_cfg.get("thresholds", {}).get("reid_lost", 0.3)
        
        gt_cfg = cfg.get("GLOBAL_TRACK", {})
        self.max_lost_age = gt_cfg.get("max_lost_age", 300)
        self.feat_smooth = gt_cfg.get("feature_smooth", 0.1)
        
        out_cfg = cfg.get("OUTPUT", {})
        self._output_video = out_cfg.get("video", "outputs/mct3_output.mp4")
        self._output_txt = out_cfg.get("txt_dir", "outputs/txt")
        self._output_fps = out_cfg.get("fps", 25)
        self._draw_local = out_cfg.get("draw_local", True)

        cameras = cfg["CAMERAS"]
        self.H_invs: Dict[int, np.ndarray] = {}
        self.workers: Dict[int, _CameraWorker] = {}
        for cam_id_str, cam_cfg in cameras.items():
            cid = int(cam_id_str)
            cal = CameraCalibration.load_from_json(cam_cfg["calibration"], cid)
            self.H_invs[cid] = cal.H_inv
            self.workers[cid] = _CameraWorker(cid, cam_cfg["video"], sct_config)

        # Global ID bookkeeping
        self._next_gid = 1
        self._known_gids: Set[int] = set()
        # feature snapshot per active global ID (for EMA update)
        self._gid_features: Dict[int, np.ndarray] = {}
        # recently-lost globals (for ReID)
        self._lost_globals: Dict[int, _LostGlobalTrack] = {}

        self.cam_ids_sorted = sorted(self.H_invs)
        self.cam_names = [f"Cam {c}" for c in self.cam_ids_sorted]

    # ---- vectorised math (same as v2) ----

    def _project_feet(self, feet: np.ndarray, cids: np.ndarray) -> np.ndarray:
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
    def _cosine_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sim = A @ B.T
        np.clip(sim, -1.0, 1.0, out=sim)
        return 1.0 - sim

    @staticmethod
    def _euclidean_dist(pts: np.ndarray) -> np.ndarray:
        sq = np.sum(pts ** 2, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2)

    @staticmethod
    def _l2(feats: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms

    # ---- allocate ----

    def _alloc_gid(self) -> int:
        gid = self._next_gid
        self._next_gid += 1
        self._known_gids.add(gid)
        return gid

    # ---- core matching ----

    def _process_frame(
        self,
        per_cam_tracks: Dict[int, List[TrackInfo]],
        frame_id: int,
    ) -> Dict[int, Dict[int, int]]:
        """Cluster + assign global IDs.

        Returns:
            ``{cam_id: {current_pid: global_id}}`` — per-camera remap.
        """
        # Flatten tracks with valid features
        tracks: List[TrackInfo] = []
        cids: List[int] = []
        for cid in self.cam_ids_sorted:
            for t in per_cam_tracks.get(cid, []):
                if t.person_id is None:
                    continue
                if t.get_representative_feature() is None:
                    continue
                tracks.append(t)
                cids.append(cid)

        if not tracks:
            self._age_lost()
            return {}

        N = len(tracks)
        cam_arr = np.array(cids, dtype=np.int32)
        feet = np.array(
            [[(t.bbox[0] + t.bbox[2]) * 0.5, t.bbox[3]] for t in tracks],
            dtype=np.float64,
        )
        feats = self._l2(
            np.array([t.get_representative_feature() for t in tracks], dtype=np.float64)
        )

        # --- cost matrix ---
        world = self._project_feet(feet, cam_arr)
        homo_dist = self._euclidean_dist(world)
        vis_dist = self._cosine_dist(feats, feats)
        homo_norm = np.minimum(homo_dist / self.th_homo, 1.0)
        cost = self.w_homo * homo_norm + self.w_vis * vis_dist

        # Hard gates: reject pairs that fail any per-metric threshold
        cost[vis_dist > self.th_vis_gate] = np.inf
        cost[homo_dist > self.th_homo_gate] = np.inf
        cost[cam_arr[:, None] == cam_arr[None, :]] = np.inf

        # --- threshold edges, Union-Find ---
        ui, uj = np.triu_indices(N, k=1)
        ec = cost[ui, uj]
        valid = ec < self.th_combined
        if valid.any():
            ei, ej, ec2 = ui[valid], uj[valid], ec[valid]
            order = np.argsort(ec2)
            ei, ej = ei[order], ej[order]
            uf = _UnionFind(N)
            comp_cams: List[set] = [{int(cam_arr[i])} for i in range(N)]
            for a, b in zip(ei, ej):
                ra, rb = uf.find(int(a)), uf.find(int(b))
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
        else:
            clusters = [[i] for i in range(N)]

        # --- assign global IDs per cluster ---
        seen_gids: Set[int] = set()
        per_cam_map: Dict[int, Dict[int, int]] = {cid: {} for cid in self.cam_ids_sorted}

        # Clusters that need feature-based ReID (no known global ID inside)
        need_reid: List[Tuple[int, np.ndarray]] = []  # (cluster_idx, cluster_feat)

        cluster_gids: List[Optional[int]] = [None] * len(clusters)

        for k, idxs in enumerate(clusters):
            # Check if any track already carries a known global ID
            existing_gids: Set[int] = set()
            cluster_cids = {int(cam_arr[i]) for i in idxs}
            for idx in idxs:
                pid = tracks[idx].person_id
                if pid in self._known_gids:
                    existing_gids.add(pid)

            cluster_feat = self._l2(np.mean(feats[idxs], axis=0, keepdims=True))[0]

            if existing_gids:
                gid = min(existing_gids)
                cluster_gids[k] = gid
                # Merge other gids into this one
                for old_gid in existing_gids:
                    if old_gid != gid:
                        self._merge_gid(old_gid, gid)
            else:
                if len(cluster_cids) >= 2:
                    need_reid.append((k, cluster_feat))

        # ReID against lost globals for clusters without a known gid
        if need_reid and self._lost_globals:
            query_feats = np.stack([f for _, f in need_reid])
            lost_gids = list(self._lost_globals.keys())
            lost_feats = np.stack([self._lost_globals[g].feature for g in lost_gids])
            reid_cost = self._cosine_dist(query_feats, lost_feats)
            row, col = linear_sum_assignment(reid_cost)
            matched_query = set()
            for r, c in zip(row, col):
                if reid_cost[r, c] < self.th_reid_lost:
                    cidx = need_reid[r][0]
                    gid = lost_gids[c]
                    cluster_gids[cidx] = gid
                    self._lost_globals.pop(gid, None)
                    matched_query.add(r)
                    logger.info("ReID lost global G%d back (dist=%.3f)", gid, reid_cost[r, c])

        # Allocate new IDs for remaining
        for k in range(len(clusters)):
            if cluster_gids[k] is None:
                cluster_cids = {int(cam_arr[i]) for i in clusters[k]}
                if len(cluster_cids) >= 2:
                    cluster_gids[k] = self._alloc_gid()

        # Build per-camera remap + update gid_features
        for k, idxs in enumerate(clusters):
            gid = cluster_gids[k]
            if gid is None:
                continue
            seen_gids.add(gid)
            self._known_gids.add(gid)

            cluster_feat = self._l2(np.mean(feats[idxs], axis=0, keepdims=True))[0]
            if gid in self._gid_features:
                old = self._gid_features[gid]
                self._gid_features[gid] = (1 - self.feat_smooth) * old + self.feat_smooth * cluster_feat
                norm = np.linalg.norm(self._gid_features[gid])
                if norm > 0:
                    self._gid_features[gid] /= norm
            else:
                self._gid_features[gid] = cluster_feat

            for idx in idxs:
                t = tracks[idx]
                cid = int(cam_arr[idx])
                current_pid = t.person_id
                if current_pid != gid:
                    per_cam_map[cid][current_pid] = gid

        # Move unseen active gids to lost
        for gid in list(self._gid_features):
            if gid not in seen_gids:
                feat = self._gid_features.pop(gid)
                if gid not in self._lost_globals:
                    self._lost_globals[gid] = _LostGlobalTrack(gid, feat)

        self._age_lost()

        return per_cam_map

    def _merge_gid(self, old_gid: int, keep_gid: int):
        """Merge old_gid into keep_gid across all galleries."""
        for worker in self.workers.values():
            gallery = worker.track_manager.gallery
            if old_gid in gallery.tracks:
                track = gallery.tracks.pop(old_gid)
                track.person_id = keep_gid
                gallery.tracks[keep_gid] = track
                gallery.map_id[track.tracker_id] = keep_gid

        self._gid_features.pop(old_gid, None)
        self._lost_globals.pop(old_gid, None)
        logger.debug("Merged G%d into G%d", old_gid, keep_gid)

    def _age_lost(self):
        for gid in list(self._lost_globals):
            self._lost_globals[gid].lost_age += 1
            if self._lost_globals[gid].lost_age > self.max_lost_age:
                del self._lost_globals[gid]
                self._known_gids.discard(gid)

    # ---- main loop ----

    def run(
        self,
        output_path: str | None = None,
        txt_dir: str | None = None,
    ):
        output_path = output_path or self._output_video
        txt_dir = txt_dir or self._output_txt

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)

        mot_files = {
            cid: open(os.path.join(txt_dir, f"cam{cid}_mct.txt"), "w")
            for cid in self.workers
        }
        writer = None
        frame_count = 0
        t0 = time.time()

        logger.info("MCT Pipeline v3 starting – %d cameras", len(self.workers))
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.workers))

        try:
            while True:
                alive = False
                futures = {
                    executor.submit(w.process_next_frame): cid
                    for cid, w in self.workers.items() if not w.stopped
                }
                
                results = {}
                for fut in concurrent.futures.as_completed(futures):
                    cid = futures[fut]
                    try:
                        results[cid] = fut.result()
                    except Exception as e:
                        logger.error(f"Error in camera worker {cid}: {e}")
                        results[cid] = False
                        
                for cid, res in results.items():
                    if res:
                        alive = True
                        
                if not alive:
                    break
                frame_count += 1

                # Collect
                per_cam: Dict[int, List[TrackInfo]] = {}
                frames: Dict[int, np.ndarray] = {}
                for cid, w in self.workers.items():
                    if w._latest_frame is not None:
                        frames[cid] = w._latest_frame
                        per_cam[cid] = list(w._latest_tracks)

                # Cluster + assign global IDs
                per_cam_map = self._process_frame(per_cam, frame_count)

                # Push global IDs back into each camera's gallery
                for cid, worker in self.workers.items():
                    remap = per_cam_map.get(cid, {})
                    if remap:
                        worker.track_manager.apply_global_ids(remap)

                # Now tracks in per_cam still have old pids; re-read from gallery
                # for correct global-ID labelled output
                per_cam_global: Dict[int, List[TrackInfo]] = {}
                for cid, worker in self.workers.items():
                    per_cam_global[cid] = worker.track_manager.get_active_tracks()

                # MOT15 output (global IDs)
                for cid in self.cam_ids_sorted:
                    fh = mot_files.get(cid)
                    if fh is None:
                        continue
                    for t in per_cam_global.get(cid, []):
                        x1, y1, x2, y2 = t.bbox
                        fh.write(
                            f"{frame_count},{t.person_id},{x1:.1f},{y1:.1f},"
                            f"{x2-x1:.1f},{y2-y1:.1f},1,-1,-1,-1\n"
                        )

                # Grid video
                ann = []
                for cid in self.cam_ids_sorted:
                    if cid not in frames:
                        continue
                    annotated = self._draw(
                        frames[cid].copy(),
                        per_cam_global.get(cid, []),
                        cid, frame_count,
                    )
                    ann.append(annotated)
                if ann:
                    grid = draw_grid(ann, self.cam_names)
                    elapsed = time.time() - t0
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    n_active = len(self._gid_features)
                    cv2.putText(
                        grid,
                        f"Frame: {frame_count} | FPS: {fps:.1f} | Globals: {n_active}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )
                    if writer is None:
                        h, w = grid.shape[:2]
                        writer = cv2.VideoWriter(
                            output_path, cv2.VideoWriter_fourcc(*"mp4v"), self._output_fps, (w, h),
                        )
                    writer.write(grid)

                if frame_count % 100 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "Frame %d | %d active gids | %d lost | %.1f FPS",
                        frame_count, len(self._gid_features),
                        len(self._lost_globals), frame_count / elapsed,
                    )

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted")
        finally:
            executor.shutdown(wait=True)
            for w in self.workers.values():
                w.release()
            if writer:
                writer.release()
            for fh in mot_files.values():
                fh.close()
            logger.info("Done: %d frames. MOT15 → %s/", frame_count, txt_dir)

    # ---- drawing ----

    def _draw(
        self,
        frame: np.ndarray,
        tracks: List[TrackInfo],
        cam_id: int,
        frame_id: int,
    ) -> np.ndarray:
        np.random.seed(42)
        palette = np.random.randint(0, 255, size=(500, 3), dtype=np.uint8)

        for t in tracks:
            gid = t.person_id
            if gid is None:
                continue

            label_parts = []
            
            # Since pipeline 3 remaps local pids via apply_global_ids,
            # tracks.tracker_id can be used to query original local IDs if needed.
            # But the gallery.tracks keys (which set t.person_id) are the remapped ones.
            # For simplicity, if it's not a global id, it's the original local id.
            is_global = gid in self._known_gids
            
            if self._draw_local:
                if not is_global:
                    label_parts.append(f"L{t.person_id}")
                else:
                    # In Pipeline 3, once assigned, person_id IS the global ID. 
                    # If we really wanted the old local ID here we would query gallery.map_id[t.tracker_id] 
                    # but for now we just show global
                    pass

            if not is_global:
                if not self._draw_local:
                    continue
                color = (128, 128, 128)
            else:
                label_parts.append(f"G{gid}")
                color = tuple(int(c) for c in palette[gid % len(palette)])

            label = " | ".join(label_parts)

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
