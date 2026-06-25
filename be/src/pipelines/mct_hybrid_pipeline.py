"""Hybrid MCT Pipeline.

Handles primary cameras for global ID creation,
and secondary cameras for ROI analysis and Re-ID.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

from modules.data_templates.mct_template import CameraCalibration
from modules.data_templates.sct_template import TrackInfo
from modules.matching.cross_camera_clustering import CrossCameraClusterer
from modules.track_manager.global_track_manager_v2 import GlobalTrackManagerV2
from pipelines.camera_worker import CameraWorker
from utils.vis import draw_grid

logger = logging.getLogger(__name__)


class MCTHybridPipeline:
    def __init__(self, sct_config: dict, mct_config_path: str):
        with open(mct_config_path) as f:
            mct_cfg = yaml.safe_load(f)

        self.clusterer = CrossCameraClusterer(mct_cfg["MATCHING"])
        self.global_manager = GlobalTrackManagerV2(
            {**mct_cfg["MATCHING"]["thresholds"], **mct_cfg["GLOBAL_TRACK"]},
        )
        self.should_stop = False

        # Open log file for secondary ReID
        os.makedirs("outputs", exist_ok=True)
        self.reid_log_file = open("outputs/reid_secondary_log.txt", "w")
        self.th_reid_sec = mct_cfg["MATCHING"]["thresholds"].get("reid_secondary", 0.7)

        cameras = mct_cfg["CAMERAS"]
        self.H_invs: Dict[int, np.ndarray] = {}
        self.workers: Dict[int, CameraWorker] = {}
        self.camera_rois: Dict[int, list] = {}
        self.camera_types: Dict[int, str] = {}
        self.secondary_mapping: Dict[Tuple[int, int], int] = {}
        
        for cam_id_str, cam_cfg in cameras.items():
            cid = int(cam_id_str)
            self.camera_types[cid] = cam_cfg.get("type", "primary")
            
            if self.camera_types[cid] == "primary":
                cal = CameraCalibration.load_from_json(cam_cfg["calibration"], cid)
                self.H_invs[cid] = cal.H_inv
            
            self.workers[cid] = CameraWorker(cid, cam_cfg["video"], sct_config)
            self.camera_rois[cid] = cam_cfg.get("rois") or []

        self.cam_ids_sorted = sorted(self.workers.keys())
        self.cam_names = [f"Cam {c} ({self.camera_types[c]})" for c in self.cam_ids_sorted]

        out_cfg = mct_cfg.get("OUTPUT", {})
        self._output_video = out_cfg.get("video", "outputs/mct_hybrid_output.mp4")
        self._output_txt = out_cfg.get("txt_dir", "outputs/txt")
        self._output_fps = out_cfg.get("fps", 25)
        self._draw_local = out_cfg.get("draw_local", True)

    def run(self, output_path: str | None = None, txt_dir: str | None = None):
        output_path = output_path or self._output_video
        txt_dir = txt_dir or self._output_txt

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)

        mot_files = {
            cid: open(os.path.join(txt_dir, f"cam{cid}_mct_hybrid.txt"), "w")
            for cid in self.workers
        }
        self._writer = None
        self._video_out_path = output_path
        frame_count = 0
        t0 = time.time()

        logger.info("Hybrid MCT Pipeline starting – %d cameras", len(self.workers))

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

                per_cam, frames = self._collect()
                mapping = self._match(per_cam, frame_count)

                self._write_mot(mot_files, per_cam, mapping, frame_count)
                self._write_video(frames, per_cam, mapping, frame_count, t0)

                if frame_count % 100 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "Frame %d | %d globals | %.1f FPS",
                        frame_count,
                        self.global_manager.num_globals,
                        frame_count / elapsed,
                    )

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted")
        finally:
            executor.shutdown(wait=True)
            for w in self.workers.values():
                w.release()
            if self._writer:
                self._writer.release()
            for fh in mot_files.values():
                fh.close()
            logger.info("Done: %d frames. MOT15 -> %s/", frame_count, txt_dir)

    def _collect(self):
        per_cam: Dict[int, List[TrackInfo]] = {}
        frames: Dict[int, np.ndarray] = {}
        for cid, w in self.workers.items():
            if w.latest_frame is not None:
                frames[cid] = w.latest_frame
                per_cam[cid] = w.latest_tracks
        return per_cam, frames

    def _match(self, per_cam: Dict[int, List[TrackInfo]], frame_count: int) -> Dict[Tuple[int, int], int]:
        mapping = {}
        
        # Primary cameras: Use full clustering and assign
        primary_tracks, primary_cids, primary_pids = [], [], []
        for cid in self.cam_ids_sorted:
            if self.camera_types[cid] != "primary":
                continue
            rois = self.camera_rois.get(cid, [])
            for t in per_cam.get(cid, []):
                if t.person_id is None or t.get_representative_feature() is None:
                    continue
                
                # Check ROI for primary cameras
                in_roi = True
                if rois:
                    in_roi = False
                    for roi in rois:
                        if self._is_in_roi(t.bbox, roi.get("polygon", [])):
                            in_roi = True
                            break
                            
                if in_roi:
                    primary_tracks.append(t)
                    primary_cids.append(cid)
                    primary_pids.append(t.person_id)

        active_primary_gids = set()
        if not primary_tracks:
            self.global_manager.age_all()
        else:
            cam_arr = np.array(primary_cids, dtype=np.int32)
            pid_arr = np.array(primary_pids, dtype=np.int32)
            feet = np.array([[(t.bbox[0] + t.bbox[2]) * 0.5, t.bbox[3]] for t in primary_tracks], dtype=np.float64)
            feats = CrossCameraClusterer.l2_normalise(np.array([t.get_representative_feature() for t in primary_tracks], dtype=np.float64))

            clusters = self.clusterer.cluster(cam_arr, feet, feats, self.H_invs)
            
            # Compute max homography distance within each cluster
            world = CrossCameraClusterer._project_feet(feet, cam_arr, self.H_invs)
            cluster_homo_dists = []
            for c in clusters:
                if len(c) < 2:
                    cluster_homo_dists.append(0.0)
                else:
                    c_world = world[c]
                    max_dist = 0.0
                    for i in range(len(c)):
                        for j in range(i + 1, len(c)):
                            d = np.linalg.norm(c_world[i] - c_world[j])
                            if d > max_dist:
                                max_dist = d
                    cluster_homo_dists.append(float(max_dist))
                    
            primary_mapping = self.global_manager.assign(clusters, feats, cam_arr, pid_arr, frame_count, log_file=self.reid_log_file, cluster_homo_dists=cluster_homo_dists)
            self.reid_log_file.flush()
            mapping.update(primary_mapping)
            active_primary_gids = set(primary_mapping.values())

        # Secondary cameras: Match against existing global tracks
        for cid in self.cam_ids_sorted:
            if self.camera_types[cid] != "secondary":
                continue
            secondary_tracks = per_cam.get(cid, [])
            valid_tracks = [t for t in secondary_tracks if t.person_id is not None and t.get_representative_feature() is not None]
            
            if valid_tracks:
                # Invalidate any stuck cached secondary mappings that are currently active in primary
                for t in valid_tracks:
                    key = (cid, t.person_id)
                    if key in self.secondary_mapping and self.secondary_mapping[key] in active_primary_gids:
                        del self.secondary_mapping[key]
                
                feats = CrossCameraClusterer.l2_normalise(np.array([t.get_representative_feature() for t in valid_tracks], dtype=np.float64))
                assigned_gids, distances, best_gids = self.global_manager.match_only(feats, threshold=self.th_reid_sec, ignore_gids=active_primary_gids)

                # Collect all candidate assignments for this camera (key -> (gid, dist, from_cache))
                # gid_to_best: gid -> (key, dist) — để resolve conflict khi nhiều track cùng gán 1 GID
                gid_to_best: Dict[int, Tuple[Tuple[int, int], float]] = {}
                candidates = []  # list of (key, gid, dist, from_cache)

                for t, gid, dist, best_gid in zip(valid_tracks, assigned_gids, distances, best_gids):
                    key = (cid, t.person_id)

                    if key in self.secondary_mapping:
                        # Cached assignment – use previous gid, treat dist as 0 (highest priority)
                        candidates.append((key, self.secondary_mapping[key], 0.0, True, dist, best_gid, gid))
                    elif gid is not None:
                        candidates.append((key, gid, dist if dist is not None else float("inf"), False, dist, best_gid, gid))
                    else:
                        # Even if no GID assigned, keep it in candidates list so we can log it
                        candidates.append((key, None, float("inf"), False, dist, best_gid, None))

                # Resolve conflicts: for each GID only keep the candidate with smallest distance
                for key, gid, dist, from_cache, orig_dist, best_gid, reid_gid in candidates:
                    if gid is not None:
                        if gid not in gid_to_best or dist < gid_to_best[gid][1]:
                            gid_to_best[gid] = (key, dist)

                # Build final mapping, only winning candidates get the GID
                winning_keys = {info[0] for info in gid_to_best.values()}
                for key, gid, dist, from_cache, orig_dist, best_gid, reid_gid in candidates:
                    final_gid = None
                    if gid is not None and key in winning_keys and gid_to_best.get(gid, (None,))[0] == key:
                        final_gid = gid
                        mapping[key] = gid
                        if not from_cache:
                            self.secondary_mapping[key] = gid
                    else:
                        # Lost the conflict — remove stale cache entry if present
                        if key in self.secondary_mapping:
                            logger.debug(
                                "Secondary cam %d: local %d lost GID %d conflict, clearing cache",
                                cid, key[1], gid,
                            )
                            del self.secondary_mapping[key]

                    # Now safely log the outcome
                    if orig_dist is not None:
                        if final_gid is not None:
                            status = "CACHED" if from_cache else "MATCHED"
                        else:
                            status = "FAILED"
                            
                        self.reid_log_file.write(
                            f"Frame {frame_count} | Cam {cid} | Local {key[1]} | "
                            f"Min Dist to GID {best_gid}: {orig_dist:.4f} | Thresh: {self.th_reid_sec} | "
                            f"Status: {status} | Assigned GID: {final_gid}\n"
                        )
                self.reid_log_file.flush()

        return mapping

    def _is_in_roi(self, bbox: List[float], polygon: List[List[float]]) -> bool:
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_bottom = bbox[3]
        pts = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (x_center, y_bottom), False) >= 0

    def _write_mot(self, mot_files, per_cam, mapping, frame_count):
        for cid, tracks in per_cam.items():
            fh = mot_files.get(cid)
            if fh is None:
                continue
            for t in tracks:
                if t.person_id is None:
                    continue
                gid = mapping.get((cid, t.person_id), -1)
                x1, y1, x2, y2 = t.bbox
                fh.write(
                    f"{frame_count},{gid},{x1:.1f},{y1:.1f},"
                    f"{x2 - x1:.1f},{y2 - y1:.1f},1,-1,-1,-1\n"
                )

    def _write_video(self, frames, per_cam, mapping, frame_count, t0):
        ann = self._annotate_frames(frames, per_cam, mapping, frame_count)
        if not ann:
            return

        grid = draw_grid(ann, self.cam_names)
        elapsed = time.time() - t0
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(
            grid,
            f"Frame: {frame_count} | FPS: {fps:.1f} | "
            f"Globals: {self.global_manager.num_globals}",
            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4,
        )

        if self._writer is None:
            h, w = grid.shape[:2]
            self._writer = cv2.VideoWriter(
                self._video_out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                self._output_fps, (w, h),
            )
        self._writer.write(grid)

    def _annotate_frames(self, frames, per_cam, mapping, frame_count):
        np.random.seed(42)
        palette = np.random.randint(0, 255, size=(500, 3), dtype=np.uint8)
        result = []
        for cid in self.cam_ids_sorted:
            if cid not in frames:
                continue
            frame = frames[cid].copy()
            
            # Draw ROIs
            rois = self.camera_rois.get(cid, [])
            for roi in rois:
                polygon = roi.get("polygon", [])
                if len(polygon) >= 3:
                    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], (241, 102, 99))
                    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
                    cv2.polylines(frame, [pts], True, (241, 102, 99), 2)
                    name = roi.get("name", "")
                    if name:
                        x, y = int(polygon[0][0]), int(polygon[0][1])
                        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), (241, 102, 99), -1)
                        cv2.putText(frame, name, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            for t in per_cam.get(cid, []):
                if t.person_id is None:
                    continue
                gid = mapping.get((cid, t.person_id))
                
                label_parts = []
                if self._draw_local:
                    label_parts.append(f"Local {t.person_id}")
                
                # ROI check
                in_roi = False
                for roi in rois:
                    if self._is_in_roi(t.bbox, roi.get("polygon", [])):
                        in_roi = True
                        break

                if gid is not None:
                    label_parts.append(f"G{gid}")
                    if in_roi:
                        label_parts.append("ROI")
                    color = tuple(int(c) for c in palette[gid % len(palette)])
                else:
                    label_parts.append("G-1")
                    if in_roi:
                        label_parts.append("ROI")
                    color = (128, 128, 128)

                label = " | ".join(label_parts)
                x1, y1, x2, y2 = map(int, t.bbox)
                
                # Thicker box if in ROI
                thickness = 4 if in_roi else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(
                    frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
                )

            active_locals = [t.person_id for t in per_cam.get(cid, []) if t.person_id is not None]
            local_str = ", ".join(map(str, active_locals))
            
            cv2.putText(
                frame, f"Cam {cid} | F{frame_count} | Locals: {local_str}",
                (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3,
            )
            result.append(frame)
        return result
