"""MCT Pipeline v4 – Cluster-based multi-camera tracking.

Combines the SCT pipeline from the existing system with cross-camera
clustering and cluster-level tracking from AIC2024_Track1_Nota.

Flow per frame:
  1. Each camera runs SCT (detect → track → track_manager).
  2. Pose estimation for each track.
  3. Perspective transform: compute world-plane locations.
  4. Assign temporary global IDs.
  5. Cross-camera clustering (pairwise matching → groups).
  6. ClusterTracker update (cluster-level tracking over time).
  7. Push global IDs back into SCT tracks.
  8. Cluster self-refinement (every N frames).
  9. Visualisation + output.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

from modules.data_templates.mct_template import CameraCalibration
from modules.data_templates.sct_template import TrackInfo
from modules.detector.factory import DetectorFactory
from modules.matching.cluster_tracker import ClusterTracker
from modules.matching.cross_camera_clustering_v2 import (
    CrossCameraClusterer,
    IDDistributor,
)
from modules.matching.perspective_projector import PerspectiveProjector
from modules.track_manager.mct_track_manager import MCTTrackManager
from modules.tracker_2D.factory import TrackerFactory
from utils.vis import draw_grid

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Per-camera worker
# -----------------------------------------------------------------------

class _CameraWorker:
    """Wraps detector + tracker + track_manager + pose estimator for one camera."""

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
        self._latest_keypoints: Dict[int, np.ndarray] = {}  # tracker_id → keypoints
        self._frame_id = 0
        self._stopped = False

    def process_next_frame(self) -> bool:
        """Read one frame and run detection + tracking + pose estimation."""
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

        # Pose estimation on tracked boxes
        kpts_dict: Dict[int, np.ndarray] = {}
        if len(tracks) > 0:
            from utils.pose import is_full_body

            track_boxes = [trk[:4] for trk in tracks]
            kpts_scores = self.pe.detect(frame, track_boxes)

            is_full_body_dict = {}
            for i, trk in enumerate(tracks):
                tracker_id = int(trk[4])
                if i < len(kpts_scores):
                    kpts_dict[tracker_id] = kpts_scores[i]
                    is_full_body_dict[tracker_id] = is_full_body(
                        kpts_scores[i], confidence_threshold=0.5,
                    )
                else:
                    is_full_body_dict[tracker_id] = False
            frame_info["is_full_body"] = is_full_body_dict

        # Track management (Re-ID)
        self._latest_tracks = self.track_manager.process(tracks, frame_info)
        self._latest_frame = frame
        self._latest_keypoints = kpts_dict

        # Attach keypoints to TrackInfo objects
        for track in self._latest_tracks:
            kpts = kpts_dict.get(track.tracker_id)
            if kpts is not None:
                track.keypoints = kpts

        return True

    @property
    def stopped(self) -> bool:
        return self._stopped

    def release(self):
        self.cap.release()


# -----------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------

class MCTPipelineV4:
    """Multi-camera tracking pipeline with cluster-based cross-camera matching.

    Integrates:
    - Per-camera SCT (existing modules)
    - Pose-aware perspective projection (from AIC2024)
    - Cross-camera pairwise clustering (from AIC2024)
    - Cluster-level tracking with self-refinement (from AIC2024)
    """

    def __init__(self, sct_config: dict, mct_config_path: str):
        with open(mct_config_path) as f:
            cfg = yaml.safe_load(f)

        # Output config
        out_cfg = cfg.get("OUTPUT", {})
        self._output_video = out_cfg.get("video", "outputs/mct4_output.mp4")
        self._output_txt = out_cfg.get("txt_dir", "outputs/txt")
        self._output_fps = out_cfg.get("fps", 25)
        self._draw_local = out_cfg.get("draw_local", True)

        # Camera setup
        cameras = cfg["CAMERAS"]
        self.calibrations: Dict[int, CameraCalibration] = {}
        self.workers: Dict[int, _CameraWorker] = {}
        for cam_id_str, cam_cfg in cameras.items():
            cid = int(cam_id_str)
            cal = CameraCalibration.load_from_json(cam_cfg["calibration"], cid)
            self.calibrations[cid] = cal
            self.workers[cid] = _CameraWorker(cid, cam_cfg["video"], sct_config)

        # Perspective projector
        perspective_cfg = cfg.get("PERSPECTIVE", {})
        self.projector = PerspectiveProjector(perspective_cfg)

        # Cross-camera clustering
        clustering_cfg = cfg.get("CLUSTERING", {})
        self.clusterer = CrossCameraClusterer(clustering_cfg)
        self.id_distributor = IDDistributor()

        # Cluster tracker (MCTracker)
        mc_tracker_cfg = cfg.get("MC_TRACKER", {})
        self.cluster_tracker = ClusterTracker(mc_tracker_cfg)
        self.refinement_interval = mc_tracker_cfg.get("refinement_interval", 5)

        self.cam_ids_sorted = sorted(self.calibrations)
        self.cam_names = [f"Cam {c}" for c in self.cam_ids_sorted]

    # ---- main loop ----

    def run(
        self,
        output_path: str | None = None,
        txt_dir: str | None = None,
    ):
        """Process all cameras frame by frame."""
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

        logger.info("MCT Pipeline v4 starting – %d cameras", len(self.workers))

        import concurrent.futures

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.workers))

        try:
            while True:
                # --- Step 1: Run per-camera SCT in parallel ---
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
                        logger.error("Error in camera worker %d: %s", cid, e)
                        results[cid] = False

                if not any(results.values()):
                    break
                frame_count += 1

                # --- Step 2: Collect tracks and compute locations ---
                per_cam_tracks: Dict[int, List[TrackInfo]] = {}
                frames: Dict[int, np.ndarray] = {}

                for cid, w in self.workers.items():
                    if w._latest_frame is not None:
                        frames[cid] = w._latest_frame
                        tracks = list(w._latest_tracks)

                        # Compute world-plane locations via perspective projection
                        if cid in self.calibrations:
                            self.projector.compute_locations(
                                tracks, self.calibrations[cid],
                            )

                        per_cam_tracks[cid] = tracks

                # --- Step 3: Reset ID distributor each frame ---
                self.id_distributor.reset()

                # --- Step 4: Cross-camera clustering ---
                groups = self.clusterer.update(per_cam_tracks, self.id_distributor)

                # --- Step 5: Cluster tracker update ---
                self.cluster_tracker.update(groups)

                # --- Step 6: Push global IDs back to SCT ---
                self.clusterer.update_using_cluster_tracker(
                    per_cam_tracks, self.cluster_tracker,
                )

                # --- Step 7: Apply global IDs to galleries ---
                for cid, worker in self.workers.items():
                    tracks = per_cam_tracks.get(cid, [])
                    remap = {}
                    for track in tracks:
                        if (track.person_id is not None
                                and track.person_id > 0
                                and hasattr(track, "_original_pid")
                                and track._original_pid != track.person_id):
                            remap[track._original_pid] = track.person_id
                    if remap:
                        worker.track_manager.apply_global_ids(remap)

                # --- Step 8: Cluster self-refinement ---
                if frame_count % self.refinement_interval == 0:
                    self.cluster_tracker.refinement_clusters()

                # --- Step 9: Visualization + Output ---
                # Re-read tracks from gallery for global-ID labelled output
                per_cam_global: Dict[int, List[TrackInfo]] = {}
                for cid, worker in self.workers.items():
                    per_cam_global[cid] = worker.track_manager.get_active_tracks()

                # Build output using cluster tracker IDs
                active_clusters = [
                    t for t in self.cluster_tracker.tracked_mtracks
                    if t.is_activated
                ]
                active_cluster_ids = {t.track_id for t in active_clusters}

                # MOT15 output
                for cid in self.cam_ids_sorted:
                    fh = mot_files.get(cid)
                    if fh is None:
                        continue
                    for t in per_cam_tracks.get(cid, []):
                        pid = t.person_id
                        if pid is None or pid < 0:
                            continue
                        x1, y1, x2, y2 = t.bbox
                        fh.write(
                            f"{frame_count},{pid},{x1:.1f},{y1:.1f},"
                            f"{x2-x1:.1f},{y2-y1:.1f},1,-1,-1,-1\n"
                        )

                # Grid video
                ann = []
                for cid in self.cam_ids_sorted:
                    if cid not in frames:
                        continue
                    annotated = self._draw(
                        frames[cid].copy(),
                        per_cam_tracks.get(cid, []),
                        active_cluster_ids,
                        cid, frame_count,
                    )
                    ann.append(annotated)

                if ann:
                    grid = draw_grid(ann, self.cam_names)
                    elapsed = time.time() - t0
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    n_active = len(active_clusters)
                    cv2.putText(
                        grid,
                        f"Frame: {frame_count} | FPS: {fps:.1f} | Globals: {n_active}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )
                    if writer is None:
                        h, w = grid.shape[:2]
                        writer = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            self._output_fps,
                            (w, h),
                        )
                    writer.write(grid)

                if frame_count % 100 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "Frame %d | %d active clusters | %d lost | %.1f FPS",
                        frame_count, len(active_clusters),
                        len(self.cluster_tracker.lost_mtracks),
                        frame_count / elapsed if elapsed > 0 else 0,
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
        active_cluster_ids: set,
        cam_id: int,
        frame_id: int,
    ) -> np.ndarray:
        """Draw bounding boxes with global IDs."""
        np.random.seed(42)
        palette = np.random.randint(0, 255, size=(500, 3), dtype=np.uint8)

        for t in tracks:
            pid = t.person_id
            if pid is None:
                continue

            is_global = pid in active_cluster_ids
            label_parts = []

            if self._draw_local and not is_global:
                label_parts.append(f"L{pid}")
                color = (128, 128, 128)
            elif is_global:
                label_parts.append(f"G{pid}")
                color = tuple(int(c) for c in palette[pid % len(palette)])
            else:
                if not self._draw_local:
                    continue
                label_parts.append(f"L{pid}")
                color = (128, 128, 128)

            label = " | ".join(label_parts)
            x1, y1, x2, y2 = map(int, t.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2,
            )
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
