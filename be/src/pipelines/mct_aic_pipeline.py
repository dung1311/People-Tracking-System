"""MCT pipeline using tracker-level SCT IDs and AIC2024-style global ID assignment."""

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
from modules.matching.cross_camera_clustering_v2 import CrossCameraClusterer, IDDistributor
from modules.matching.perspective_projector import PerspectiveProjector
from modules.pose_estimator.factory import PoseEstimatorFactory
from modules.tracker_2D.factory import TrackerFactory
from utils.box import filter_overlapping_boxes
from utils.vis import draw_grid

logger = logging.getLogger(__name__)


class _SCTCamera:
    """One camera SCT worker: detector -> SORT/OCSort with internal ReID -> pose."""

    def __init__(self, cam_id: int, video_path: str, sct_config: dict):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(video_path)
        self.detector = DetectorFactory(sct_config["DETECTION"]).get_detector()
        self.tracker = TrackerFactory(
            sct_config["TRACKING"],
            sct_config.get("TRACK_MANAGER"),
        ).get_tracker()
        self.pose_estimator = PoseEstimatorFactory(
            sct_config["POSE_ESTIMATION"],
        ).get_pose_estimator()

        self.frame_id = 0
        self.stopped = False
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_tracks: List[TrackInfo] = []

    def process_next_frame(self) -> bool:
        ret, frame = self.cap.read()
        if not ret:
            self.stopped = True
            return False

        self.frame_id += 1
        h, w = frame.shape[:2]
        frame_info = {
            "cam_id": self.cam_id,
            "frame_id": self.frame_id,
            "frame": frame,
            "img_info": (h, w),
            "img_size": (h, w),
        }

        detections = self.detector.detect(frame)
        detections = filter_overlapping_boxes(detections, iou_threshold=0.3)
        tracker_rows = self.tracker.update(detections, frame_info)

        self.latest_tracks = self._rows_to_trackinfo(tracker_rows, frame_info)
        self._attach_pose(frame)
        self.latest_frame = frame
        return True

    def release(self):
        self.cap.release()

    def _rows_to_trackinfo(self, rows: np.ndarray, frame_info: dict) -> List[TrackInfo]:
        tracks: List[TrackInfo] = []
        for row in rows:
            person_id = int(row[4])
            track = TrackInfo(
                tracker_id=person_id,
                bbox=row[:4].tolist() if hasattr(row, "tolist") else list(row[:4]),
                score=float(row[5]) if len(row) > 5 else 1.0,
                class_id=int(row[6]) if len(row) > 6 else 0,
                cam_id=self.cam_id,
                frame_id=frame_info["frame_id"],
            )
            track.person_id = person_id

            feat = self.tracker.identity.get_feature(person_id) if hasattr(self.tracker, "identity") else None
            if feat is not None:
                track.features = [feat]
            tracks.append(track)
        return tracks

    def _attach_pose(self, frame: np.ndarray):
        if not self.latest_tracks:
            return
        boxes = [track.bbox for track in self.latest_tracks]
        keypoints = self.pose_estimator.detect(frame, boxes)
        for idx, track in enumerate(self.latest_tracks):
            if idx < len(keypoints):
                track.keypoints = keypoints[idx]


class MCTAICPipeline:
    """Multi-camera tracking from SCT tracker output + AIC2024-style MCT."""

    def __init__(self, sct_config: dict, mct_config_path: str):
        with open(mct_config_path) as f:
            cfg = yaml.safe_load(f)

        out_cfg = cfg.get("OUTPUT", {})
        self.output_video = out_cfg.get("video", "outputs/mct_aic_output.mp4")
        self.output_txt_dir = out_cfg.get("txt_dir", "outputs/txt")
        self.output_fps = out_cfg.get("fps", 25)
        self.draw_local = out_cfg.get("draw_local", True)

        self.calibrations: Dict[int, CameraCalibration] = {}
        self.workers: Dict[int, _SCTCamera] = {}
        for cam_id_str, cam_cfg in cfg["CAMERAS"].items():
            cam_id = int(cam_id_str)
            self.calibrations[cam_id] = CameraCalibration.load_from_json(
                cam_cfg["calibration"],
                cam_id,
            )
            self.workers[cam_id] = _SCTCamera(cam_id, cam_cfg["video"], sct_config)

        self.projector = PerspectiveProjector(cfg.get("PERSPECTIVE", {}))

        # These modules are the local, dependency-light port of
        # AIC2024_Track1_Nota/trackers/multicam_tracker/{clustering,cluster_track}.py.
        self.clusterer = CrossCameraClusterer(cfg.get("CLUSTERING", {}))
        self.cluster_tracker = ClusterTracker(cfg.get("MC_TRACKER", {}))
        self.id_distributor = IDDistributor()
        self.refinement_interval = cfg.get("MC_TRACKER", {}).get("refinement_interval", 5)
        self.local_to_global: Dict[tuple[int, int], int] = {}

        self.cam_ids = sorted(self.workers)
        self.cam_names = [f"Cam {cam_id}" for cam_id in self.cam_ids]

    def run(self, output_path: str | None = None, txt_dir: str | None = None):
        output_path = output_path or self.output_video
        txt_dir = txt_dir or self.output_txt_dir
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)

        mot_files = {
            cam_id: open(os.path.join(txt_dir, f"cam{cam_id}_mct_aic.txt"), "w")
            for cam_id in self.cam_ids
        }
        writer = None
        frame_count = 0
        t0 = time.time()

        logger.info("MCT AIC pipeline starting with %d cameras", len(self.workers))
        try:
            while True:
                active = False
                frames: Dict[int, np.ndarray] = {}
                per_cam_tracks: Dict[int, List[TrackInfo]] = {}

                for cam_id in self.cam_ids:
                    worker = self.workers[cam_id]
                    if not worker.stopped:
                        active = worker.process_next_frame() or active
                    if worker.latest_frame is None:
                        continue

                    frames[cam_id] = worker.latest_frame
                    tracks = list(worker.latest_tracks)
                    self.projector.compute_locations(tracks, self.calibrations[cam_id])
                    per_cam_tracks[cam_id] = tracks

                if not active:
                    break
                frame_count += 1

                self.id_distributor.reset()
                groups = self.clusterer.update(per_cam_tracks, self.id_distributor)
                self.cluster_tracker.update(groups)
                self.clusterer.update_using_cluster_tracker(
                    per_cam_tracks,
                    self.cluster_tracker,
                    identity_cache=self.local_to_global,
                )
                if frame_count % self.refinement_interval == 0:
                    self.cluster_tracker.refinement_clusters()

                self._write_mot(mot_files, per_cam_tracks, frame_count)
                writer = self._write_video(
                    writer,
                    output_path,
                    frames,
                    per_cam_tracks,
                    frame_count,
                    t0,
                )

                if frame_count % 100 == 0:
                    logger.info(
                        "MCT frame %d | active clusters=%d | fps=%.1f",
                        frame_count,
                        len([t for t in self.cluster_tracker.tracked_mtracks if t.is_activated]),
                        frame_count / max(time.time() - t0, 1e-6),
                    )
        finally:
            for worker in self.workers.values():
                worker.release()
            for fh in mot_files.values():
                fh.close()
            if writer is not None:
                writer.release()

        logger.info("MCT AIC done: video=%s txt_dir=%s", output_path, txt_dir)

    def _write_mot(
        self,
        mot_files: Dict[int, object],
        per_cam_tracks: Dict[int, List[TrackInfo]],
        frame_count: int,
    ):
        for cam_id, tracks in per_cam_tracks.items():
            fh = mot_files[cam_id]
            for track in tracks:
                if track.global_id is None or track.global_id < 0:
                    continue
                x1, y1, x2, y2 = track.bbox
                fh.write(
                    f"{frame_count},{int(track.global_id)},{x1:.1f},{y1:.1f},"
                    f"{x2 - x1:.1f},{y2 - y1:.1f},1,-1,-1,-1\n"
                )

    def _write_video(
        self,
        writer,
        output_path: str,
        frames: Dict[int, np.ndarray],
        per_cam_tracks: Dict[int, List[TrackInfo]],
        frame_count: int,
        t0: float,
    ):
        annotated = []
        active_ids = {
            track.track_id
            for track in self.cluster_tracker.tracked_mtracks
            if track.is_activated
        }
        for cam_id in self.cam_ids:
            if cam_id not in frames:
                continue
            annotated.append(
                self._draw(
                    frames[cam_id].copy(),
                    per_cam_tracks.get(cam_id, []),
                    active_ids,
                    cam_id,
                    frame_count,
                )
            )

        if not annotated:
            return writer

        grid = draw_grid(annotated, self.cam_names)
        gh, gw = grid.shape[:2]
        if gh % 2 or gw % 2:
            grid = grid[: gh - (gh % 2), : gw - (gw % 2)]

        fps = frame_count / max(time.time() - t0, 1e-6)
        cv2.putText(
            grid,
            f"Frame: {frame_count} | FPS: {fps:.1f} | Globals: {len(active_ids)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        if writer is None:
            h, w = grid.shape[:2]
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.output_fps,
                (w, h),
            )
        writer.write(grid)
        return writer

    def _draw(
        self,
        frame: np.ndarray,
        tracks: List[TrackInfo],
        active_global_ids: set,
        cam_id: int,
        frame_id: int,
    ) -> np.ndarray:
        np.random.seed(42)
        palette = np.random.randint(0, 255, size=(500, 3), dtype=np.uint8)

        for track in tracks:
            local_id = track.person_id
            global_id = track.global_id
            if local_id is None and global_id is None:
                continue

            is_global = global_id in active_global_ids if global_id is not None else False
            if is_global:
                color = tuple(int(c) for c in palette[int(global_id) % len(palette)])
            else:
                color = (128, 128, 128)
                if not self.draw_local:
                    continue

            labels = []
            if self.draw_local and local_id is not None:
                labels.append(f"L{int(local_id)}")
            if is_global:
                labels.append(f"G{int(global_id)}")
            label = " | ".join(labels)

            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            frame,
            f"Cam {cam_id} | F{frame_id}",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        return frame
