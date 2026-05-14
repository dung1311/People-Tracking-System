"""Multi-Camera Tracking pipeline.

Each camera runs its own SCT pipeline independently in a background thread.
The MCT layer collects per-camera results every frame and performs
cross-camera matching to assign global person IDs.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from modules.data_templates.mct_template import (
    CameraCalibration,
    CameraPairCalibration,
)
from modules.data_templates.sct_template import TrackInfo
from modules.detector.factory import DetectorFactory
from modules.matching.cross_camera_matching import CrossCameraMatcher
from modules.track_manager.global_track_manager import GlobalTrackManager
from modules.track_manager.single_track_manager import SingleTrackManager
from modules.tracker_2D.factory import TrackerFactory
from utils.vis import draw_grid

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Hardcoded config (temporary)
# -----------------------------------------------------------------------

MCT_CONFIG = {
    "matching_interval": 1,
    "weights": {
        "epipolar": 0,
        "homography": 1,
        "visual": 0,
        "frechet": 0,
    },
    "thresholds": {
        "epipolar": 100.0,
        "homography": 50,
        "visual": 0.5,
        "frechet": 300.0,
        "combined": 0.7,
    },
    "global_track": {
        "max_lost_age": 300,
        "max_features": 50,
        "max_trajectory_len": 100,
    },
}


# -----------------------------------------------------------------------
# Per-camera worker (runs in its own thread)
# -----------------------------------------------------------------------

class _CameraWorker:
    """Wraps detector + tracker + SCT track-manager for one camera."""

    def __init__(self, cam_id: int, video_path: str, sct_config: dict):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(video_path)
        self.detector = DetectorFactory(sct_config["DETECTION"]).get_detector()
        self.tracker = TrackerFactory(sct_config["TRACKING"]).get_tracker()
        self.track_manager = SingleTrackManager(sct_config["TRACK_MANAGER"])

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_tracks: List[TrackInfo] = []
        self._frame_id = 0
        self._stopped = False

    def process_next_frame(self) -> bool:
        """Read one frame and run the full SCT pipeline.

        Returns False when the video is exhausted.
        """
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
        live_tracks = self.track_manager.process(tracks, frame_info)

        with self._lock:
            self._latest_frame = frame
            self._latest_tracks = live_tracks

        return True

    def get_latest(self) -> Tuple[Optional[np.ndarray], List[TrackInfo], int]:
        with self._lock:
            return self._latest_frame, list(self._latest_tracks), self._frame_id

    @property
    def stopped(self) -> bool:
        return self._stopped

    def release(self):
        self.cap.release()


# -----------------------------------------------------------------------
# MCT Pipeline
# -----------------------------------------------------------------------

class MCTPipeline:
    """Orchestrates multiple cameras with cross-camera matching."""

    def __init__(
        self,
        sct_config: dict,
        camera_video_map: Dict[int, str],
        camera_calib_map: Dict[int, str],
        mct_config: dict | None = None,
    ):
        """
        Args:
            sct_config: shared SCT config dict (detection, tracking, track_manager).
            camera_video_map: ``{cam_id: video_path}``.
            camera_calib_map: ``{cam_id: calibration_json_path}``.
            mct_config: optional override for ``MCT_CONFIG``.
        """
        cfg = mct_config or MCT_CONFIG
        self.matching_interval: int = cfg.get("matching_interval", 1)

        # Camera calibrations
        self.calibrations: Dict[int, CameraCalibration] = {}
        for cam_id, calib_path in camera_calib_map.items():
            self.calibrations[cam_id] = CameraCalibration.load_from_json(calib_path, cam_id)

        # Pre-compute pair calibrations for every camera pair
        cam_ids = sorted(self.calibrations.keys())
        self.pair_calibrations: Dict[Tuple[int, int], CameraPairCalibration] = {}
        for ci, cj in itertools.combinations(cam_ids, 2):
            pair = CameraPairCalibration(self.calibrations[ci], self.calibrations[cj])
            self.pair_calibrations[(ci, cj)] = pair

        # Per-camera SCT workers
        self.workers: Dict[int, _CameraWorker] = {}
        for cam_id, video_path in camera_video_map.items():
            self.workers[cam_id] = _CameraWorker(cam_id, video_path, sct_config)

        # Global track manager + matcher
        self.global_manager = GlobalTrackManager(cfg.get("global_track"))
        self.matcher = CrossCameraMatcher(
            weights=cfg.get("weights"),
            thresholds=cfg.get("thresholds"),
        )

        self.cam_names = [f"Cam {cid}" for cid in cam_ids]

    # ------------------------------------------------------------------
    # Run (synchronous frame-by-frame for offline video processing)
    # ------------------------------------------------------------------

    def run(self, output_path: str = "outputs/mct_output.mp4", txt_dir: str = "outputs/txt"):
        """Process all cameras frame by frame and write a grid video."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)

        # MOT15 txt files: one per camera, using global IDs
        mot_files: Dict[int, object] = {}
        for cam_id in self.workers:
            mot_files[cam_id] = open(os.path.join(txt_dir, f"cam{cam_id}_mct.txt"), "w")

        writer = None
        frame_count = 0
        start_time = time.time()

        logger.info("MCT pipeline starting with %d cameras", len(self.workers))

        try:
            while True:
                # Step each camera one frame
                any_alive = False
                for worker in self.workers.values():
                    if not worker.stopped:
                        if worker.process_next_frame():
                            any_alive = True
                if not any_alive:
                    break

                frame_count += 1

                # Collect per-camera results
                per_cam_tracks: Dict[int, List[TrackInfo]] = {}
                frames: Dict[int, np.ndarray] = {}
                for cam_id, worker in self.workers.items():
                    frame, tracks, _ = worker.get_latest()
                    if frame is not None:
                        frames[cam_id] = frame
                        per_cam_tracks[cam_id] = tracks

                # Cross-camera matching
                if frame_count % self.matching_interval == 0:
                    all_match_results = []
                    for (ci, cj), pair_cal in self.pair_calibrations.items():
                        tracks_i = per_cam_tracks.get(ci, [])
                        tracks_j = per_cam_tracks.get(cj, [])
                        results = self.matcher.match(
                            tracks_i, tracks_j, pair_cal,
                            global_tracks=self.global_manager.tracks,
                        )
                        all_match_results.extend(results)

                    active_globals = self.global_manager.update(
                        all_match_results, per_cam_tracks,
                        self.calibrations, frame_count,
                    )
                else:
                    active_globals = [
                        t for t in self.global_manager.tracks.values()
                        if t.state.name == "ACTIVE"
                    ]

                # Write MOT15 txt per camera (global ID)
                for cam_id, tracks in per_cam_tracks.items():
                    f = mot_files.get(cam_id)
                    if f is None:
                        continue
                    for t in tracks:
                        if t.person_id is None:
                            continue
                        gid = self.global_manager.get_global_id_for(cam_id, t.person_id)
                        if gid is None:
                            continue
                        x1, y1, x2, y2 = t.bbox
                        w = x2 - x1
                        h = y2 - y1
                        # MOT15: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
                        f.write(f"{frame_count},{gid},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},1,-1,-1,-1\n")

                # Annotate frames with global IDs
                annotated_frames = []
                for cam_id in sorted(frames.keys()):
                    annotated = self._draw_global_ids(
                        frames[cam_id].copy(),
                        per_cam_tracks.get(cam_id, []),
                        cam_id,
                        frame_count,
                    )
                    annotated_frames.append(annotated)

                if not annotated_frames:
                    continue

                grid = draw_grid(annotated_frames, self.cam_names)

                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(
                    grid,
                    f"Frame: {frame_count} | FPS: {fps:.1f} | Globals: {len(active_globals)}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

                if writer is None:
                    h, w = grid.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, 25, (w, h))

                writer.write(grid)

                if frame_count % 100 == 0:
                    stats = self.global_manager.tracks
                    logger.info(
                        "Frame %d | %d global tracks | FPS %.1f",
                        frame_count, len(stats), fps,
                    )

        except KeyboardInterrupt:
            logger.info("MCT pipeline interrupted by user")
        finally:
            for worker in self.workers.values():
                worker.release()
            if writer is not None:
                writer.release()
            for f in mot_files.values():
                f.close()
            logger.info(
                "MCT pipeline finished after %d frames. MOT15 files in %s/",
                frame_count, txt_dir,
            )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _draw_global_ids(
        self,
        frame: np.ndarray,
        tracks: List[TrackInfo],
        cam_id: int,
        frame_id: int,
    ) -> np.ndarray:
        """Draw bboxes labelled with ``local_pid -> global_id``."""
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(500, 3), dtype=np.uint8)

        for t in tracks:
            if t.person_id is None:
                continue
            gid = self.global_manager.get_global_id_for(cam_id, t.person_id)
            label = f"L{t.person_id}"
            if gid is not None:
                label += f"->G{gid}"
                color = tuple(int(c) for c in colors[gid % len(colors)])
            else:
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
            frame, f"Cam {cam_id} | Frame {frame_id}",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
        return frame
