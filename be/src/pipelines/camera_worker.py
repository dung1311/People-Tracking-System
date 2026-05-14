"""Reusable per-camera SCT worker for MCT pipelines."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from modules.data_templates.sct_template import TrackInfo
from modules.detector.factory import DetectorFactory
from modules.track_manager.single_track_manager import SingleTrackManager
from modules.tracker_2D.factory import TrackerFactory


class CameraWorker:
    """Wraps detector + tracker + track_manager for one camera stream."""

    def __init__(self, cam_id: int, video_path: str, sct_config: dict):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(video_path)
        self.detector = DetectorFactory(sct_config["DETECTION"]).get_detector()
        self.tracker = TrackerFactory(sct_config["TRACKING"]).get_tracker()
        self.track_manager = SingleTrackManager(sct_config["TRACK_MANAGER"])
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_tracks: List[TrackInfo] = []
        self.frame_id = 0
        self._stopped = False

    def process_next_frame(self) -> bool:
        """Read and process one frame. Returns False when the video ends."""
        ret, frame = self.cap.read()
        if not ret:
            self._stopped = True
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
        bboxes = self.detector.detect(frame)
        tracks = self.tracker.update(bboxes, frame_info)
        self.latest_tracks = self.track_manager.process(tracks, frame_info)
        self.latest_frame = frame
        return True

    @property
    def stopped(self) -> bool:
        return self._stopped

    def release(self):
        self.cap.release()
