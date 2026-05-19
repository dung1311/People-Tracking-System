"""Reusable per-camera SCT worker for MCT pipelines."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from modules.data_templates.sct_template import TrackInfo
from modules.detector.factory import DetectorFactory
from modules.pose_estimator.factory import PoseEstimatorFactory
from modules.track_manager.single_track_manager import SingleTrackManager
from modules.tracker_2D.factory import TrackerFactory
from utils.pose import is_full_body


class CameraWorker:
    """Wraps detector + tracker + track_manager for one camera stream.

    Có thể inject ``detector`` / ``pose_estimator`` / ``embedder`` dùng chung
    giữa nhiều camera để giảm VRAM và thời gian load.
    """

    def __init__(
        self,
        cam_id: int,
        video_path: str,
        sct_config: dict,
        *,
        detector=None,
        pose_estimator=None,
        embedder=None,
        enable_pose_full_body: bool = True,
        use_mct_track_manager: bool = False,
    ):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(video_path)
        self.detector = (
            detector
            if detector is not None
            else DetectorFactory(sct_config["DETECTION"]).get_detector()
        )
        self.tracker = TrackerFactory(sct_config["TRACKING"]).get_tracker()
        self.enable_pose_full_body = enable_pose_full_body
        if enable_pose_full_body:
            self.pe = (
                pose_estimator
                if pose_estimator is not None
                else PoseEstimatorFactory(
                    sct_config["POSE_ESTIMATION"],
                ).get_pose_estimator()
            )
        else:
            self.pe = None
        if use_mct_track_manager:
            from modules.track_manager.mct_track_manager import MCTTrackManager

            self.track_manager = MCTTrackManager(
                sct_config["TRACK_MANAGER"],
                embedder=embedder,
            )
        else:
            self.track_manager = SingleTrackManager(
                sct_config["TRACK_MANAGER"],
                embedder=embedder,
            )
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

        if len(tracks) > 0 and self.pe is not None:
            track_boxes = [trk[:4] for trk in tracks]
            kpts_scores = self.pe.detect(frame, track_boxes)
            is_full_body_dict = {}
            for i, trk in enumerate(tracks):
                tracker_id = int(trk[4])
                is_full_body_dict[tracker_id] = (
                    is_full_body(kpts_scores[i], confidence_threshold=0.5)
                    if i < len(kpts_scores)
                    else False
                )
            frame_info["is_full_body"] = is_full_body_dict

        self.latest_tracks = self.track_manager.process(tracks, frame_info)
        self.latest_frame = frame
        return True

    @property
    def stopped(self) -> bool:
        return self._stopped

    def release(self):
        self.cap.release()
