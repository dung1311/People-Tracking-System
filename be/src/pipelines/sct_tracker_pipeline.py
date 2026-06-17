from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from modules.data_templates.sct_template import TrackInfo
from modules.detector.factory import DetectorFactory
from modules.tracker_2D.factory import TrackerFactory
from utils.box import filter_overlapping_boxes
from utils.vis import draw_tracks, setup_video_writer

logger = logging.getLogger(__name__)


class SCTTrackerPipeline:
    """SCT pipeline that takes final local IDs directly from SORT/OCSort."""

    def __init__(self, config: dict, video_path: Path | str):
        self.config = config
        self.video_path = str(video_path)
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]

        self.detector = DetectorFactory(config["DETECTION"]).get_detector()
        self.tracker = TrackerFactory(
            config["TRACKING"],
            config.get("TRACK_MANAGER"),
        ).get_tracker()
        self.cap = cv2.VideoCapture(self.video_path)

    def run(
        self,
        output_path: str | None = None,
        txt_path: str | None = None,
    ):
        output_path = output_path or f"{self.video_name}_tracker_sct.mp4"
        txt_path = txt_path or f"outputs/txt/{self.video_name}_tracker_sct.txt"
        os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = setup_video_writer(self.cap, output_path=output_path)
        mot_file = open(txt_path, "w")

        frame_id = 0
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_id += 1

                h, w = frame.shape[:2]
                detections = self.detector.detect(frame)
                detections = filter_overlapping_boxes(detections, iou_threshold=0.3)

                frame_info = {
                    "cam_id": 1,
                    "frame_id": frame_id,
                    "frame": frame,
                    "img_info": (h, w),
                    "img_size": (h, w),
                }
                tracker_rows = self.tracker.update(detections, frame_info)
                tracks = self._rows_to_trackinfo(tracker_rows, frame_info)

                for track in tracks:
                    pid = int(track.person_id)
                    x1, y1, x2, y2 = map(int, track.bbox)
                    mot_file.write(
                        f"{frame_id},{pid},{x1},{y1},{x2 - x1},{y2 - y1},1,-1,-1,-1\n"
                    )

                writer.write(draw_tracks(frame, tracks, frame_info, pid_only=True))
                if frame_id % 50 == 0:
                    logger.info("SCT frame %d/%d", frame_id, total_frames)
        finally:
            mot_file.close()
            writer.release()
            self.cap.release()

        logger.info("SCT done: video=%s txt=%s", output_path, txt_path)

    def _rows_to_trackinfo(self, rows: np.ndarray, frame_info: dict) -> List[TrackInfo]:
        tracks: List[TrackInfo] = []
        for row in rows:
            person_id = int(row[4])
            track = TrackInfo(
                tracker_id=person_id,
                bbox=row[:4].tolist() if hasattr(row, "tolist") else list(row[:4]),
                score=float(row[5]) if len(row) > 5 else 1.0,
                class_id=int(row[6]) if len(row) > 6 else 0,
                cam_id=frame_info["cam_id"],
                frame_id=frame_info["frame_id"],
            )
            track.person_id = person_id

            feat = self.tracker.identity.get_feature(person_id) if hasattr(self.tracker, "identity") else None
            if feat is not None:
                track.features = [feat]
            tracks.append(track)
        return tracks
