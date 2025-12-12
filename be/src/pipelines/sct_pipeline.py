import logging
from typing import Dict, List
from datetime import datetime

import cv2
from sqlmodel import Session

from modules.detector.factory import DetectorFactory
from modules.pose_estimator.factory import PoseEstimatorFactory
from modules.embedder.factory import EmbedderFactory
from modules.tracker_2D.factory import TrackerFactory
from modules.track_manager.single_track_manager import SingleTrackManager
from utils.box import crop_detections
from utils.vis import draw_tracks, setup_video_writer

from database.session import engine
from models.track import Track

logging.getLogger(__name__)


class Pipeline:
    def __init__(self, pipeline_config: Dict, input_config: Dict, camera_id: int):
        self.detector = DetectorFactory(pipeline_config["DETECTION"]).get_detector()
        # self.pe = PoseEstimatorFactory(pipeline_config["POSE_ESTIMATOR"]).get_pose_estimator()
        self.tracker = TrackerFactory(pipeline_config["TRACKING"]).get_tracker()
        
        self.track_manager = SingleTrackManager(pipeline_config["TRACK_MANAGER"])
        
        video_path = input_config.get("video_path", "/home/dungnt/People-Tracking-System/be/data/videos/video_2min.mp4")
        self.cap = cv2.VideoCapture(video_path)
        self.camera_id = camera_id
    
    def run(self):
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        writer = setup_video_writer(self.cap, output_path="Video_2min.mp4")
           
        with Session(engine) as session:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                current_frame += 1
                if current_frame % 10 == 0:
                    print(f"Đang xử lý frame {current_frame}/{total_frames}", end="\r")

                boxes = self.detector.detect(frame)
                frame_info = {
                    "cam_id": self.camera_id,
                    "frame_id": current_frame
                }
                tracks = self.tracker.update(boxes, frame_info)
                live_tracks = self.track_manager.process(tracks, frame_info)
                annotated_frame = draw_tracks(frame, live_tracks)
                writer.write(annotated_frame)
                
                
        self.cap.release()
        writer.release()
        cv2.destroyAllWindows()
