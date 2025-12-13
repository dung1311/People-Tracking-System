import os
import logging
from typing import Dict, List
from datetime import datetime

import cv2
import numpy as np
from sqlmodel import Session, delete  # Thêm import delete

from modules.detector.factory import DetectorFactory
from modules.pose_estimator.factory import PoseEstimatorFactory
from modules.tracker_2D.factory import TrackerFactory
from modules.track_manager.single_track_manager import SingleTrackManager
from utils.vis import draw_tracks, setup_video_writer
from utils.io import FileVideoStream

from database.session import engine
from models.track import Track

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, pipeline_config: Dict, input_config: Dict, camera_id: int):
        self.detector = DetectorFactory(pipeline_config["DETECTION"]).get_detector()
        # self.pe = PoseEstimatorFactory(pipeline_config["POSE_ESTIMATOR"]).get_pose_estimator()
        self.tracker = TrackerFactory(pipeline_config["TRACKING"]).get_tracker()
        
        self.track_manager = SingleTrackManager(pipeline_config["TRACK_MANAGER"])
        video_path = input_config["video_path"]
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        self.cap = cv2.VideoCapture(video_path)
        self.camera_id = camera_id
    
    def run(self):
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        writer = setup_video_writer(self.cap, output_path=f'{self.video_name}.mp4')
        
        # Batch configuration
        BATCH_SIZE = 100
        track_buffer = []

        with Session(engine) as session:
            # --- CLEANUP: Delete old tracks for this camera before processing ---
            try:
                # Delete logic: DELETE FROM track WHERE camera_id = self.camera_id
                statement = delete(Track).where(Track.camera_id == self.camera_id)
                session.exec(statement)
                session.commit()
                logger.info(f"Successfully cleared old tracks for camera_id: {self.camera_id}")
            except Exception as e:
                logger.error(f"Error clearing old data: {e}")
                session.rollback()
            # ------------------------------------------------------------------

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                current_frame += 1
                
                boxes = self.detector.detect(frame)
                frame_info = {
                    "cam_id": self.camera_id,
                    "frame_id": current_frame,
                    'frame': frame
                }
                
                tracks = self.tracker.update(boxes, frame_info)
                live_tracks = self.track_manager.process(tracks, frame_info)
                
                # Prepare data for database
                for track_info in live_tracks:
                    feat_array = track_info.get_representative_feature()
                    
                    if feat_array is not None:
                        # Handle bbox and score extraction
                        bbox_data = track_info.bbox
                        score = 0.0
                        final_box = []

                        if len(bbox_data) >= 5:
                            final_box = [float(x) for x in bbox_data[:4]]
                            score = float(bbox_data[4])
                        else:
                            final_box = [float(x) for x in bbox_data[:4]]
                            score = 1.0

                        track_db = Track(
                            camera_id=self.camera_id,
                            person_id=track_info.person_id,
                            frame_id=current_frame,
                            bbox=final_box,
                            score=score,
                            class_id=0,
                            timestamp=datetime.now(),
                            feature=feat_array.tolist() 
                        )
                        
                        track_buffer.append(track_db)

                # Save batch if limit reached
                if len(track_buffer) >= BATCH_SIZE:
                    try:
                        session.add_all(track_buffer)
                        session.commit()
                        track_buffer.clear()
                    except Exception as e:
                        logger.error(f"Error saving batch: {e}")
                        session.rollback()
                        track_buffer.clear()

                annotated_frame = draw_tracks(frame, live_tracks, frame_info)
                writer.write(annotated_frame)
            
            # Flush remaining tracks in buffer
            if track_buffer:
                try:
                    session.add_all(track_buffer)
                    session.commit()
                except Exception as e:
                    logger.error(f"Error saving final batch: {e}")
                    session.rollback()
                
        self.cap.release()
        writer.release()
        cv2.destroyAllWindows()