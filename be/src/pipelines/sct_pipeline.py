import logging
import time
import os
from pathlib import Path
from queue import Queue, Empty
import cv2
import numpy as np
from sqlalchemy.orm import Session
from sqlmodel import select

from modules.detector.factory import DetectorFactory
from modules.tracker_2D.factory import TrackerFactory
from modules.track_manager.single_track_manager import SingleTrackManager
from utils.vis import draw_tracks
from utils.io import WebcamVideoStream
from database.session import get_session, engine
from models.track import Track
from utils.recorder import VideoRecorder

logger = logging.getLogger(__name__)

class SCTPipeline:
    def __init__(self, config: dict, input_config: dict, camera_id: int):
        self.config = config
        self.camera_id = camera_id
        video_path = input_config.get("video_path", 0)
        
        # Initialize modules
        self.detector = DetectorFactory(config["DETECTION"]).get_detector()
        self.tracker = TrackerFactory(config["TRACKING"]).get_tracker()
        self.track_manager = SingleTrackManager(config["TRACK_MANAGER"])
        
        # Stream setup
        self.cap = WebcamVideoStream(src=video_path).start()
        self.running = False
        
        # Recorder
        self.recorder = VideoRecorder(camera_id=camera_id)
        
    def run_generator(self):
        """
        Generator that yields annotated frames (numpy arrays).
        """
        self.running = True
        logger.info(f"Starting pipeline for Camera {self.camera_id}")
        
        current_frame = 0
        
        try:
            with Session(engine) as session:
                while self.running:
                    frame = self.cap.read()
                    
                    if frame is None:
                        if self.cap.stopped:
                            break
                        time.sleep(0.01)
                        continue
                        
                    current_frame += 1
                    
                    # 0. Recording
                    self.recorder.write_frame(frame)
                    
                    # 1. Detection
                    bboxes = self.detector.detect(frame)
                    
                    h, w = frame.shape[:2]
                    frame_info = {
                        "cam_id": self.camera_id,
                        "frame_id": current_frame,
                        'frame': frame,
                        "img_info": (h, w),
                        "img_size": (h, w)
                    }
                    
                    # 2. Tracking
                    tracks = self.tracker.update(bboxes, frame_info)
                    
                    # Treat all tracks as valid full body tracks in camera database pipeline
                    frame_info["is_full_body"] = {int(trk[4]): True for trk in tracks}
                    
                    # 3. Track Management (Re-ID, etc)
                    live_tracks = self.track_manager.process(tracks, frame_info)
                    
                    # 4. Save to Database
                    if not self.running:
                        break
                        
                    try:
                        db_tracks = []
                        for t in live_tracks:
                            if t.person_id is None: 
                                continue
                                
                            # Convert feature to list if it's numpy
                            feat = t.features[-1].tolist() if t.features else None
                            
                            db_track = Track(
                                camera_id=self.camera_id,
                                person_id=t.person_id,
                                frame_id=t.frame_id,
                                bbox=t.bbox,
                                score=t.score,
                                class_id=t.class_id,
                                timestamp=t.timestamp,
                                feature=feat
                            )
                            session.add(db_track)
                        
                        session.commit()
                    except Exception as e:
                        logger.error(f"Error saving tracks to DB: {e}")
                        session.rollback()
                    
                    # 5. Visualization
                    annotated_frame = draw_tracks(frame.copy(), live_tracks, frame_info)
                    yield annotated_frame

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
        finally:
            self.stop()
            logger.info("Pipeline generator finished.")

    def stream_generator(self):
        """
        Generator that yields MJPEG frames for streaming.
        Wrapper around run_generator.
        """
        for annotated_frame in self.run_generator():
            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.stop()
        if self.recorder:
            self.recorder.stop()

# Alias for compatibility with StreamManager
Pipeline = SCTPipeline