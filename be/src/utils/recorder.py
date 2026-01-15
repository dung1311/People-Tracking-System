import cv2
import os
import time
from datetime import datetime
from sqlmodel import Session
from database.session import engine
from models.video_segment import VideoSegment

class VideoRecorder:
    def __init__(self, camera_id: int, save_dir: str = "data/recordings", segment_duration: int = 300):
        self.camera_id = camera_id
        self.base_dir = save_dir
        self.segment_duration = segment_duration
        self.current_writer = None
        self.current_segment_start = None
        self.current_file_path = None
        self.fps = None
        self.frame_size = None
        
        # Ensure base dir exists
        os.makedirs(self.base_dir, exist_ok=True)

    def start_segment(self, frame_size, fps):
        self.frame_size = frame_size
        self.fps = fps
        self.current_segment_start = datetime.now()
        
        # Structure: data/recordings/{camera_id}/{YYYY-MM-DD}/
        date_str = self.current_segment_start.strftime("%Y-%m-%d")
        cam_dir = os.path.join(self.base_dir, str(self.camera_id), date_str)
        os.makedirs(cam_dir, exist_ok=True)
        
        # Filename: HH-MM-SS.mp4
        time_str = self.current_segment_start.strftime("%H-%M-%S")
        self.current_file_path = os.path.join(cam_dir, f"{time_str}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.current_writer = cv2.VideoWriter(self.current_file_path, fourcc, self.fps, self.frame_size)
    
    def write_frame(self, frame):
        if frame is None:
            return

        if self.current_writer is None:
             h, w = frame.shape[:2]
             self.start_segment((w, h), 25.0) # Assume 25fps if not set, or update logic to detect
             
        # Check if segment duration exceeded
        if (datetime.now() - self.current_segment_start).total_seconds() > self.segment_duration:
            self.close_segment()
            h, w = frame.shape[:2]
            self.start_segment((w, h), self.fps)
            
        self.current_writer.write(frame)

    def close_segment(self):
        if self.current_writer:
            self.current_writer.release()
            
            # Save to DB
            end_time = datetime.now()
            duration = (end_time - self.current_segment_start).total_seconds()
            
            with Session(engine) as session:
                segment = VideoSegment(
                    camera_id=self.camera_id,
                    file_path=self.current_file_path,
                    start_time=self.current_segment_start,
                    end_time=end_time,
                    duration_seconds=duration
                )
                session.add(segment)
                session.commit()
            
            self.current_writer = None
            self.current_file_path = None

    def stop(self):
        self.close_segment()
