from typing import Dict
import cv2
import numpy as np

from modules.detection.factory import get_detector
from modules.tracker.factory import get_tracker_sct

class SCT:
    def __init__(self, config: Dict, input_config: Dict):
        self.config = config
        self.input_config = input_config

        detector_name = config['DETECTION']['name']
        self.detector = get_detector(detector_name, config["DETECTION"][detector_name])

        tracker_name = config['TRACKING']['name']
        self.tracker = get_tracker_sct(tracker_name, config["TRACKING"][tracker_name])

        self.cap = cv2.VideoCapture("/home/dungnt/People-Tracking-System/backend/ai/assets/video_2mins/camera_0361_2mins.mp4")

    def run(self):
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame)

            dets = np.array(detections)
            tracks = self.tracker.update(dets)
            print("Tracks:", tracks)

    
