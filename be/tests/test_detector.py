import sys
import os

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modules.detector.factory import DetectorFactory
from src.utils.load_config import load_config

import cv2

if __name__ == "__main__":
    # Resolve config path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/sct_config.yaml")
    
    cfg = load_config(config_path)
    detector = DetectorFactory(cfg["DETECTION"]).get_detector()
    
    cap = cv2.VideoCapture("/home/dungnt/workspaces/HUST/DATN/be/data/videos/video_2min.mp4")
    while cap.isOpened:
        ret, frame = cap.read()
        boxes = detector.detect(frame)
        print(boxes)
        break
    

