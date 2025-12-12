import sys
import os
from typing import List

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modules.detector.factory import DetectorFactory
from src.modules.embedder.factory import EmbedderFactory
from src.utils.load_config import load_config

import cv2
import numpy as np

def crop_detections(frame_img: np.ndarray, detections: List[List[float]]) -> List[np.ndarray]:
    crops = []
    img_h, img_w = frame_img.shape[:2]
    
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img_w, int(x2))
        y2 = min(img_h, int(y2))

        crop = frame_img[y1:y2, x1:x2]
        
        if crop.size > 0:
            crops.append(crop)
    
    return crops

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/sct_config.yaml")
    
    cfg = load_config(config_path)
    detector = DetectorFactory(cfg["DETECTION"]).get_detector()
    embedder = EmbedderFactory(cfg["EMBEDDING"]).get_embedder()
    cap = cv2.VideoCapture("/home/dungnt/workspaces/HUST/DATN/be/data/videos/video_2min.mp4")
    while cap.isOpened:
        ret, frame = cap.read()
        boxes = detector.detect(frame)
        embeddings = embedder.extract_feature(crop_detections(frame, boxes))
        print(len(embeddings))
        break
    
    