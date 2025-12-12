from typing import List
import math
import os

import cv2
import numpy as np  

from modules.data_templates.sct_template import TrackInfo

def draw_grid(frames: List[np.ndarray], camera_names: List[str]):
    if len(frames) == 1:
        return frames[0]

    num_cameras = len(frames)
    if camera_names and len(camera_names) == num_cameras:
        frame_names = camera_names
    else:
        frame_names = [f"Cam_{i+1}" for i in range(num_cameras)]
    
    num_cols = 2
    num_rows = math.ceil(num_cameras/num_cols)
    
    h, w = frames[0].shape[:2]
    resized_frames = []
    for i, f in enumerate(frames):
        frame_resize = cv2.resize(f, (w, h))
        cv2.putText(frame_resize, frame_names[i], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        resized_frames.append(frame_resize)
    
    grid_height = num_rows * h 
    grid_width = num_cols * w
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    for idx, frame in enumerate(resized_frames):
        row = idx // num_cols
        col = idx % num_cols
        y1, y2 = row*h, (row+1)*h
        x1, x2 = col*w, (col+1)*w
        grid[y1:y2, x1:x2] = frame
    
    return grid

def draw_tracks(frame, tracks: List[TrackInfo]):
    for track in tracks:

        tlbr = track.bbox
        track_id = track.tracker_id

        x1, y1, x2, y2 = map(int, tlbr)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"ID: {track_id}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame

def setup_video_writer(cap, output_path="output.mp4"):
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, output_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    return writer