from typing import List, Dict
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

def draw_tracks(frame, tracks: List[TrackInfo], frame_info: Dict = None):
    frame_id = frame_info["frame_id"]
    frame_label = f"Frame: {frame_id}"

    # ===== Draw frame_id =====
    cv2.putText(
        frame,
        frame_label,
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    for track in tracks:
        tlbr = track.bbox
        tid = int(track.tracker_id)
        pid = int(track.person_id)

        x1, y1, x2, y2 = map(int, tlbr)
        box_w = x2 - x1

        # Draw bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"ID:{tid}->{pid}"

        # ===== Auto scale text =====
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        font_scale = 0.8
        min_scale = 0.35

        while font_scale > min_scale:
            (text_w, text_h), _ = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            if text_w <= box_w - 4:
                break
            font_scale -= 0.05

        # Background
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - 6),
            (x1 + min(text_w, box_w), y1),
            (0, 255, 0),
            -1
        )

        # Text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

    return frame


def setup_video_writer(cap, output_path="output.mp4"):
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, output_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    return writer