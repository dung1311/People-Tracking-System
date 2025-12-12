from typing import List

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