from typing import List

import numpy as np
from rtmlib import Body
from .pose import is_full_body

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

def selection_boxes(detections: List[List[float]], min_area: float = 7000, ratio_range: tuple = (2.0, 3.0)) -> List[List[float]]:
    selected = []
    for det in detections:
        area = calculate_box_area(det)
        ratio = calculate_box_ratio(det)
        if area > min_area and ratio >= ratio_range[0] and ratio <= ratio_range[1]:
            selected.append(det)
            
    return selected

def calculate_box_ratio(box: List[float]) -> float:
    x1, y1, x2, y2 = box[:4]
    w = x2 - x1
    h = y2 - y1
    if w > 0:
        return h / w
    return 0.0

def calculate_box_area(box: List[float]) -> float:
    x1, y1, x2, y2 = box[:4]
    w = x2 - x1
    h = y2 - y1
    if w > 0 and h > 0:
        return w * h
    return 0.0

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2, ...]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def filter_overlapping_boxes(boxes: List[List[float]], iou_threshold: float = 0.5) -> List[List[float]]:
    """Remove both boxes in any pair whose IoU exceeds the threshold."""
    if len(boxes) <= 1:
        return boxes

    n = len(boxes)
    removed = [False] * n

    for i in range(n):
        for j in range(i + 1, n):
            if compute_iou(boxes[i], boxes[j]) > iou_threshold:
                removed[i] = True
                removed[j] = True

    return [boxes[i] for i in range(n) if not removed[i]]