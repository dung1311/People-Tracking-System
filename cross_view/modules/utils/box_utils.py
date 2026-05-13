import numpy as np

def iou_box_i(i, j):
    """
    calculate the ratio between intersection area and the smaller box.
    @a: array of first bounding box in format [xmin, ymin, xmax, ymax]
    @b: array of second bounding boxes in format [xmin, ymin, xmax, ymax]
    @return: area of (a and b)/ min( area of a, area of b)
    """
    w_intsec = np.maximum(0, (np.minimum(i[2], j[2]) - np.maximum(i[0], j[0])))
    h_intsec = np.maximum(0, (np.minimum(i[3], j[3]) - np.maximum(i[1], j[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (i[2] - i[0]) * (i[3] - i[1])
    s_b = (j[2] - j[0]) * (j[3] - j[1])
    return float(s_intsec) / s_a

from typing import Dict

def get_w_h_ratio_area_box(box):
    x1, y1, x2, y2 = box[:4]
    w = x2 - x1
    h = y2 - y1
    r = h/(1.0*w)
    area = w*h
    return w, h, r, area

def is_good_box(box, select_cfg: Dict = None):
    if not select_cfg:
        select_cfg = {
            "w_min": 20,
            "h_min": 40,
            "h_w_r_min": 1.5,
            "h_w_r_max": 4.0,
            "a_min": 450
        }
    w, h, r, area = get_w_h_ratio_area_box(box)
    if w < select_cfg["w_min"] or \
    h < select_cfg["h_min"] or \
    r < select_cfg["h_w_r_min"] or \
    r > select_cfg["h_w_r_max"] or \
    area < select_cfg["a_min"]: 
        return False
    
    return True

def select_boxes(batch_boxes, select_cfg: Dict = None):
    if not select_cfg:
        select_cfg = {
            "w_min": 20,
            "h_min": 40,
            "h_w_r_min": 1.8,
            "h_w_r_max": 4.0,
            "a_min": 450
        }
    ret = []
    for box in batch_boxes:
        w, h, r, area = get_w_h_ratio_area_box(box)
        if w < select_cfg["w_min"] or \
        h < select_cfg["h_min"] or \
        r < select_cfg["h_w_r_min"] or \
        r > select_cfg["h_w_r_max"] or \
        area < select_cfg["a_min"]: 
            continue
        
        ret.append(box)
    
    return ret
