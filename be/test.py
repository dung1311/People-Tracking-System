import time

import cv2

from rtmlib import Body, draw_skeleton
from src.utils.pose import is_full_body
# import numpy as np

device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

cap = cv2.VideoCapture("/home/dungnt/workspaces/HUST/DATN/be/debug/full.jpg")

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

body = Body(
    mode='balanced',  # balanced, performance, lightweight
    backend=backend,
    device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break
    s = time.time()
    keypoints, scores = body(frame)
    print(scores)
    print(is_full_body(scores, confidence_threshold=0.4))
    det_time = time.time() - s
    print('det: ', det_time)

    img_show = frame.copy()

    # if you want to use black background instead of original image,
    # img_show = np.zeros(img_show.shape, dtype=np.uint8)

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.3,
                             line_width=2)

    img_show = cv2.resize(img_show, (960, 640))
    cv2.imshow('img', img_show)
    cv2.waitKey(0)