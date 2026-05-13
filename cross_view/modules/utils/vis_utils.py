import cv2
import matplotlib.pyplot as plt
from typing import List
import math
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), 
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

def visualize_3d_skeleton(pose_3ds, ax):
    ax.clear() 

    for pid, pose in enumerate(pose_3ds):
        if pose is None or all(p is None for p in pose):
            continue

        joints = [p for p in pose if p is not None]
        joints = np.array(joints).reshape(-1, 3)

        color = np.random.rand(3,)

        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                   c=[color], s=30, label=f"Pose {pid}")

        for (i, j) in SKELETON:
            if i < len(pose) and j < len(pose):
                if pose[i] is not None and pose[j] is not None:
                    ax.plot([pose[i][0], pose[j][0]],
                            [pose[i][1], pose[j][1]],
                            [pose[i][2], pose[j][2]], c=color)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    max_range = 2
    ax.set_xlim([-10, 20])
    ax.set_ylim([-10, 20])
    ax.set_zlim([0, 2])
    ax.view_init(elev=15, azim=0)
    plt.draw()
    plt.pause(0.001)  # update frame


def draw_poses(frame: np.ndarray, poses, conf_thres: float = 0.3):
    for pose in poses:
        for (x, y, score) in pose:
            if score > conf_thres:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        for (i, j) in SKELETON:
            if i < len(pose) and j < len(pose):
                if pose[i][2] > conf_thres and pose[j][2] > conf_thres:
                    pt1 = (int(pose[i][0]), int(pose[i][1]))
                    pt2 = (int(pose[j][0]), int(pose[j][1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

    return frame


def visualize_results(frames: List[np.ndarray], camera_names: List[str] = None):
    if len(frames) == 1:
        return frames[0]

    num_cameras = len(frames)
    
    if camera_names and len(camera_names) == num_cameras:
        frame_names = camera_names
    else:
        frame_names = [f"Camera_{i}" for i in range(num_cameras)]

    num_cols = 2
    num_rows = math.ceil(num_cameras / num_cols)

    height, width = frames[0].shape[:2]
    resized_frames = []
    for i, f in enumerate(frames):
        frame_resized = cv2.resize(f, (width, height))
        cv2.putText(frame_resized, frame_names[i], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        resized_frames.append(frame_resized)

    grid_height = num_rows * height
    grid_width = num_cols * width
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for idx, frame in enumerate(resized_frames):
        row = idx // num_cols
        col = idx % num_cols
        y1, y2 = row * height, (row + 1) * height
        x1, x2 = col * width, (col + 1) * width
        grid[y1:y2, x1:x2] = frame

    return grid


