from typing import Dict
import cv2
import datetime
import numpy as np
import torch
import os
import time

from modules.pose_3d.geometry.camera import Camera
from modules.pose_3d.baseline import Pose3D

from modules.detection.yolo11.detect import Yolo11Detector
from modules.pose_estimation.rtmpose.pe import RTMPoseEstimator
from modules.embedding.fastreid.embed import Embedding
from modules.utils.box_utils import iou_box_i, select_boxes
from modules.utils.vis_utils import draw_poses, visualize_results


class CrossViewTracking:
    def __init__(self, config: Dict, input_config: Dict):
        self.detector = Yolo11Detector(config["DETECTION"]["yolov11"])
        self.pe = RTMPoseEstimator(config["POSE"]["rtmpose"])
        self.embed = Embedding(config["VECTORIZATION"]["fast_reid"])

        if input_config["INPUT"]["type"] == "stream":
            pass
        elif input_config["INPUT"]["type"] == "video":
            self.cap0 = cv2.VideoCapture(input_config["INPUT"]["cam_0"]["stream_path"])
            self.cap1 = cv2.VideoCapture(input_config["INPUT"]["cam_1"]["stream_path"])
            self.fps = input_config["INPUT"]["cam_0"]["fps"]
        else:
            raise ValueError("[ERROR]: Not support type of video")

        # load camera
        self.calibs = []
        for cid, cname in enumerate(["cam_0", "cam_1"]):
            extrinsic_file_path = input_config["INPUT"][cname]["extrinsic_file"]
            self.calibs.append(Camera.load_from_file(extrinsic_file_path, cid))

        self.pose_3d_mapper = Pose3D(config["POSE_3D"], self.calibs)

        # random màu cho kết quả matching
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(500, 3), dtype=np.uint8)

    def run(self, output_dir=None, file_name=None, visualize=False):
        frame_count = 0
        format = "%Y-%m-%d %H:%M:%S"
        start_time = datetime.datetime.strptime("2025-06-12 14:00:00", format)

        # writer để lưu video
        writer = None
        output_path = None
        if output_dir and file_name:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name)

        start_wall_time = time.time()

        while self.cap0.isOpened() and self.cap1.isOpened():
            ret0, frame0 = self.cap0.read()
            ret1, frame1 = self.cap1.read()
            if not ret0 or not ret1:
                break

            frame_count += 1
            print(f"[INFO]: Frame {frame_count}")
            
            frames = [frame0, frame1]

            # detection + pose estimation
            pair_poses = []
            for cid, frame in enumerate(frames):
                l_bboxes = self.detector.detect(frame)
                bboxes = select_boxes(l_bboxes)
                poses = self.pe.predict(frame, bboxes)
                ious = [0] * len(bboxes)
                for i, box_i in enumerate(bboxes):
                    ious_i = [iou_box_i(box_i, box_j) for j, box_j in enumerate(bboxes) if i != j]
                    ious[i] = max(ious_i) if ious_i else 0

                croped_img = [
                    frame[int(y1):int(y2), int(x1):int(x2)] for (x1, y1, x2, y2) in bboxes
                ]
                embeddings = (
                    self.embed.extract_feature(croped_img)
                    if len(croped_img)
                    else torch.empty((0, 512))
                )

                poses_formatted = [
                    [pose, embed, box, idx, iou]
                    for idx, (pose, embed, box, iou) in enumerate(
                        zip(poses, embeddings, bboxes, ious)
                    )
                ]
                pair_poses.append(poses_formatted)

            # 3D pose estimation / Matching
            predictions = self.pose_3d_mapper.estimate(pair_poses)
            
            # vẽ kết quả
            vis_frames = []
            for cid, frame in enumerate(frames):
                # Lưu thông tin theo cid của framework cũ
                vis_frame = frame.copy()
                poses_to_draw = []
                
                for idx, prediction in enumerate(predictions):
                    pose_3d, kpts_emb_box_bid_ious, cam_ids = prediction
                    
                    if cid in cam_ids:
                        cam_index = cam_ids.index(cid)
                        kpts, embed, box, bid, iou = kpts_emb_box_bid_ious[cam_index]
                        
                        x1, y1, x2, y2 = map(int, box)
                        poses_to_draw.append(kpts)

                        color = tuple(int(c) for c in self.colors[idx % len(self.colors)])
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                        # Auto-adjust label position to prevent clipping at the top boundary of the image
                        text_y = y1 - 10 if y1 - 10 >= 0 else y1 + 20
                        cv2.putText(
                            vis_frame,
                            f"Pair: {idx}",
                            (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

                vis_frame = draw_poses(vis_frame, poses_to_draw)
                vis_frames.append(vis_frame)

            vis_grid = visualize_results(vis_frames, camera_names=["Cam0", "Cam1"])

            # khởi tạo writer nếu chưa có
            if writer is None and output_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = vis_grid.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

            # tính fps real-time
            elapsed_time = time.time() - start_wall_time
            fps_display = frame_count / elapsed_time if elapsed_time > 0 else 0.0
            cv2.putText(
                vis_grid,
                f"Frame: {frame_count} | FPS: {fps_display:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            if visualize:
                cv2.namedWindow("Entrance Matching 3D", cv2.WINDOW_NORMAL)
                cv2.imshow("Entrance Matching 3D", vis_grid)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer:
                writer.write(vis_grid)

        self.cap0.release()
        self.cap1.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
