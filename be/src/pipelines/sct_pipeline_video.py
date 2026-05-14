import os 
import logging 
from pathlib import Path

import cv2

from modules.detector.factory import DetectorFactory
from modules.tracker_2D.factory import TrackerFactory
from modules.pose_estimator.factory import PoseEstimatorFactory
from modules.track_manager.single_track_manager import SingleTrackManager
from utils.vis import draw_tracks, setup_video_writer
from utils.box import selection_boxes, filter_overlapping_boxes
from utils.pose import is_full_body

logger = logging.getLogger(__name__)

class SCTVideoPipeline:
    def __init__(self, config: dict, video_path: Path | str):
        self.detector = DetectorFactory(config["DETECTION"]).get_detector()
        self.tracker = TrackerFactory(config["TRACKING"]).get_tracker()
        self.pe = PoseEstimatorFactory(config["POSE_ESTIMATION"]).get_pose_estimator()
        self.track_manager = SingleTrackManager(config["TRACK_MANAGER"])
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.cap = cv2.VideoCapture(video_path)
        
    def run(self):
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        writer = setup_video_writer(self.cap, output_path=f'{self.video_name}.mp4')
        mot_file = open(f'outputs/txt/{self.video_name}.txt', 'w')
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            current_frame += 1
            
            bboxes = self.detector.detect(frame)
            kpts_scores = self.pe.detect(frame, bboxes)
            indices_to_keep = [i for i, kpt_score in enumerate(kpts_scores) if is_full_body(kpt_score, confidence_threshold=0.5)]
            selected_bboxes = [bboxes[i] for i in indices_to_keep]

            # Filter out boxes with max IoU > 0.5 (keep larger box)
            selected_bboxes = filter_overlapping_boxes(selected_bboxes, iou_threshold=0.3)
            
            frame_info = {
                    "cam_id": "1",
                    "frame_id": current_frame,
                    'frame': frame
                }
        
            tracks = self.tracker.update(selected_bboxes, frame_info)
            live_tracks = self.track_manager.process(tracks, frame_info)
            
            for track in live_tracks:
                pid = int(track.person_id)
                x1, y1, x2, y2 = map(int, track.bbox)
                w = x2 - x1
                h = y2 - y1
                mot_file.write(f"{current_frame},{pid},{x1},{y1},{w},{h},1,-1,-1,-1\n")
            
            annotated_frame = draw_tracks(frame, live_tracks, frame_info, pid_only=True)
            writer.write(annotated_frame)
            logger.info("Frame %d/%d", current_frame, total_frames)
        
        mot_file.close()
        self.cap.release()