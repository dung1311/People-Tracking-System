import logging
import json
import numpy as np
import cv2
import time
import os
from typing import Dict, List, Optional
from scipy.optimize import linear_sum_assignment

from modules.detector.factory import DetectorFactory
from modules.tracker_2D.factory import TrackerFactory
from modules.track_manager.single_track_manager import SingleTrackManager
from utils.vis import draw_grid, draw_tracks

logger = logging.getLogger(__name__)

def frechet_distance(P, Q):
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return float('inf')
    
    ca = np.zeros((n, m))
    ca[0, 0] = np.linalg.norm(P[0] - Q[0])
    
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], np.linalg.norm(P[i] - Q[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], np.linalg.norm(P[0] - Q[j]))
        
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]), np.linalg.norm(P[i] - Q[j]))
            
    return ca[n-1, m-1]

def project_to_bev(bbox, H):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    bc_x = x1 + w / 2.0
    bc_y = y2
    pt = np.array([bc_x, bc_y, 1.0])
    proj = H @ pt
    if proj[2] == 0:
        return np.array([0, 0])
    return np.array([proj[0] / proj[2], proj[1] / proj[2]])


class TrackCopy:
    def __init__(self, t, new_pid):
        self.bbox = t.bbox
        self.tracker_id = t.tracker_id
        self.person_id = new_pid

class CameraWorker:
    def __init__(self, cam_id: int, video_path: str, sct_config: dict):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(video_path)
        self.detector = DetectorFactory(sct_config["DETECTION"]).get_detector()
        self.tracker = TrackerFactory(sct_config["TRACKING"]).get_tracker()
        self.track_manager = SingleTrackManager(sct_config["TRACK_MANAGER"])
        self.frame_id = 0
        self.stopped = False

    def process_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stopped = True
            return False, None, []
        
        self.frame_id += 1
        h, w = frame.shape[:2]
        frame_info = {
            "cam_id": self.cam_id,
            "frame_id": self.frame_id,
            "frame": frame,
            "img_info": (h, w),
            "img_size": (h, w),
        }

        bboxes = self.detector.detect(frame)
        tracks = self.tracker.update(bboxes, frame_info)
        
        # Treat all tracks as valid full body tracks in camera database pipeline
        frame_info["is_full_body"] = {int(trk[4]): True for trk in tracks}
        
        live_tracks = self.track_manager.process(tracks, frame_info)
        return True, frame, live_tracks

    def release(self):
        self.cap.release()


class FrechetMCTPipeline:
    def __init__(
        self,
        sct_config: dict,
        camera_video_map: Dict[int, str],
        camera_calib_map: Dict[int, str],
        window_size: int = 30
    ):
        self.window_size = window_size
        self.cam_ids = sorted(list(camera_video_map.keys()))
        
        # Load calibrations
        self.homographies = {}
        for cam_id, calib_path in camera_calib_map.items():
            with open(calib_path, 'r') as f:
                data = json.load(f)
                self.homographies[cam_id] = np.array(data['homography matrix'])

        # Per-camera workers
        self.workers = {}
        for cam_id, video_path in camera_video_map.items():
            self.workers[cam_id] = CameraWorker(cam_id, video_path, sct_config)

        self.global_ids = {}
        self.next_global_id = 1
        self.cam_names = [f"Cam {cid}" for cid in self.cam_ids]

    def get_or_create_global_id(self, cam_id, local_id):
        if (cam_id, local_id) not in self.global_ids:
            self.global_ids[(cam_id, local_id)] = self.next_global_id
            self.next_global_id += 1
        return self.global_ids[(cam_id, local_id)]

    def merge_global_ids(self, camA, localA, camB, localB):
        gA = self.get_or_create_global_id(camA, localA)
        gB = self.get_or_create_global_id(camB, localB)
        if gA != gB:
            min_g = min(gA, gB)
            max_g = max(gA, gB)
            for k, v in list(self.global_ids.items()):
                if v == max_g:
                    self.global_ids[k] = min_g

    def match_cameras(self, traj_A, traj_B, dist_thresh=10.0):
        tids_A = list(traj_A.keys())
        tids_B = list(traj_B.keys())
        
        if not tids_A or not tids_B:
            return []
            
        cost_matrix = np.full((len(tids_A), len(tids_B)), 1e9)
        for i, tid_A in enumerate(tids_A):
            for j, tid_B in enumerate(tids_B):
                frames_A = set(traj_A[tid_A].keys())
                frames_B = set(traj_B[tid_B].keys())
                common_frames = sorted(list(frames_A.intersection(frames_B)))
                
                if not common_frames:
                    continue
                    
                pts_A = np.array([traj_A[tid_A][f] for f in common_frames])
                pts_B = np.array([traj_B[tid_B][f] for f in common_frames])
                
                # Check Euclidean distance at identical timestamps
                max_euclidean = np.max(np.linalg.norm(pts_A - pts_B, axis=1))
                if max_euclidean > dist_thresh:
                    continue
                    
                cost_matrix[i, j] = frechet_distance(pts_A, pts_B)
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        for r, c in zip(row_ind, col_ind):
            dist = cost_matrix[r, c]
            if dist < 1e9:
                matches.append((tids_A[r], tids_B[c], dist))
                
        return matches

    def run(self, output_path="outputs/frechet_mct_output.mp4"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Starting Frechet Pipeline with window_size={self.window_size}")
        frame_count = 0
        start_time = time.time()
        
        writer = None
        
        # Buffer format: cam_id -> {track_id: list of points}
        traj_buffer = {cid: {} for cid in self.cam_ids}
        # Buffer to keep frames and tracks for the current window
        window_frames = []

        try:
            while True:
                any_alive = False
                frame_data = {}
                for cam_id, worker in self.workers.items():
                    if worker.stopped:
                        continue
                    ret, frame, live_tracks = worker.process_next_frame()
                    if ret:
                        any_alive = True
                        H = self.homographies[cam_id]
                        # Keep original tracks logic intact for SCT
                        # We just keep a copy of the list of tracks for drawing
                        frame_data[cam_id] = (frame.copy(), list(live_tracks))
                        for t in live_tracks:
                            if t.person_id is None:
                                continue
                            bev_pt = project_to_bev(t.bbox, H)
                            if t.person_id not in traj_buffer[cam_id]:
                                traj_buffer[cam_id][t.person_id] = {}
                            traj_buffer[cam_id][t.person_id][frame_count] = bev_pt

                if not any_alive:
                    break
                    
                frame_count += 1
                window_frames.append(frame_data)
                
                # Run matching every window_size frames
                if frame_count % self.window_size == 0:
                    start_win = frame_count - self.window_size + 1
                    logger.info(f"--- Matching for window {start_win} to {frame_count} ---")
                    
                    np_traj_buffer = traj_buffer
                            
                    for i in range(len(self.cam_ids)):
                        for j in range(i + 1, len(self.cam_ids)):
                            camA = self.cam_ids[i]
                            camB = self.cam_ids[j]
                            matches = self.match_cameras(np_traj_buffer[camA], np_traj_buffer[camB], dist_thresh=10.0)
                            if matches:
                                logger.info(f"  [Cam{camA} <-> Cam{camB}] Matches: {[(m[0], m[1]) for m in matches]}")
                                for (tidA, tidB, dist) in matches:
                                    self.merge_global_ids(camA, tidA, camB, tidB)
                                
                    # Flush window_frames to video
                    for idx, fd in enumerate(window_frames):
                        current_f_id = start_win + idx
                        annotated_frames = []
                        valid_f = next((v[0] for v in fd.values()), None)
                        if valid_f is None:
                            continue
                            
                        for cid in self.cam_ids:
                            if cid in fd:
                                f, tracks = fd[cid]
                                # Create pseudo tracks with global IDs
                                tracks_copy = [TrackCopy(t, self.get_or_create_global_id(cid, t.person_id)) 
                                               for t in tracks if t.person_id is not None]
                                f_info = {"frame_id": current_f_id}
                                ann_f = draw_tracks(f, tracks_copy, f_info)
                                annotated_frames.append(ann_f)
                            else:
                                annotated_frames.append(np.zeros_like(valid_f))
                                
                        grid = draw_grid(annotated_frames, self.cam_names)
                        if writer is None:
                            h, w = grid.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(output_path, fourcc, 25, (w, h))
                        writer.write(grid)
                        
                    # Clear buffer for next window
                    traj_buffer = {cid: {} for cid in self.cam_ids}
                    window_frames = []
                    
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {frame_count} frames | FPS: {fps:.1f}")
                    
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        finally:
            # flush remaining window frames if any
            if window_frames:
                start_win = frame_count - len(window_frames) + 1
                for idx, fd in enumerate(window_frames):
                    current_f_id = start_win + idx
                    annotated_frames = []
                    valid_f = next((v[0] for v in fd.values()), None)
                    if valid_f is None:
                        continue
                        
                    for cid in self.cam_ids:
                        if cid in fd:
                            f, tracks = fd[cid]
                            tracks_copy = [TrackCopy(t, self.get_or_create_global_id(cid, t.person_id)) 
                                           for t in tracks if t.person_id is not None]
                            f_info = {"frame_id": current_f_id}
                            ann_f = draw_tracks(f, tracks_copy, f_info)
                            annotated_frames.append(ann_f)
                        else:
                            annotated_frames.append(np.zeros_like(valid_f))
                            
                    grid = draw_grid(annotated_frames, self.cam_names)
                    if writer is None:
                        h, w = grid.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_path, fourcc, 25, (w, h))
                    writer.write(grid)

            for worker in self.workers.values():
                worker.release()
            if writer is not None:
                writer.release()
            logger.info(f"Pipeline finished processing {frame_count} frames. Video saved to {output_path}")
