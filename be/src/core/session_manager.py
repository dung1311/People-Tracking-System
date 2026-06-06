import os
import json
import yaml
import time
import cv2
import logging
import threading
import base64
from datetime import datetime
from typing import Dict, List, Any, Callable
from sqlmodel import Session, select, delete
from models.track import Track

from database.session import engine
from models.camera_network import CameraNetwork
from models.camera import Camera
from core.minio_client import get_minio_client
from pipelines.mct_pipeline_2 import MCTPipeline2

logger = logging.getLogger(__name__)

class VideoWriterInterceptor:
    def __init__(self, real_writer, pipeline, callback):
        self.real_writer = real_writer
        self.pipeline = pipeline
        self.callback = callback

    def write(self, image):
        if self.real_writer:
            self.real_writer.write(image)
        if self.callback:
            try:
                # Capture frame statistics from pipeline state
                frame_id = getattr(self.pipeline, "_frame_count_captured", 0)
                active_globals = self.pipeline.global_manager.num_globals
                fps = getattr(self.pipeline, "_fps_captured", 25.0)
                self.callback(image, frame_id, active_globals, fps)
            except Exception as e:
                logger.error(f"Error in frame callback: {e}")

    def release(self):
        if self.real_writer:
            self.real_writer.release()


class StreamableMCTPipeline2(MCTPipeline2):
    def __init__(self, sct_config: dict, mct_config_path: str, session_id: int, callback: Callable = None):
        self.session_id = session_id
        self.callback = callback
        self._frame_count_captured = 0
        self._fps_captured = 25.0
        self.should_stop = False
        self._pending_db_tracks = []
        super().__init__(sct_config, mct_config_path)

    def _write_video(self, frames, per_cam, mapping, frame_count, t0):
        self._frame_count_captured = frame_count
        elapsed = time.time() - t0
        self._fps_captured = frame_count / elapsed if elapsed > 0 else 0
        
        # Intercept before cv2.VideoWriter is created/called
        writer_is_none = (self._writer is None)
        super()._write_video(frames, per_cam, mapping, frame_count, t0)
        
        if writer_is_none and self._writer is not None:
            self._writer = VideoWriterInterceptor(self._writer, self, self.callback)

    def run(self, output_path: str | None = None, txt_dir: str | None = None):
        # Override run loop to allow graceful interruption/stop
        output_path = output_path or self._output_video
        txt_dir = txt_dir or self._output_txt

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)

        mot_files = {
            cid: open(os.path.join(txt_dir, f"cam{cid}_mct.txt"), "w")
            for cid in self.workers
        }
        self._writer = None
        self._video_out_path = output_path
        frame_count = 0
        t0 = time.time()

        import concurrent.futures

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.workers))
        try:
            while not self.should_stop:
                alive = False
                futures = {
                    executor.submit(w.process_next_frame): cid
                    for cid, w in self.workers.items() if not w.stopped
                }
                
                results = {}
                for fut in concurrent.futures.as_completed(futures):
                    cid = futures[fut]
                    try:
                        results[cid] = fut.result()
                    except Exception as e:
                        logger.error(f"Error in camera worker {cid}: {e}")
                        results[cid] = False
                        
                for cid, res in results.items():
                    if res:
                        alive = True
                        
                if not alive:
                    break
                frame_count += 1

                per_cam, frames = self._collect(frame_count)
                mapping = self._match(per_cam, frame_count)

                self._write_mot(mot_files, per_cam, mapping, frame_count)
                self._write_video(frames, per_cam, mapping, frame_count, t0)

                # Save global tracks to database in real-time
                try:
                    for cid, tracks in per_cam.items():
                        for t in tracks:
                            if t.person_id is None:
                                continue
                            gid = mapping.get((cid, t.person_id))
                            if gid is None:
                                continue
                            
                            feat = t.features[-1].tolist() if t.features else None
                            db_track = Track(
                                camera_id=cid,
                                person_id=gid,  # Store the global ID as person_id
                                frame_id=frame_count,
                                bbox=t.bbox,
                                score=t.score,
                                class_id=t.class_id,
                                timestamp=t.timestamp,
                                feature=feat
                            )
                            self._pending_db_tracks.append(db_track)

                    # Batch commit every 10 frames or if pipeline is stopping
                    if frame_count % 10 == 0 or self.should_stop:
                        if self._pending_db_tracks:
                            with Session(engine) as db_sess:
                                for db_track in self._pending_db_tracks:
                                    db_sess.add(db_track)
                                db_sess.commit()
                            self._pending_db_tracks.clear()
                except Exception as e:
                    logger.error(f"Error saving global tracks to database: {e}")

                if frame_count % 10 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "Session %d | Frame %d | %d globals | %.1f FPS",
                        self.session_id,
                        frame_count,
                        self.global_manager.num_globals,
                        frame_count / elapsed,
                    )
        except Exception as e:
            logger.error(f"Pipeline running error in Session {self.session_id}: {e}")
            raise e
        finally:
            executor.shutdown(wait=True)
            # Flush any remaining pending DB tracks on exit
            try:
                if hasattr(self, "_pending_db_tracks") and self._pending_db_tracks:
                    logger.info(f"Flushing {len(self._pending_db_tracks)} remaining tracks to database...")
                    with Session(engine) as db_sess:
                        for db_track in self._pending_db_tracks:
                            db_sess.add(db_track)
                        db_sess.commit()
                    self._pending_db_tracks.clear()
            except Exception as e:
                logger.error(f"Error flushing pending tracks on exit: {e}")

            for w in self.workers.values():
                w.release()
            if self._writer:
                self._writer.release()
            for fh in mot_files.values():
                fh.close()
            logger.info("Done: %d frames. MOT15 → %s/", frame_count, txt_dir)
            return frame_count, self.global_manager.num_globals, self._fps_captured


class SessionManager:
    def __init__(self):
        self.active_threads: Dict[int, threading.Thread] = {}
        self.active_pipelines: Dict[int, StreamableMCTPipeline2] = {}
        self.ws_callbacks: Dict[int, List[Callable]] = {}
        self.last_broadcast_time: Dict[int, float] = {}

    def register_ws_callback(self, session_id: int, callback: Callable):
        if session_id not in self.ws_callbacks:
            self.ws_callbacks[session_id] = []
        self.ws_callbacks[session_id].append(callback)
        logger.info(f"Registered WebSocket callback for Session {session_id}")

    def deregister_ws_callback(self, session_id: int, callback: Callable):
        if session_id in self.ws_callbacks and callback in self.ws_callbacks[session_id]:
            self.ws_callbacks[session_id].remove(callback)
            logger.info(f"Deregistered WebSocket callback for Session {session_id}")

    def _broadcast_frame(self, session_id: int, image, frame_id: int, active_globals: int, fps: float):
        callbacks = self.ws_callbacks.get(session_id, [])
        if not callbacks:
            return
            
        # Throttle to max 10 FPS to prevent choking the WebSocket and browser thread
        now = time.time()
        last_time = self.last_broadcast_time.get(session_id, 0.0)
        if now - last_time < 0.1:  # Max 10 FPS
            return
        self.last_broadcast_time[session_id] = now
            
        # Resize to reduce CPU, memory, and network overhead for streaming
        h, w = image.shape[:2]
        max_width = 1280
        if w > max_width:
            scale = max_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_to_encode = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            image_to_encode = image

        # Compress grid frame to JPEG and encode to base64
        ret, buffer = cv2.imencode('.jpg', image_to_encode)
        if not ret:
            return
            
        base64_str = base64.b64encode(buffer).decode('utf-8')
        event_data = {
            "type": "frame",
            "data": {
                "frame_id": frame_id,
                "active_globals": active_globals,
                "fps": round(fps, 1),
                "base64_jpg": base64_str
            }
        }
        
        for cb in callbacks:
            try:
                cb(event_data)
            except Exception as e:
                logger.error(f"Error calling WS callback for Session {session_id}: {e}")

    def start_session(self, session_id: int):
        if session_id in self.active_threads:
            raise ValueError(f"Session {session_id} is already running.")

        thread = threading.Thread(target=self._run_session_worker, args=(session_id,))
        self.active_threads[session_id] = thread
        thread.start()
        logger.info(f"Background thread started for Session {session_id}")

    def stop_session(self, session_id: int):
        pipeline = self.active_pipelines.get(session_id)
        if pipeline:
            logger.info(f"Signaling stop to Session {session_id} pipeline...")
            pipeline.should_stop = True
            
        thread = self.active_threads.get(session_id)
        if thread:
            thread.join(timeout=10.0) # Wait up to 10 seconds for graceful exit
            
        self.active_threads.pop(session_id, None)
        self.active_pipelines.pop(session_id, None)
        logger.info(f"Session {session_id} stopped.")

    def _run_session_worker(self, session_id: int):
        minio = get_minio_client()
        
        # 1. Update session status to running in DB
        with Session(engine) as db:
            session = db.get(CameraNetwork, session_id)
            if not session:
                logger.error(f"Session {session_id} not found in DB")
                return
            session.status = "running"
            session.started_at = datetime.now()
            db.add(session)
            db.commit()
            db.refresh(session)
            sct_config = session.sct_config
            if not isinstance(sct_config, dict) or "DETECTION" not in sct_config:
                logger.warning(f"SCT config for Session {session_id} is missing or has old structure. Auto-patching using configs/sct_config.yaml.")
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                sct_yaml_path = os.path.join(base_dir, "configs", "sct_config.yaml")
                loaded_sct = {}
                if os.path.exists(sct_yaml_path):
                    try:
                        with open(sct_yaml_path, "r") as f:
                            loaded_sct = yaml.safe_load(f)
                    except Exception as e:
                        logger.error(f"Failed to read fallback configs/sct_config.yaml: {e}")
                
                if loaded_sct and "DETECTION" in loaded_sct:
                    sct_config = loaded_sct
                else:
                    sct_config = {
                        "DETECTION": {
                            "name": "yolov11",
                            "yolov11": {
                                "model_path": "weights/yolo11x.pt",
                                "task": "detect",
                                "imgsz": 640,
                                "conf_thres": 0.5,
                                "iou_thres": 0.7,
                                "device": "cpu",
                                "classes": 0,
                                "max_det": 300
                            }
                        },
                        "POSE_ESTIMATION": {
                            "name": "rtmpose",
                            "rtmpose": {
                                "device": "cpu",
                                "model_path": "weights/rtmpose-l_256x192/end2end.onnx",
                                "input_size": [192, 256]
                            }
                        },
                        "TRACKING": {
                            "name": "sort",
                            "sort": {
                                "max_age": 30,
                                "min_hits": 3,
                                "iou_threshold": 0.3
                            }
                        },
                        "TRACK_MANAGER": {
                            "is_join_track": True,
                            "smooth_factor": 0.1,
                            "appearance_threshold": 0.5,
                            "GALLERY": {
                                "max_live_time": 24000,
                                "min_hits": 3,
                                "max_features": 50
                            },
                            "MATCHING": {
                                "distance_threshold": 0.5
                            },
                            "EMBEDDING": {
                                "name": "fastreid",
                                "fastreid": "configs/osnet-ain_x1.0-ibn_512_256x192_ccdmmps.yaml"
                            }
                        }
                    }
                session.sct_config = sct_config
                db.add(session)
                db.commit()
                db.refresh(session)
            
            # Dynamic device patching based on actual PyTorch CUDA availability
            device_str = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    device_str = "cuda"
            except Exception:
                pass

            sct_updated = False
            if isinstance(sct_config, dict):
                if "DETECTION" in sct_config and "yolov11" in sct_config["DETECTION"]:
                    current_device = sct_config["DETECTION"]["yolov11"].get("device")
                    if current_device != device_str:
                        logger.info(f"Dynamically patching yolov11 device from '{current_device}' to '{device_str}' for Session {session_id}")
                        sct_config["DETECTION"]["yolov11"]["device"] = device_str
                        sct_updated = True
                
                if "POSE_ESTIMATION" in sct_config and "rtmpose" in sct_config["POSE_ESTIMATION"]:
                    current_device = sct_config["POSE_ESTIMATION"]["rtmpose"].get("device")
                    if current_device != device_str:
                        logger.info(f"Dynamically patching rtmpose device from '{current_device}' to '{device_str}' for Session {session_id}")
                        sct_config["POSE_ESTIMATION"]["rtmpose"]["device"] = device_str
                        sct_updated = True

            if sct_updated:
                session.sct_config = sct_config
                db.add(session)
                db.commit()
                db.refresh(session)
            
            mct_config = session.mct_config
            camera_ids = [cam.id for cam in session.cameras] if session.cameras else []

            # Get active cameras
            cameras_db = db.exec(select(Camera).where(Camera.id.in_(camera_ids))).all()
            from types import SimpleNamespace
            cameras = [
                SimpleNamespace(
                    id=cam.id,
                    calibration_path=cam.calibration_path,
                    source=cam.source,
                    rois=cam.rois
                )
                for cam in cameras_db
            ]

            # Clear old tracks for these cameras
            if camera_ids:
                db.exec(delete(Track).where(Track.camera_id.in_(camera_ids)))
                db.commit()

        # 2. Prepare paths and config file
        temp_dir = f"data/temp_sessions/{session_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Structure for CAMERAS in mct_config
        cameras_cfg = {}
        
        for cam in cameras:
            # Download calibration JSON to a temp path
            local_calib_path = f"{temp_dir}/calib_cam{cam.id}.json"
            if cam.calibration_path:
                try:
                    minio.download_file("calibrations", cam.calibration_path, local_calib_path)
                except Exception as e:
                    logger.error(f"Error downloading calibration for Cam {cam.id}: {e}")
                    # Create empty/mock calibration as fallback
                    with open(local_calib_path, "w") as f:
                        json.dump({"H_inv": [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]}, f)
            else:
                # Standard fallback calibration if none is provided
                with open(local_calib_path, "w") as f:
                    json.dump({"H_inv": [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]}, f)
                    
            # Resolve video path
            local_video_path = cam.source
            if local_video_path.startswith("videos/"):
                if not minio.fallback_mode:
                    # Download video cache
                    cached_video = f"{temp_dir}/cam{cam.id}_video.mp4"
                    logger.info(f"Downloading stream source for Session {session_id} Cam {cam.id}...")
                    object_name = local_video_path.replace("videos/", "", 1)
                    minio.download_file("videos", object_name, cached_video)
                    local_video_path = cached_video
                else:
                    local_video_path = f"data/{local_video_path}"
            elif not local_video_path.startswith("rtsp") and not os.path.exists(local_video_path):
                # Fallback project root video
                if os.path.exists("mct_demo.mp4"):
                    local_video_path = "mct_demo.mp4"
            
            cameras_cfg[str(cam.id)] = {
                "video": local_video_path,
                "calibration": local_calib_path,
                "rois": getattr(cam, "rois", [])
            }

        # Ensure MATCHING config has correct structure for MCTPipeline2 (homography & visual weights)
        db_matching = mct_config.get("MATCHING", {})
        if not isinstance(db_matching, dict) or "weights" not in db_matching or "homography" not in db_matching.get("weights", {}):
            logger.warning(f"MCT config MATCHING section for Session {session_id} is missing or has old structure. Auto-patching to modern weights/thresholds.")
            matching_cfg = {
                "weights": {
                    "homography": 0.5,
                    "visual": 0.5
                },
                "thresholds": {
                    "homography": 10.0,
                    "visual_gate": 0.5,
                    "homo_gate": 10.0,
                    "combined": 0.6,
                    "reid": 0.4
                }
            }
        else:
            matching_cfg = db_matching

        # Ensure GLOBAL_TRACK has correct keys for MCTPipeline2
        db_global_track = mct_config.get("GLOBAL_TRACK", {})
        if not isinstance(db_global_track, dict) or "max_lost_age" not in db_global_track:
            global_track_cfg = {
                "max_lost_age": 300,
                "feature_smooth": 0.1
            }
        else:
            global_track_cfg = db_global_track

        # 3. Create full mct config dictionary
        full_mct_cfg = {
            "MATCHING": matching_cfg,
            "GLOBAL_TRACK": global_track_cfg,
            "OUTPUT": {
                "video": f"{temp_dir}/mct_output.mp4",
                "txt_dir": f"{temp_dir}/txt",
                "fps": mct_config.get("OUTPUT", {}).get("fps", 25),
                "draw_local": mct_config.get("OUTPUT", {}).get("draw_local", False)
            },
            "CAMERAS": cameras_cfg
        }
        
        # Save config dictionary to YAML file
        yaml_config_path = f"{temp_dir}/mct_config.yaml"
        with open(yaml_config_path, "w") as f:
            yaml.dump(full_mct_cfg, f)

        # 4. Instantiate pipeline
        def frame_callback(image, frame_id, active_globals, fps):
            self._broadcast_frame(session_id, image, frame_id, active_globals, fps)

        pipeline = StreamableMCTPipeline2(
            sct_config=sct_config,
            mct_config_path=yaml_config_path,
            session_id=session_id,
            callback=frame_callback
        )
        self.active_pipelines[session_id] = pipeline

        # 5. Run the pipeline
        status_str = "completed"
        try:
            frames, globals_count, avg_fps = pipeline.run(
                output_path=f"{temp_dir}/output.mp4",
                txt_dir=f"{temp_dir}/txt"
            )
            
            # Upload final video and txt files to MinIO/local storage
            logger.info(f"Session {session_id} finished processing. Uploading outputs...")
            
            out_video_object = f"{session_id}/output.mp4"
            if os.path.exists(f"{temp_dir}/output.mp4"):
                # Re-encode output.mp4 to highly compatible H.264 (libx264) for browser/player native playback
                raw_video_path = f"{temp_dir}/output_raw.mp4"
                h264_video_path = f"{temp_dir}/output.mp4"
                try:
                    import subprocess
                    logger.info("Converting tracking output video to highly compatible H264 format using ffmpeg...")
                    os.rename(h264_video_path, raw_video_path)
                    
                    import sys
                    ffmpeg_executable = "ffmpeg"
                    bin_dir = os.path.dirname(sys.executable)
                    env_ffmpeg = os.path.join(bin_dir, "ffmpeg")
                    if os.path.exists(env_ffmpeg):
                        ffmpeg_executable = env_ffmpeg
                        logger.info(f"Using environment ffmpeg binary at: {ffmpeg_executable}")
                    
                    cmd = [
                        ffmpeg_executable, "-y",
                        "-i", raw_video_path,
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-preset", "fast",
                        h264_video_path
                    ]
                    # Run re-encoding with 600s timeout
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600.0)
                    if result.returncode == 0:
                        logger.info("ffmpeg conversion to H.264 (yuv420p) completed successfully.")
                    else:
                        logger.error(f"ffmpeg conversion failed (code {result.returncode}): {result.stderr}")
                        # Fallback to original
                        if os.path.exists(h264_video_path):
                            os.remove(h264_video_path)
                        if os.path.exists(raw_video_path):
                            os.rename(raw_video_path, h264_video_path)
                except Exception as e:
                    logger.error(f"Failed to run ffmpeg video conversion: {e}")
                    # Fallback to original
                    if os.path.exists(h264_video_path):
                        os.remove(h264_video_path)
                    if os.path.exists(raw_video_path) and not os.path.exists(h264_video_path):
                        os.rename(raw_video_path, h264_video_path)

                minio.upload_file(
                    bucket_name="recordings",
                    object_name=out_video_object,
                    file_data=h264_video_path,
                    content_type="video/mp4"
                )
                
            # Upload MOT txt results to calibrations/snapshots or just recordings bucket
            # Let's save txt under snapshots/session_id/txt/
            txt_files_count = 0
            if os.path.exists(f"{temp_dir}/txt"):
                for txt_file in os.listdir(f"{temp_dir}/txt"):
                    if txt_file.endswith(".txt"):
                        file_path = os.path.join(f"{temp_dir}/txt", txt_file)
                        minio.upload_file(
                            bucket_name="recordings",
                            object_name=f"{session_id}/txt/{txt_file}",
                            file_data=file_path,
                            content_type="text/plain"
                        )
                        txt_files_count += 1
                        
        except Exception as e:
            logger.error(f"Error running pipeline in background worker: {e}")
            status_str = "failed"
            frames, globals_count, avg_fps = 0, 0, 0.0

        # 6. Save final session statistics to DB
        with Session(engine) as db:
            session = db.get(CameraNetwork, session_id)
            if session:
                session.status = status_str
                session.stopped_at = datetime.now()
                session.total_frames = frames
                session.total_global_ids = globals_count
                session.avg_fps = avg_fps
                if status_str == "completed":
                    session.output_video_path = f"recordings/{session_id}/output.mp4"
                    session.output_txt_dir = f"recordings/{session_id}/txt"
                    
                    # Also create VideoSegment for each camera in the network so they can be analyzed
                    from models.video_segment import VideoSegment
                    duration = (session.stopped_at - session.started_at).total_seconds() if (session.stopped_at and session.started_at) else 0.0
                    for cam in session.cameras:
                        segment = VideoSegment(
                            camera_id=cam.id,
                            file_path=session.output_video_path,
                            start_time=session.started_at or session.created_at or datetime.now(),
                            end_time=session.stopped_at,
                            duration_seconds=duration
                        )
                        db.add(segment)
                db.add(session)
                db.commit()
                
        # Cleanup temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")
            
        self.active_threads.pop(session_id, None)
        self.active_pipelines.pop(session_id, None)
        logger.info(f"Background worker for Session {session_id} has terminated with status: {status_str}")
        
        # Broadcast final status
        callbacks = self.ws_callbacks.get(session_id, [])
        for cb in callbacks:
            try:
                cb({"type": "session_status", "data": {"status": status_str}})
            except Exception:
                pass


# Singleton Session Manager instance
session_manager = SessionManager()

def get_session_manager() -> SessionManager:
    return session_manager
