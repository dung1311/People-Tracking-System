import os
import yaml
import time
import cv2
import logging
import threading
import base64
from datetime import datetime
from typing import Dict, List, Any, Callable
from sqlmodel import Session, select

from database.session import engine
from models.tracking_session import TrackingSession
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

        logger.info("Streamable MCT Pipeline v2 starting – %d cameras", len(self.workers))

        try:
            while not self.should_stop:
                alive = False
                for w in self.workers.values():
                    if not w.stopped and w.process_next_frame():
                        alive = True
                if not alive:
                    break
                frame_count += 1

                per_cam, frames = self._collect(frame_count)
                mapping = self._match(per_cam, frame_count)

                self._write_mot(mot_files, per_cam, mapping, frame_count)
                self._write_video(frames, per_cam, mapping, frame_count, t0)

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
            
        # Compress grid frame to JPEG and encode to base64
        ret, buffer = cv2.imencode('.jpg', image)
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
            session = db.get(TrackingSession, session_id)
            if not session:
                logger.error(f"Session {session_id} not found in DB")
                return
            session.status = "running"
            session.started_at = datetime.utcnow()
            db.add(session)
            db.commit()
            db.refresh(session)
            
            sct_config = session.sct_config
            mct_config = session.mct_config
            camera_ids = session.camera_ids

            # Get active cameras
            cameras = db.exec(select(Camera).where(Camera.id.in_(camera_ids))).all()

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
                "calibration": local_calib_path
            }

        # 3. Create full mct config dictionary
        full_mct_cfg = {
            "MATCHING": mct_config.get("MATCHING", {
                "thresholds": {"spatial_thresh": 50.0, "reid_thresh": 0.45, "time_thresh": 30}
            }),
            "GLOBAL_TRACK": mct_config.get("GLOBAL_TRACK", {
                "max_lost_frames": 100,
                "confirm_frames": 3
            }),
            "OUTPUT": {
                "video": f"{temp_dir}/mct_output.mp4",
                "txt_dir": f"{temp_dir}/txt",
                "fps": 25,
                "draw_local": True
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
                minio.upload_file(
                    bucket_name="recordings",
                    object_name=out_video_object,
                    file_data=f"{temp_dir}/output.mp4",
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
            session = db.get(TrackingSession, session_id)
            if session:
                session.status = status_str
                session.stopped_at = datetime.utcnow()
                session.total_frames = frames
                session.total_global_ids = globals_count
                session.avg_fps = avg_fps
                if status_str == "completed":
                    session.output_video_path = f"recordings/{session_id}/output.mp4"
                    session.output_txt_dir = f"recordings/{session_id}/txt"
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
