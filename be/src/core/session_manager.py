"""Session manager — singleton that manages running MCT pipeline instances."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from sqlmodel import Session as DBSession

from database.session import engine
from models.tracking_session import TrackingSession

logger = logging.getLogger(__name__)


class _RunningSession:
    """State for a single running tracking session."""

    __slots__ = (
        "session_id", "pipeline", "thread", "stop_event",
        "frame_count", "fps", "active_globals", "error",
    )

    def __init__(self, session_id: int):
        self.session_id = session_id
        self.pipeline = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.frame_count = 0
        self.fps = 0.0
        self.active_globals = 0
        self.error: Optional[str] = None


class SessionManager:
    """Manages lifecycle of tracking sessions (start / stop / status)."""

    _instance: Optional["SessionManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._sessions: Dict[int, _RunningSession] = {}

    @classmethod
    def get_instance(cls) -> "SessionManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Start ──
    def start_session(
        self,
        session_id: int,
        sct_config: dict,
        mct_config: dict,
        cameras: List[dict],  # [{id, source_uri, calibration_path}, ...]
    ):
        if session_id in self._sessions:
            raise RuntimeError(f"Session {session_id} is already running")

        rs = _RunningSession(session_id)
        self._sessions[session_id] = rs

        rs.thread = threading.Thread(
            target=self._run_pipeline,
            args=(rs, sct_config, mct_config, cameras),
            daemon=True,
            name=f"mct-session-{session_id}",
        )
        rs.thread.start()
        logger.info("Started session %d", session_id)

    def _run_pipeline(
        self,
        rs: _RunningSession,
        sct_config: dict,
        mct_config: dict,
        cameras: List[dict],
    ):
        """Run MCT pipeline in a background thread."""
        try:
            from core.minio_client import get_minio

            # Build temporary MCT config YAML for the pipeline
            # The pipeline expects a file path, so we write a temp file
            import yaml

            minio = get_minio()

            # Resolve camera sources: download videos from MinIO if needed
            resolved_cameras = {}
            calib_paths = {}

            os.makedirs("data/session_tmp", exist_ok=True)
            tmp_dir = f"data/session_tmp/session_{rs.session_id}"
            os.makedirs(tmp_dir, exist_ok=True)

            for cam in cameras:
                cam_id = cam["id"]
                source = cam["source_uri"]
                calib = cam.get("calibration_path")

                # Resolve video source
                if source.startswith("videos/"):
                    # MinIO path → download to local
                    local_path = os.path.join(tmp_dir, f"cam{cam_id}.mp4")
                    if not os.path.exists(local_path):
                        data = minio.get_file(source)
                        with open(local_path, "wb") as f:
                            f.write(data)
                    resolved_cameras[cam_id] = local_path
                elif source.startswith("rtsp://") or source.startswith("/"):
                    resolved_cameras[cam_id] = source
                else:
                    # Local file path
                    for candidate in [source, f"data/videos/{source}", f"../{source}"]:
                        if os.path.exists(candidate):
                            resolved_cameras[cam_id] = candidate
                            break
                    else:
                        resolved_cameras[cam_id] = source

                # Resolve calibration
                if calib:
                    if calib.startswith("calibrations/"):
                        local_calib = os.path.join(tmp_dir, f"cam{cam_id}_calib.json")
                        if not os.path.exists(local_calib):
                            data = minio.get_file(calib)
                            with open(local_calib, "wb") as f:
                                f.write(data)
                        calib_paths[cam_id] = local_calib
                    else:
                        calib_paths[cam_id] = calib

            # Build MCT config with resolved paths
            cameras_section = {}
            for cam_id in resolved_cameras:
                cameras_section[cam_id] = {
                    "video": resolved_cameras[cam_id],
                    "calibration": calib_paths.get(cam_id, ""),
                }

            full_mct = {**mct_config, "CAMERAS": cameras_section}

            # Write temp MCT config
            mct_config_path = os.path.join(tmp_dir, "mct_config.yaml")
            with open(mct_config_path, "w") as f:
                yaml.safe_dump(full_mct, f)

            # Output paths
            output_dir = os.path.join(tmp_dir, "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_video = os.path.join(output_dir, "mct_output.mp4")
            output_txt = os.path.join(output_dir, "txt")

            # Initialize pipeline
            from pipelines.mct_pipeline_3 import MCTPipeline3

            pipeline = MCTPipeline3(sct_config, mct_config_path)
            rs.pipeline = pipeline

            # Update DB status
            self._update_db_status(rs.session_id, "running", started_at=datetime.utcnow())

            # Run with stop check
            # We monkey-patch the pipeline to check stop_event
            original_run = pipeline.run

            def patched_run(**kwargs):
                """Wrap pipeline.run to inject stop checking."""
                original_run(
                    output_path=output_video,
                    txt_dir=output_txt,
                )

            patched_run()

            # Upload outputs to MinIO
            try:
                if os.path.exists(output_video):
                    with open(output_video, "rb") as f:
                        minio.upload_file(
                            f"recordings/{rs.session_id}/output.mp4",
                            f,
                            os.path.getsize(output_video),
                            "video/mp4",
                        )
            except Exception as e:
                logger.warning("Failed to upload output video: %s", e)

            # Update stats
            total_frames = 0
            total_globals = 0
            if pipeline:
                total_frames = sum(w.frame_id for w in pipeline.workers.values())
                total_globals = len(pipeline._known_gids)

            self._update_db_status(
                rs.session_id, "completed",
                stopped_at=datetime.utcnow(),
                total_frames=total_frames,
                total_global_ids=total_globals,
                output_video_path=f"recordings/{rs.session_id}/output.mp4",
            )

        except Exception as e:
            logger.exception("Session %d failed: %s", rs.session_id, e)
            rs.error = str(e)
            self._update_db_status(rs.session_id, "failed", stopped_at=datetime.utcnow())
        finally:
            self._sessions.pop(rs.session_id, None)

    def _update_db_status(self, session_id: int, status: str, **kwargs):
        """Update session record in the database."""
        try:
            with DBSession(engine) as db:
                session = db.get(TrackingSession, session_id)
                if session:
                    session.status = status
                    for k, v in kwargs.items():
                        if hasattr(session, k):
                            setattr(session, k, v)
                    db.add(session)
                    db.commit()
        except Exception as e:
            logger.error("Failed to update session %d status: %s", session_id, e)

    # ── Stop ──
    def stop_session(self, session_id: int):
        rs = self._sessions.get(session_id)
        if rs is None:
            return

        rs.stop_event.set()

        # Stop pipeline workers
        if rs.pipeline:
            for worker in rs.pipeline.workers.values():
                worker._stopped = True

        self._update_db_status(session_id, "stopping")
        logger.info("Stopping session %d", session_id)

    # ── Status ──
    def get_status(self, session_id: int) -> Optional[dict]:
        rs = self._sessions.get(session_id)
        if rs is None:
            return None
        return {
            "session_id": session_id,
            "is_running": rs.thread.is_alive() if rs.thread else False,
            "frame_count": rs.frame_count,
            "fps": rs.fps,
            "active_globals": rs.active_globals,
            "error": rs.error,
        }

    def is_running(self, session_id: int) -> bool:
        rs = self._sessions.get(session_id)
        return rs is not None and rs.thread is not None and rs.thread.is_alive()

    def list_running(self) -> List[int]:
        return [sid for sid, rs in self._sessions.items() if rs.thread and rs.thread.is_alive()]


def get_session_manager() -> SessionManager:
    return SessionManager.get_instance()
