"""MCT Pipeline v2 – orchestrator.

Delegates to:
  - ``CameraWorker``                per-camera SCT
  - ``CrossCameraClusterer``        cost matrix + Union-Find clustering
  - ``GlobalTrackManagerV2``        ReID, global ID allocation, aging
  - ``mct_config.yaml``             external configuration

Flow per frame:
  1. Each camera runs SCT independently.
  2. Tracks with valid features are flattened.
  3. CrossCameraClusterer builds an (N×N) cost matrix and returns clusters.
  4. GlobalTrackManagerV2 assigns global IDs via ReID + allocation.
  5. MOT15 txt per camera + annotated grid video.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

from modules.data_templates.mct_template import CameraCalibration
from modules.data_templates.sct_template import TrackInfo
from modules.matching.cross_camera_clustering import CrossCameraClusterer
from modules.track_manager.global_track_manager_v2 import GlobalTrackManagerV2
from pipelines.camera_worker import CameraWorker
from utils.vis import draw_grid

logger = logging.getLogger(__name__)


class MCTPipeline2:
    """MCT orchestrator: wires SCT workers, clusterer, and global manager."""

    def __init__(self, sct_config: dict, mct_config_path: str):
        with open(mct_config_path) as f:
            mct_cfg = yaml.safe_load(f)

        self.clusterer = CrossCameraClusterer(mct_cfg["MATCHING"])
        self.global_manager = GlobalTrackManagerV2(
            {**mct_cfg["MATCHING"]["thresholds"], **mct_cfg["GLOBAL_TRACK"]},
        )

        cameras = mct_cfg["CAMERAS"]
        self.H_invs: Dict[int, np.ndarray] = {}
        self.workers: Dict[int, CameraWorker] = {}
        for cam_id_str, cam_cfg in cameras.items():
            cid = int(cam_id_str)
            cal = CameraCalibration.load_from_json(cam_cfg["calibration"], cid)
            self.H_invs[cid] = cal.H_inv
            self.workers[cid] = CameraWorker(cid, cam_cfg["video"], sct_config)

        self.cam_ids_sorted = sorted(self.H_invs)
        self.cam_names = [f"Cam {c}" for c in self.cam_ids_sorted]

        out_cfg = mct_cfg.get("OUTPUT", {})
        self._output_video = out_cfg.get("video", "outputs/mct_output.mp4")
        self._output_txt = out_cfg.get("txt_dir", "outputs/txt")
        self._output_fps = out_cfg.get("fps", 25)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        output_path: str | None = None,
        txt_dir: str | None = None,
    ):
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

        logger.info("MCT Pipeline v2 starting – %d cameras", len(self.workers))

        try:
            while True:
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

                if frame_count % 100 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "Frame %d | %d globals | %.1f FPS",
                        frame_count,
                        self.global_manager.num_globals,
                        frame_count / elapsed,
                    )

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted")
        finally:
            for w in self.workers.values():
                w.release()
            if self._writer:
                self._writer.release()
            for fh in mot_files.values():
                fh.close()
            logger.info("Done: %d frames. MOT15 → %s/", frame_count, txt_dir)

    # ------------------------------------------------------------------
    # Per-frame helpers
    # ------------------------------------------------------------------

    def _collect(self, frame_count: int):
        per_cam: Dict[int, List[TrackInfo]] = {}
        frames: Dict[int, np.ndarray] = {}
        for cid, w in self.workers.items():
            if w.latest_frame is not None:
                frames[cid] = w.latest_frame
                per_cam[cid] = w.latest_tracks
        return per_cam, frames

    def _match(
        self,
        per_cam: Dict[int, List[TrackInfo]],
        frame_count: int,
    ) -> Dict[Tuple[int, int], int]:
        all_tracks, all_cids, all_pids = [], [], []
        for cid in self.cam_ids_sorted:
            for t in per_cam.get(cid, []):
                if t.person_id is None or t.get_representative_feature() is None:
                    continue
                all_tracks.append(t)
                all_cids.append(cid)
                all_pids.append(t.person_id)

        if not all_tracks:
            self.global_manager.age_all()
            return {}

        cam_arr = np.array(all_cids, dtype=np.int32)
        pid_arr = np.array(all_pids, dtype=np.int32)
        feet = np.array(
            [[(t.bbox[0] + t.bbox[2]) * 0.5, t.bbox[3]] for t in all_tracks],
            dtype=np.float64,
        )
        feats = CrossCameraClusterer.l2_normalise(
            np.array(
                [t.get_representative_feature() for t in all_tracks],
                dtype=np.float64,
            )
        )

        clusters = self.clusterer.cluster(cam_arr, feet, feats, self.H_invs)
        return self.global_manager.assign(
            clusters, feats, cam_arr, pid_arr, frame_count,
        )

    def _write_mot(self, mot_files, per_cam, mapping, frame_count):
        for cid, tracks in per_cam.items():
            fh = mot_files.get(cid)
            if fh is None:
                continue
            for t in tracks:
                if t.person_id is None:
                    continue
                gid = mapping.get((cid, t.person_id))
                if gid is None:
                    continue
                x1, y1, x2, y2 = t.bbox
                fh.write(
                    f"{frame_count},{gid},{x1:.1f},{y1:.1f},"
                    f"{x2 - x1:.1f},{y2 - y1:.1f},1,-1,-1,-1\n"
                )

    def _write_video(self, frames, per_cam, mapping, frame_count, t0):
        ann = self._annotate_frames(frames, per_cam, mapping, frame_count)
        if not ann:
            return

        grid = draw_grid(ann, self.cam_names)
        elapsed = time.time() - t0
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(
            grid,
            f"Frame: {frame_count} | FPS: {fps:.1f} | "
            f"Globals: {self.global_manager.num_globals}",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )

        if self._writer is None:
            h, w = grid.shape[:2]
            self._writer = cv2.VideoWriter(
                self._video_out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                self._output_fps, (w, h),
            )
        self._writer.write(grid)

    def _annotate_frames(
        self,
        frames: Dict[int, np.ndarray],
        per_cam: Dict[int, List[TrackInfo]],
        mapping: Dict[Tuple[int, int], int],
        frame_count: int,
    ) -> List[np.ndarray]:
        np.random.seed(42)
        palette = np.random.randint(0, 255, size=(500, 3), dtype=np.uint8)
        result = []
        for cid in self.cam_ids_sorted:
            if cid not in frames:
                continue
            frame = frames[cid].copy()
            for t in per_cam.get(cid, []):
                if t.person_id is None:
                    continue
                gid = mapping.get((cid, t.person_id))
                if gid is not None:
                    label = f"G{gid}"
                    color = tuple(int(c) for c in palette[gid % len(palette)])
                else:
                    label = f"L{t.person_id}"
                    color = (128, 128, 128)

                x1, y1, x2, y2 = map(int, t.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2,
                )
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(
                    frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                )

            cv2.putText(
                frame, f"Cam {cid} | F{frame_count}",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
            result.append(frame)
        return result
