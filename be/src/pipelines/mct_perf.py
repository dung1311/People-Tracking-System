"""Cấu hình và factory model dùng chung cho MCT pipeline 2 / 3."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from modules.detector.factory import DetectorFactory
from modules.embedder.factory import EmbedderFactory
from modules.pose_estimator.factory import PoseEstimatorFactory

logger = logging.getLogger(__name__)


def get_mct_perf_flags(mct_cfg: Dict) -> Tuple[bool, bool]:
    """Đọc ``MCT_PERFORMANCE``; fallback ``MCT_PIPELINE3`` (tương thích cũ)."""
    perf = mct_cfg.get("MCT_PERFORMANCE")
    if perf is None:
        perf = mct_cfg.get("MCT_PIPELINE3", {})
    shared = bool(perf.get("shared_models", True))
    pose = bool(perf.get("enable_pose_full_body", True))
    return shared, pose


def build_optional_shared_models(
    sct_config: Dict,
    *,
    shared_models: bool,
    enable_pose_full_body: bool,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Nếu ``shared_models`` thì tạo một detector + embedder (+ pose nếu bật).

    Returns:
        ``(detector, embedder, pose_or_none)`` — toàn ``None`` nếu không shared.
    """
    if not shared_models:
        logger.info("MCT: shared_models=false (mỗi camera load model riêng)")
        return None, None, None

    logger.info(
        "MCT: shared_models=true (một detector + một ReID%s)",
        " + một pose" if enable_pose_full_body else "",
    )
    detector = DetectorFactory(sct_config["DETECTION"]).get_detector()
    embedder = EmbedderFactory(
        sct_config["TRACK_MANAGER"]["EMBEDDING"],
    ).get_embedder()
    pose = None
    if enable_pose_full_body:
        pose = PoseEstimatorFactory(
            sct_config["POSE_ESTIMATION"],
        ).get_pose_estimator()
    return detector, embedder, pose
