"""ONNX-based Re-ID embedder.

Replaces the PyTorch-heavy fastreid.Embedding with a lightweight
ONNX Runtime backend so any model exported to ``.onnx`` (e.g. FastReID,
OSNet, BOT-ReID, …) can be plugged in via config without code changes.

Expected config keys (under ``EMBEDDING.onnx_reid`` in sct_config.yaml):
    model_path  (str)  : Path to the ``.onnx`` weight file.
    device      (str)  : ``"cuda"`` / ``"cuda:0"`` / ``"cpu"``.
    mean        (list) : Per-channel mean for normalisation, e.g. [0.485, 0.456, 0.406].
    std         (list) : Per-channel std  for normalisation, e.g. [0.229, 0.224, 0.225].
    norm_type   (str)  : ``"imagenet"`` (default) or ``"half"`` (mean=127.5, std=127.5).
                         If ``mean``/``std`` are given explicitly they take precedence.

Input images are expected in **BGR uint8** format (OpenCV convention), the
same as ``fastreid.Embedding``.

Output: ``np.ndarray`` of shape ``[N, D]`` with **L2-normalised** row vectors.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import cv2
import numpy as np

from ..base import BaseEmbedder
from ...inference_engine.onnx_runtime import OnnxNetwork

logger = logging.getLogger(__name__)

# Default ImageNet statistics (matches most person Re-ID models)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OnnxReidEmbedder(BaseEmbedder):
    """Lightweight ONNX Re-ID embedder compatible with ``BaseEmbedder``.

    The preprocessing pipeline mirrors the reference ``feature_extractor.py``:
        BGR  →  RGB  →  resize  →  normalise  →  CHW  →  batch float32
    """

    def __init__(self, config: Dict):
        super().__init__()

        model_path: str = config["model_path"]
        device: str     = config.get("device", "cpu")

        self._network = OnnxNetwork(model_path, device)

        # ---------- normalisation statistics ----------
        if "mean" in config and "std" in config:
            self._mean = np.array(config["mean"], dtype=np.float32)
            self._std  = np.array(config["std"],  dtype=np.float32)
        elif config.get("norm_type", "imagenet") == "half":
            self._mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self._std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        else:
            self._mean = _IMAGENET_MEAN
            self._std  = _IMAGENET_STD

        # If mean values are in 0-255 range (e.g. 127.5) the pixel must NOT
        # be divided by 255 first.  Otherwise (0-1 range) divide first.
        self._pixel_scale: float = 1.0 if float(self._mean.max()) > 1.0 else 1.0 / 255.0

        # ---------- input spatial size ----------
        # Config takes priority; fall back to what the ONNX model declares.
        if "input_size" in config:
            # Expected format: [H, W]  (height first, matching OpenCV convention)
            h, w = config["input_size"]
            self._input_h: int = int(h)
            self._input_w: int = int(w)
        else:
            self._input_h = self._network.input_height
            self._input_w = self._network.input_width

        self._feat_dim: int = self._network.output_shape[1]

        logger.info(
            "OnnxReidEmbedder ready | model=%s | device=%s | input=(%d×%d) | feat_dim=%d",
            model_path, device, self._input_h, self._input_w, self._feat_dim,
        )

    # ------------------------------------------------------------------
    # BaseEmbedder interface
    # ------------------------------------------------------------------

    def _preprocess(self, imgs_bgr: List[np.ndarray]) -> np.ndarray:
        """Convert a list of BGR crops to a batched float32 tensor [N, 3, H, W].

        Steps (identical to the reference feature_extractor.py):
          1. BGR → RGB
          2. Resize to model input size
          3. Normalise: (pixel / 255 - mean) / std
          4. HWC → CHW
        """
        batch: List[np.ndarray] = []
        for img in imgs_bgr:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (self._input_w, self._input_h),
                                 interpolation=cv2.INTER_LINEAR)
            img_f = img_rgb.astype(np.float32) * self._pixel_scale   # 0-1 or 0-255
            img_f = (img_f - self._mean) / self._std                 # shape: [H, W, 3]
            img_f = img_f.transpose(2, 0, 1)                         # shape: [3, H, W]
            batch.append(img_f)

        return np.stack(batch, axis=0).astype(np.float32)  # [N, 3, H, W]

    def extract_feature(self, imgs_bgr: List[np.ndarray]) -> np.ndarray:
        """Extract L2-normalised feature vectors.

        Args:
            imgs_bgr: List of BGR crop images (any size; resized internally).

        Returns:
            np.ndarray of shape ``[N, D]``, L2-normalised.
        """
        if not imgs_bgr:
            return np.empty((0, self._feat_dim), dtype=np.float32)

        if not isinstance(imgs_bgr, list):
            imgs_bgr = [imgs_bgr]

        batch = self._preprocess(imgs_bgr)               # [N, 3, H, W]
        feats: np.ndarray = self._network.inference(batch)[0]  # [N, D]

        # L2 normalise each row (equivalent to F.normalize in fastreid embed.py)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        feats = feats / norms

        return feats.astype(np.float32)
