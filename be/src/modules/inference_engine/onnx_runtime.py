"""Shared ONNX Runtime inference engine.

Wraps onnxruntime.InferenceSession with automatic provider selection
(CUDA if available, else CPU) and exposes input/output metadata.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import onnxruntime

logger = logging.getLogger(__name__)


def _resolve_providers(device_id: str | int | None) -> list:
    """Return the best available ONNX execution providers for *device_id*.

    Supported formats for *device_id*:
      - ``"cuda"`` or ``"cuda:0"``  → CUDAExecutionProvider (if available)
      - ``"cpu"`` or ``None``        → CPUExecutionProvider
    """
    requested = str(device_id or "cpu").strip().lower()
    available = set(onnxruntime.get_available_providers())

    if requested.startswith("cuda") and "CUDAExecutionProvider" in available:
        idx = requested.split(":")[-1]
        if idx == "cuda":
            idx = "0"
        providers = [("CUDAExecutionProvider", {"device_id": idx}), "CPUExecutionProvider"]
        logger.debug("OnnxRuntime: using CUDAExecutionProvider (device %s)", idx)
        return providers

    logger.debug("OnnxRuntime: using CPUExecutionProvider")
    return ["CPUExecutionProvider"]


class OnnxNetwork:
    """Thin wrapper around an ONNX InferenceSession.

    Attributes:
        input_names  (List[str]): Names of the model input nodes.
        output_names (List[str]): Names of the model output nodes.
        input_shape  (tuple): Shape of the first input tensor (e.g. [1, 3, H, W]).
        input_height (int): Spatial height expected by the model.
        input_width  (int): Spatial width expected by the model.
        output_shape (tuple): Shape of the first output tensor.
    """

    def __init__(self, model_path: str, device_id: str | int | None = "cpu"):
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=_resolve_providers(device_id),
        )
        self._init_input_details()
        self._init_output_details()
        logger.info(
            "OnnxNetwork loaded: %s | input %s | output %s",
            model_path,
            self.input_shape,
            self.output_shape,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names: List[str] = [inp.name for inp in model_inputs]
        self.input_shape: tuple = tuple(model_inputs[0].shape)
        # Shape is expected to be [N, C, H, W]
        self.input_height: int = self.input_shape[2]
        self.input_width: int = self.input_shape[3]

    def _init_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names: List[str] = [out.name for out in model_outputs]
        self.output_shape: tuple = tuple(model_outputs[0].shape)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Run a forward pass.

        Args:
            input_tensor: Float32 array of shape ``[N, C, H, W]``.

        Returns:
            List of output arrays, one per output node.
        """
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})
