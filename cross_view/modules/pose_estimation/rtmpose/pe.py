from typing import Dict, List
import numpy as np

from rtmlib import RTMPose
from ..interface import IPoseEsimation

class RTMPoseEstimator(IPoseEsimation):
    def __init__(self, config: Dict):
        self.device = config["device"]
        self.weight = config["path"]
        self.input_size = tuple(config["input_size"][:2])  # Ensure it's (w, h)
        self.model = RTMPose(
            onnx_model=self.weight,
            model_input_size=self.input_size,
            device=self.device
        )
    
    def _preprocess(self, img_bgr):
        return img_bgr

    def estimate(self, image: np.ndarray, boxes: List[List[float]]):
        img = self._preprocess(image)
        predicts = self.predict(img, boxes)
        return self._postprocess(predicts)

    def predict(self, image: np.ndarray, boxes: List[List[float]]):
        if len(boxes) == 0:
            return []
    
        keypoints, scores = self.model(image, boxes)
        assert len(boxes) == len(keypoints), f"#bboxes: {len(boxes)}, #poses: {len(keypoints)}"

        scores_expand = scores[..., np.newaxis]
        keypoints_with_scores = np.concatenate([keypoints, scores_expand], axis=-1)
        
        return keypoints_with_scores

    def _postprocess(self, preds):
        return preds
