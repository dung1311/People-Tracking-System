from typing import Dict

from rtmlib import RTMPose
import numpy as np  

from ..base import BasePoseEstimator

class RTMPoseEstimator(BasePoseEstimator):
    def __init__(self, config: Dict):
        self.__device = config["device"]
        self.__weight = config["model_path"]
        self.__input_size = tuple(config["input_size"][:2])
        self.__model = RTMPose(
            onnx_model=self.__weight,
            model_input_size=self.__input_size,
            device=self.__device
        )
    
    def _preprocess(self, imgs):
        return imgs

    def detect(self, imgs, boxes):
        if len(boxes) == 0:
            return []
        
        # Only use [x1, y1, x2, y2] format
        boxes = [box[:4] for box in boxes]
        
        kpts, scores = self.__model(imgs, boxes)
        assert len(kpts) == len(boxes)
        
        scores_expand = scores[..., np.newaxis]
        kpts_scores = np.concatenate([kpts, scores_expand], axis=-1)
        
        return self._postprocess(kpts_scores)
    
    def _postprocess(self, preds):
        return preds