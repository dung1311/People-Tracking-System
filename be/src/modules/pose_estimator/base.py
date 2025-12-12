from abc import abstractmethod
import numpy as np
from typing import Any, List, Union

class BasePoseEstimator:
    def __init__(self):
        pass
    
    @abstractmethod
    def _preprocess(self, imgs: np.ndarray) -> Any:
        """_summary_

        Args:
            imgs (np.ndarray): List of BGR images
        """
        raise NotImplementedError
    
    @abstractmethod
    def detect(self, imgs: np.ndarray, boxes: Union[List[List[float]]]) -> List[Any]:
        """_summary_

        Args:
            imgs (np.ndarray): List of BGR images
        """
        raise NotImplementedError
    
    @abstractmethod
    def _postprocess(self, preds: Any) -> List[Any]:
        """_summary_

        Args:
            preds (Any): List predictions from object detection model
        """
        raise NotImplementedError
    