from abc import abstractmethod
from typing import List

from torch import Tensor
import numpy as np  

class BaseEmbedder:
    def __init__(self):
        pass
    
    @abstractmethod
    def _preprocess(self, imgs_bgr: List[np.ndarray]):
        """_summary_

        Args:
            imgs_bgr (List[np.ndarray]): List images of crop person

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def extract_feature(self, imgs_bgr: List[np.ndarray]) -> List[Tensor]:
        """_summary_

        Args:
            imgs_bgr (List[np.ndarray]): List images of crop person
        """
        raise NotImplementedError
    