from abc import abstractmethod
from typing import List

class BaseTracker:
    def __init__(self):
        pass
    
    @abstractmethod
    def update(self, dets: List[List[float]], **kwargs):
        raise NotImplementedError