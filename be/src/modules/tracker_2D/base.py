from abc import abstractmethod
from typing import List, Optional, Dict

class BaseTracker:
    def __init__(self):
        pass
    
    @abstractmethod
    def update(self, dets: List[List[float]], frame_info: Optional[Dict]):
        raise NotImplementedError