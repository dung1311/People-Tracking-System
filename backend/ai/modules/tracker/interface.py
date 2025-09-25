from abc import ABC, abstractmethod

class ITracker(ABC):
    @abstractmethod
    def update(self, detections):
        pass
