from abc import ABC, abstractmethod
from .track import FrameInfo, TrackInfo

class TrackState(ABC):
    @abstractmethod
    def update(self, track: TrackInfo, box: list, frame_info: FrameInfo):
        track.box = box
        track.last_frame = frame_info.frame_id

    
    @property
    def name(self):
        return self.__class__.__name__

class UnconfirmState(TrackState):
    def update(self, track: TrackInfo, box: list, frame_info: FrameInfo):
        super().update(track, box, frame_info)
        