from abc import ABC, abstractmethod
from dataclasses import dataclass
from .track_state import TrackState, UnconfirmState
from .events import *

@dataclass
class FrameInfo:
    frame_id: int

EVENTS = {
    "TrackConfirmedEvent": TrackConfirmedEvent,
    "TrackLostEvent": TrackLostEvent,
}


class TrackInfo:
    def __init__(self, tid: int, box: list, frame_info: FrameInfo):
        self.tid = tid 
        self.pid = -1 
        self.box = box
        self.start_frame = frame_info.frame_id
        self.last_frame = frame_info.frame_id
        self.min_hit_streak = 0

        self.state: TrackState = UnconfirmState()
        self.events: list[Event] = []

    def set_state(self, state: TrackState):
        self.state = state
        state_name = state.name
        if state_name in EVENTS.keys():
            event_cls = EVENTS[state_name]
            event = event_cls(tracker_id=self.tid)
            self.events.append(event)
    
    def update(self, box: list, frame_info: FrameInfo):
        self.state.update(self, box, frame_info)
