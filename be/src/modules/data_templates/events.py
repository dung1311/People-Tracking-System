from dataclasses import dataclass

@dataclass
class Event:
    pass

@dataclass
class TrackConfirmedEvent(Event):
    tracker_id: int

@dataclass
class TrackLostEvent(Event):
    tracker_id: int
