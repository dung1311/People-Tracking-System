from typing import Dict

from .sct.sort import Sort
from .interface import ITracker

def get_tracker_sct(name: str, config: Dict) -> ITracker:
    if name == 'SORT':
        return Sort(config)
    else:
        raise ValueError(f"Unknown tracker name: {name}")
