from typing import Dict

from .yolo11.detect import Yolo11Detector
from .interface import IDetector

def get_detector(name: str, config: Dict) -> IDetector:
    """"
    Factory method to get the appropriate detector instance based on the name.
    """
    if name == 'yolov11':
        return Yolo11Detector(config)
    else:
        raise ValueError(f"Unknown detector name: {name}")