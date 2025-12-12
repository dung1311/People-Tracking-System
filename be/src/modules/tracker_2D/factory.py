from typing import Dict
import logging

from .base import BaseTracker
from .sort.sort import Sort
from .ocsort.ocsort import OCSort

logger = logging.getLogger(__name__)

class TrackerFactory:
    def __init__(self, config: Dict) -> BaseTracker:
        self.__tracker_name = config["name"]
        self.__config = config
        logger.debug(f"TrackerFactory created with config for: {self.__tracker_name}")
    
    def get_tracker(self):
        logger.info(f"Start initializing tracker: {self.__tracker_name}...")
        
        try:
            if self.__tracker_name == 'sort':
                tracker = Sort(self.__config[self.__tracker_name])
            elif self.__tracker_name == 'ocsort':
                tracker = OCSort(self.__config[self.__tracker_name])
            else:
                raise ValueError(f"Tracker {self.__tracker_name} not supported")
                
            logger.info(f"Successfully initialized tracker: {self.__tracker_name}")
            return tracker
        except Exception as e:
            logger.exception(f"Failed to initialize tracker {self.__tracker_name}")
            raise e
    
    def get_list_trackers(self):
        return ["sort", "ocsort"]