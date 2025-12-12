from typing import Dict, List
import logging

from .base import BaseDetector
from .yolov11.detect import Yolov11Detector

logger = logging.getLogger(__name__)

class DetectorFactory:
    def __init__(self, config: Dict):
        self.__model_name = config["name"]
        self.__config = config
        logger.debug(f"DetectorFactory created with config for: {self.__model_name}")
    
    def get_detector(self) -> BaseDetector:
        logger.info(f"Start initializing model: {self.__model_name}...")

        if self.__model_name == "yolov11":
            try:
                model = Yolov11Detector(self.__config[self.__model_name])
                
                logger.info(f"Successfully initialized object detection model: {self.__model_name}")
                
                return model
            except Exception as e:
                logger.exception(f"Failed to initialize model {self.__model_name}")
                raise e

        else:
            logger.error(f"Unsupported model name requested: {self.__model_name}")
            raise ValueError(f"Unsupport model name: {self.__model_name}")

    def get_list_detector(self) -> List[str]:
        return ["yolov11"]