from typing import Dict, List
import logging

from .base import BasePoseEstimator
from .rtmpose.detect import RTMPoseEstimator

logger = logging.getLogger(__name__)

class PoseEstimatorFactory:
    def __init__(self, config: Dict):
        self.__model_name = config["name"]
        self.__config = config
        logger.debug(f"PoseEstimatorFactory created with config for: {self.__model_name}")
    
    def get_pose_estimator(self) -> BasePoseEstimator:
        logger.info(f"Start initializing model: {self.__model_name}...")

        if self.__model_name == "rtmpose":
            try:
                model = RTMPoseEstimator(self.__config[self.__model_name])
                
                logger.info(f"Successfully initialized pose estimation model: {self.__model_name}")
                
                return model
            except Exception as e:
                logger.exception(f"Failed to initialize model {self.__model_name}")
                raise e

        else:
            logger.error(f"Unsupported model name requested: {self.__model_name}")
            raise ValueError(f"Unsupport model name: {self.__model_name}")

    def get_list_pose_estimator(self) -> List[str]:
        return ["rtmpose"]