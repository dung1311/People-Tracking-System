from typing import Dict, List
import logging

from .base import BaseEmbedder
from .fastreid.embed import Embedding
from .onnx_reid.embed import OnnxReidEmbedder

logger = logging.getLogger(__name__)

class EmbedderFactory:
    def __init__(self, config: Dict):
        self.__model_name = config["name"]
        self.__config = config
        logger.debug(f"EmbeddingFactory created with config for: {self.__model_name}")
    
    def get_embedder(self) -> BaseEmbedder:
        logger.info(f"Start initializing model: {self.__model_name}...")

        if self.__model_name == "fastreid":
            try:
                model = Embedding(self.__config[self.__model_name])
                logger.info(f"Successfully initialized reid model: {self.__model_name}")
                return model
            except Exception as e:
                logger.exception(f"Failed to initialize model {self.__model_name}")
                raise e

        elif self.__model_name == "onnx_reid":
            try:
                model = OnnxReidEmbedder(self.__config[self.__model_name])
                logger.info(f"Successfully initialized reid model: {self.__model_name}")
                return model
            except Exception as e:
                logger.exception(f"Failed to initialize model {self.__model_name}")
                raise e

        else:
            logger.error(f"Unsupported model name requested: {self.__model_name}")
            raise ValueError(f"Unsupport model name: {self.__model_name}")

    def get_list_embedders(self) -> List[str]:
        return ["fastreid", "onnx_reid"]