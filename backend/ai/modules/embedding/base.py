
from abc import ABCMeta, abstractmethod

class IEmbedding(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def _preprocess(self, imgs_bgr):
        """
        @imgs_BRG: BRG images (person crops)
        """
        pass

    @abstractmethod
    def extract_feature(self, imgs_BRG):
        """
        @imgs_BRG: BRG images (person crops)
        """
        pass