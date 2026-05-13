from abc import ABCMeta, abstractmethod


class IDetector(object, metaclass=ABCMeta):
    def __init__(self):
        """
        @The constructor
        """
        pass

    @abstractmethod
    def _preprocess(self, img_brg):
        raise NotImplementedError

    @abstractmethod
    def detect(self, img_brg):
        """
        @return: pose/bbox coordinates
        """
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, preds):
        raise NotImplementedError
