from abc import ABCMeta, abstractmethod

class IPoseEsimation(object, metaclass=ABCMeta):
    def __init__(self):
        """
        @The constructor
        """
        pass

    @abstractmethod
    def _preprocess(self, img_brg):
        """
            @params: img_bgr Image in BGR color
        """
        raise NotImplementedError

    @abstractmethod
    def estimate(self, img_brg):
        """
        @return: pose/bbox coordinates
        """
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, preds):
        raise NotImplementedError
