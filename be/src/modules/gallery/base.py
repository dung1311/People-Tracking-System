from abc import abstractmethod

class BaseGallery:
    def __init__(self):
        pass
    
    @abstractmethod
    def match_one_id(self, query, gallery):
        pass
    