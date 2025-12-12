from typing import Dict

from .base import BaseGallery

class GalleryFactory:
    def __init__(self, config: Dict) -> BaseGallery:
        self.config = config
