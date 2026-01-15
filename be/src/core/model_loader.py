from typing import Optional
from modules.detector.factory import DetectorFactory
from modules.embedder.factory import EmbedderFactory
from utils.load_config import load_config

class ModelLoader:
    _instance = None
    
    def __init__(self, config_path: str = "configs/sct_config.yaml"):
        print(f"Loading models from {config_path}...")
        self.cfg = load_config(config_path)
        
        # Initialize Detector
        self.detector = DetectorFactory(self.cfg["DETECTION"]).get_detector()
        
        # Initialize Embedder (nested under TRACK_MANAGER in this config structure)
        self.embedder = EmbedderFactory(self.cfg["TRACK_MANAGER"]["EMBEDDING"]).get_embedder()
        print("Models loaded successfully.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def get_model_loader() -> ModelLoader:
    return ModelLoader.get_instance()
