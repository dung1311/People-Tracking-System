import yaml
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Auto-patch device configuration based on PyTorch CUDA availability
    if isinstance(config, dict):
        device_str = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device_str = "cuda"
        except Exception:
            pass
            
        patched = False
        if "DETECTION" in config and isinstance(config["DETECTION"], dict):
            # Patch yolov11
            yolo_cfg = config["DETECTION"].get("yolov11")
            if isinstance(yolo_cfg, dict) and yolo_cfg.get("device") != device_str:
                yolo_cfg["device"] = device_str
                patched = True
                
        if "POSE_ESTIMATION" in config and isinstance(config["POSE_ESTIMATION"], dict):
            # Patch rtmpose
            rtm_cfg = config["POSE_ESTIMATION"].get("rtmpose")
            if isinstance(rtm_cfg, dict) and rtm_cfg.get("device") != device_str:
                rtm_cfg["device"] = device_str
                patched = True
                
        if patched:
            logger.info(f"Dynamically patched config loaded from {config_path} to use '{device_str}' device.")
            
    return config