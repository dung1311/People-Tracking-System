import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config