import yaml
from typing import Dict, Any
from pprint import pprint

def load_config(config_path: str | Any) -> Dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config

if __name__ == '__main__':
    config = load_config('/home/dungnt/AWL/Tracking3DPose/src/config/config.yaml') 
    pprint(config)