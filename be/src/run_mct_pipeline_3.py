"""Run MCT Pipeline v3 on cam64/cam65/cam66."""

import logging

from utils.load_config import load_config
from pipelines.mct_pipeline_3 import MCTPipeline3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

def main():
    sct_config = load_config("configs/sct_config.yaml")
    mct_config_path = "configs/mct_config.yaml"
    pipeline = MCTPipeline3(
        sct_config=sct_config,
        mct_config_path=mct_config_path,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
