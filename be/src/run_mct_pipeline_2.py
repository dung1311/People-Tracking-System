"""Run MCT Pipeline v2."""

import argparse
import logging

from utils.load_config import load_config
from pipelines.mct_pipeline_2 import MCTPipeline2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="MCT Pipeline v2")
    parser.add_argument(
        "--sct-config", default="configs/sct_config.yaml",
        help="Path to SCT config YAML",
    )
    parser.add_argument(
        "--mct-config", default="configs/awl_config.yaml",
        help="Path to MCT config YAML",
    )
    args = parser.parse_args()

    sct_config = load_config(args.sct_config)
    pipeline = MCTPipeline2(sct_config, args.mct_config)
    pipeline.run()


if __name__ == "__main__":
    main()
