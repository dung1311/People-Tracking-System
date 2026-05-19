"""Run MCT Pipeline v3 on cam64/cam65/cam66."""

import logging
import sys

try:
    import mct  # noqa: F401
except ImportError:
    print(
        "Missing package ``mct``. Install from repo root: pip install -e ./MCT",
        file=sys.stderr,
    )
    sys.exit(1)

from utils.load_config import load_config
from pipelines.mct_pipeline_3 import MCTPipeline3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

def main():
    sct_config = load_config("configs/sct_config.yaml")
    mct_config_path = "configs/pipeline3.yaml"
    pipeline = MCTPipeline3(
        sct_config=sct_config,
        mct_config_path=mct_config_path,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
