"""Run MCT Pipeline v4."""

import argparse
import logging
import sys

try:
    import mct  # noqa: F401 — ensures ``people-mct`` is installed
except ImportError:
    print(
        "Missing package ``mct``. Install from repo root: pip install -e ./MCT",
        file=sys.stderr,
    )
    sys.exit(1)

from utils.load_config import load_config
from pipelines.mct_pipeline_4 import MCTPipeline4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="MCT Pipeline v4")
    parser.add_argument(
        "--sct-config", default="configs/sct_config.yaml",
        help="Path to SCT config YAML",
    )
    parser.add_argument(
        "--mct-config", default="configs/pipeline4.yaml",
        help="Path to MCT config YAML",
    )
    args = parser.parse_args()

    sct_config = load_config(args.sct_config)
    pipeline = MCTPipeline4(sct_config, args.mct_config)
    pipeline.run()


if __name__ == "__main__":
    main()
