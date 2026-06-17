"""Run MCT from tracker-level SCT IDs + AIC2024-style global ID assignment."""

from __future__ import annotations

import argparse

from core.log_setup import configure_logging
from pipelines.mct_aic_pipeline import MCTAICPipeline
from utils.load_config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sct-config", default="configs/sct_config.yaml")
    parser.add_argument("--mct-config", default="configs/mct_config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--txt-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging()
    sct_config = load_config(args.sct_config)
    pipeline = MCTAICPipeline(
        sct_config=sct_config,
        mct_config_path=args.mct_config,
    )
    pipeline.run(output_path=args.output, txt_dir=args.txt_dir)


if __name__ == "__main__":
    main()
