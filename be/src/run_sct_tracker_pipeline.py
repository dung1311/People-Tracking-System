"""Run SCT with ReID handled directly inside SORT/OCSort."""

from __future__ import annotations

import argparse

from core.log_setup import configure_logging
from pipelines.sct_tracker_pipeline import SCTTrackerPipeline
from utils.load_config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sct_config.yaml")
    parser.add_argument("--video", default="data/4cams/cam2.mp4")
    parser.add_argument("--output", default=None)
    parser.add_argument("--txt", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging()
    config = load_config(args.config)
    pipeline = SCTTrackerPipeline(config, args.video)
    pipeline.run(output_path=args.output, txt_path=args.txt)


if __name__ == "__main__":
    main()
