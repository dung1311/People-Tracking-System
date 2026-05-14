"""Run the Frechet matching pipeline on the three calibrated videos."""

import logging
import sys

from utils.load_config import load_config
from pipelines.frechet_pipeline import FrechetMCTPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

CAMERA_VIDEO_MAP = {
    64: "data/videos/cam64.mp4",
    65: "data/videos/cam65.mp4",
    66: "data/videos/cam66.mp4",
}

CAMERA_CALIB_MAP = {
    64: "data/cameras/cam64.json",
    65: "data/cameras/cam65.json",
    66: "data/cameras/cam66.json",
}


def main():
    sct_config = load_config("configs/sct_config.yaml")

    pipeline = FrechetMCTPipeline(
        sct_config=sct_config,
        camera_video_map=CAMERA_VIDEO_MAP,
        camera_calib_map=CAMERA_CALIB_MAP,
        window_size=30
    )

    pipeline.run()


if __name__ == "__main__":
    main()
