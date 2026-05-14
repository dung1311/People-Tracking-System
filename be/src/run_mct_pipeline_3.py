"""Run MCT Pipeline v3 on cam64/cam65/cam66."""

import logging

from utils.load_config import load_config
from pipelines.mct_pipeline_3 import MCTPipeline3

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
    pipeline = MCTPipeline3(
        sct_config=sct_config,
        camera_video_map=CAMERA_VIDEO_MAP,
        camera_calib_map=CAMERA_CALIB_MAP,
    )
    pipeline.run(
        output_path="outputs/mct3_output.mp4",
        txt_dir="outputs/txt",
    )


if __name__ == "__main__":
    main()
