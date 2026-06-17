from pipelines.sct_pipeline_video import SCTVideoPipeline
from utils.load_config import load_config
from core.log_setup import configure_logging

if __name__ == "__main__":
    configure_logging()
    config = load_config("./configs/sct_config.yaml")
    video_path = "./data/cam3_longs.mp4"
    video_pipeline = SCTVideoPipeline(config, video_path)
    video_pipeline.run()