from pipelines.sct_pipeline_video import SCTVideoPipeline
from utils.load_config import load_config

if __name__ == "__main__":
    config = load_config("./configs/sct_config.yaml")
    video_path = "./data/videos/cam0.mp4"
    video_pipeline = SCTVideoPipeline(config, video_path)
    video_pipeline.run()