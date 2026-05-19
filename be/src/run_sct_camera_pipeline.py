from pipelines.sct_pipeline_camera import SCTCameraPipeline
from utils.load_config import load_config
from core.log_setup import configure_logging

if __name__ == "__main__":
    configure_logging()
    config = load_config("./configs/sct_config.yaml")
    video_path = "https://192.168.100.210:8080/video" # Use integer for webcam ID 
    video_pipeline = SCTCameraPipeline(config, video_path)
    video_pipeline.run()