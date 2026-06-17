from pipelines.mct_hybrid_pipeline import MCTHybridPipeline
from utils.load_config import load_config
from core.log_setup import configure_logging

if __name__ == "__main__":
    configure_logging()
    
    # Load configuration files
    sct_config = load_config("./configs/sct_config.yaml")
    mct_config_path = "./configs/mct_hybrid_config.yaml"
    
    # Initialize and run the pipeline
    pipeline = MCTHybridPipeline(sct_config, mct_config_path)
    pipeline.run(
        output_path="outputs/mct_hybrid_output.mp4",
        txt_dir="outputs/txt"
    )
