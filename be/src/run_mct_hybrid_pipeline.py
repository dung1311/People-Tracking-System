import os
from pipelines.mct_hybrid_pipeline import MCTHybridPipeline
from utils.load_config import load_config
from core.log_setup import configure_logging

if __name__ == "__main__":
    configure_logging()
    
    # Load configuration files using absolute paths to avoid ambiguity
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sct_config_path = os.path.join(base_dir, "configs", "sct_config.yaml")
    mct_config_path = os.path.join(base_dir, "configs", "mct_hybrid_config.yaml")
    
    sct_config = load_config(sct_config_path)
    
    # Initialize and run the pipeline
    pipeline = MCTHybridPipeline(sct_config, mct_config_path)
    pipeline.run(
        output_path="outputs/mct_hybrid_output.mp4",
        txt_dir="outputs/txt"
    )
