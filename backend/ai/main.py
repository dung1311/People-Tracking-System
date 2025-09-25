from pipelines.sct import SCT
import load_config

if __name__ == "__main__":
    config = load_config.load_config("/home/dungnt/People-Tracking-System/backend/configs/sct_config.yaml")

    sct = SCT(config, config)
    sct.run()