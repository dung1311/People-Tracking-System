from utils.load_config import load_config
from src.modules.tracker_2D.factory import TrackerFactory

config = load_config("/home/dungnt/People-Tracking-System/be/configs/sct_config.yaml")
sort = TrackerFactory(config["TRACKING"]).get_tracker()
