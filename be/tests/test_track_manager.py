import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modules.track_manager.single_track_manager import SingleTrackManager
from src.utils.load_config import load_config

config = load_config("/home/dungnt/workspaces/HUST/DATN/be/configs/sct_config.yaml")

stm = SingleTrackManager(config["TRACK_MANAGER"])
