"""Multi-camera tracking core (NumPy/SciPy)."""

from mct.calibration import CameraCalibration, CameraPairCalibration
from mct.clustering import CrossCameraClusterer
from mct.distance import (
    cosine_distance_matrix,
    euclidean_distance_matrix,
    l2_normalise_rows,
    project_feet_to_world,
)
from mct.global_tracks import GlobalTrack, GlobalTrackManagerV2, GlobalTrackV4, GlobalTrackManagerV4
from mct.union_find import UnionFind

__all__ = [
    "CameraCalibration",
    "CameraPairCalibration",
    "CrossCameraClusterer",
    "GlobalTrack",
    "GlobalTrackManagerV2",
    "GlobalTrackV4",
    "GlobalTrackManagerV4",
    "UnionFind",
    "cosine_distance_matrix",
    "euclidean_distance_matrix",
    "l2_normalise_rows",
    "project_feet_to_world",
]

__version__ = "0.1.0"
