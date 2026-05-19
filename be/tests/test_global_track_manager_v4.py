import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from mct.global_tracks import GlobalTrackManagerV4


def test_global_track_manager_v4_basic():
    config = {
        "reid": 0.4,
        "max_lost_age": 30,
        "feature_smooth": 0.1
    }
    manager = GlobalTrackManagerV4(config)

    # First frame matching
    clusters = [[0], [1]]
    features = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Feature for track 0 on cam 1
        [0.0, 1.0, 0.0, 0.0]   # Feature for track 1 on cam 2
    ], dtype=np.float32)
    cam_ids = np.array([1, 2], dtype=np.int32)
    person_ids = np.array([101, 201], dtype=np.int32)

    mapping = manager.assign(clusters, features, cam_ids, person_ids, frame_id=1)
    
    assert (1, 101) in mapping
    assert (2, 201) in mapping
    
    gid1 = mapping[(1, 101)]
    gid2 = mapping[(2, 201)]
    assert gid1 != gid2

    # Second frame: same track seen on cam 1 again, should match gid1
    clusters_f2 = [[0]]
    features_f2 = np.array([
        [0.98, 0.02, 0.0, 0.0] # Very close to first track's feature
    ], dtype=np.float32)
    cam_ids_f2 = np.array([1], dtype=np.int32)
    person_ids_f2 = np.array([102], dtype=np.int32) # Different local track ID on same camera

    mapping_f2 = manager.assign(clusters_f2, features_f2, cam_ids_f2, person_ids_f2, frame_id=2)
    assert (1, 102) in mapping_f2
    assert mapping_f2[(1, 102)] == gid1

    # Check that cam_features dictionary has separate cameras stored in the global tracks
    gt1 = manager.tracks[gid1]
    assert 1 in gt1.cam_features
    assert 2 not in gt1.cam_features

    # Match across camera: same person 101 moves from cam 1 to cam 2, with slightly similar feature.
    # It should match gid1 based on the stored cam 1 feature of gid1 since it compares features per camera
    clusters_f3 = [[0]]
    features_f3 = np.array([
        [0.95, 0.05, 0.0, 0.0] # Close to original cam 1 feature
    ], dtype=np.float32)
    cam_ids_f3 = np.array([2], dtype=np.int32) # Appears on Cam 2
    person_ids_f3 = np.array([202], dtype=np.int32)

    mapping_f3 = manager.assign(clusters_f3, features_f3, cam_ids_f3, person_ids_f3, frame_id=3)
    assert (2, 202) in mapping_f3
    assert mapping_f3[(2, 202)] == gid1
    
    # Now global track 1 should have features for both Cam 1 and Cam 2
    assert 1 in gt1.cam_features
    assert 2 in gt1.cam_features


if __name__ == "__main__":
    test_global_track_manager_v4_basic()
    print("All tests passed!")
