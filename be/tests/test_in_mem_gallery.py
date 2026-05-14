import sys
import os
import pytest
import numpy as np
from typing import Dict, List, Set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from modules.data_templates.sct_template import TrackInfo, TrackState
from modules.gallery.in_mem_gallery import InMemGallery


@pytest.fixture
def gallery_config():
    return {
        "max_live_time": 3,
        "min_hits": 3,
        "max_features": 50,
    }


@pytest.fixture
def gallery(gallery_config):
    return InMemGallery(gallery_config)


@pytest.fixture
def sample_feat():
    feat = np.random.rand(128).astype(np.float32)
    return feat / (np.linalg.norm(feat) + 1e-8)


CAM_ID = 1
FRAME_ID = 1


def test_init(gallery, gallery_config):
    assert gallery.max_age == gallery_config["max_live_time"]
    assert gallery.min_hits == gallery_config["min_hits"]
    assert gallery.next_id == 1
    assert len(gallery.tracks) == 0


def test_add_unconfirmed(gallery, sample_feat):
    tracker_id = 101
    bbox = [10, 10, 50, 50]

    res = gallery.add_or_update_unconfirmed(tracker_id, bbox, sample_feat, CAM_ID, FRAME_ID)
    assert res is None
    assert gallery.unconfirmed[tracker_id].hits == 1

    gallery.add_or_update_unconfirmed(tracker_id, bbox, sample_feat, CAM_ID, FRAME_ID)

    res = gallery.add_or_update_unconfirmed(tracker_id, bbox, sample_feat, CAM_ID, FRAME_ID)
    assert res is not None
    assert res.tracker_id == tracker_id
    assert res.hits == 3


def test_promote_new_person(gallery, sample_feat):
    tracker_id = 200
    track = TrackInfo(tracker_id=tracker_id, bbox=[0, 0, 1, 1], feat=sample_feat,
                      cam_id=CAM_ID, frame_id=FRAME_ID)
    gallery.unconfirmed[tracker_id] = track

    pid = gallery.promote_to_active(track, person_id=None)

    assert pid == 1
    assert gallery.next_id == 2
    assert 1 in gallery.tracks
    assert gallery.map_id[tracker_id] == 1
    assert tracker_id not in gallery.unconfirmed


def test_promote_reid(gallery, sample_feat):
    old_track = TrackInfo(tracker_id=100, bbox=[0, 0, 0, 0], feat=sample_feat,
                          cam_id=CAM_ID, frame_id=FRAME_ID)
    old_track.person_id = 5
    gallery.tracks[5] = old_track
    gallery.map_id[100] = 5

    new_track = TrackInfo(tracker_id=200, bbox=[10, 10, 20, 20], feat=sample_feat,
                          cam_id=CAM_ID, frame_id=FRAME_ID)

    pid = gallery.promote_to_active(new_track, person_id=5)

    assert pid == 5
    assert 200 in gallery.map_id
    assert gallery.map_id[200] == 5
    assert 100 not in gallery.map_id
    assert gallery.tracks[5].tracker_id == 200
    assert gallery.tracks[5].state == TrackState.ACTIVE


def test_cleanup_lifecycle(gallery, sample_feat):
    track = TrackInfo(tracker_id=10, bbox=[0, 0, 0, 0], feat=sample_feat,
                      cam_id=CAM_ID, frame_id=FRAME_ID)
    track.person_id = 1
    track.state = TrackState.ACTIVE
    gallery.tracks[1] = track
    gallery.map_id[10] = 1

    gallery.clean_up(current_tracker_ids=set(), current_person_ids=set())
    assert gallery.tracks[1].state == TrackState.LOST
    assert gallery.tracks[1].lost_age == 1

    gallery.clean_up(current_tracker_ids=set(), current_person_ids=set())  # age=2
    gallery.clean_up(current_tracker_ids=set(), current_person_ids=set())  # age=3 == max_age

    assert 1 not in gallery.tracks
    assert 10 not in gallery.map_id
