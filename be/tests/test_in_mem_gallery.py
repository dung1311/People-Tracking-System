import sys
import os
import pytest
import numpy as np
from typing import Dict, List, Set

# ==========================================
# 1. SETUP PATH (GIỐNG HỆT CODE CỦA BẠN)
# ==========================================
# Đoạn này đảm bảo Python nhìn thấy thư mục 'src' từ thư mục cha
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==========================================
# 2. IMPORT MODULE TỪ SRC
# ==========================================
try:
    # Import đúng theo đường dẫn bạn cung cấp trong code mẫu
    from src.modules.data_templates.sct_template import TrackInfo, TrackState
    from be.src.modules.gallery.in_mem_gallery import InMemGallery
except ImportError as e:
    # Fallback cho trường hợp chạy pytest tại root mà không cần sys.path
    try:
        from src.modules.data_templates.sct_template import TrackInfo, TrackState
        from be.src.modules.gallery.in_mem_gallery import InMemGallery
    except ImportError:
        raise ImportError(f"Không thể import module. Hãy đảm bảo bạn đang chạy từ root project hoặc thư mục tests. Lỗi chi tiết: {e}")

# ==========================================
# 3. FIXTURES & HELPER
# ==========================================

@pytest.fixture
def gallery_config():
    return {
        "max_age": 3,
        "min_hits": 3
    }

@pytest.fixture
def gallery(gallery_config):
    return InMemGallery(gallery_config)

@pytest.fixture
def dummy_frame_info():
    return {"frame_id": 1, "timestamp": 1000}

@pytest.fixture
def sample_feat():
    # Giả lập feature vector (ví dụ 128 chiều)
    return np.random.rand(128).astype(np.float32)

# ==========================================
# 4. TEST CASES
# ==========================================

def test_init(gallery, gallery_config):
    """Kiểm tra khởi tạo gallery."""
    assert gallery.max_age == gallery_config["max_age"]
    assert gallery.min_hits == gallery_config["min_hits"]
    assert gallery.next_id == 1
    assert len(gallery.tracks) == 0

def test_add_unconfirmed(gallery, dummy_frame_info, sample_feat):
    """Test logic xác nhận track (min_hits)."""
    tracker_id = 101
    bbox = [10, 10, 50, 50]
    
    # Lần 1: Chưa đủ min_hits (3)
    res = gallery.add_or_update_unconfirmed(tracker_id, bbox, sample_feat, dummy_frame_info)
    assert res is None
    assert gallery.unconfirmed[tracker_id].hits == 1

    # Lần 2
    gallery.add_or_update_unconfirmed(tracker_id, bbox, sample_feat, dummy_frame_info)
    
    # Lần 3: Đủ -> Trả về TrackInfo
    res = gallery.add_or_update_unconfirmed(tracker_id, bbox, sample_feat, dummy_frame_info)
    assert res is not None
    assert res.tracker_id == tracker_id
    assert res.hits == 3

def test_promote_new_person(gallery, dummy_frame_info, sample_feat):
    """Test promote track mới -> sinh person_id mới."""
    tracker_id = 200
    # Tạo track giả (như thể đã qua bước unconfirmed)
    track = TrackInfo(tracker_id=tracker_id, bbox=[0,0,1,1], feat=sample_feat, frame_info=dummy_frame_info)
    
    # Setup trạng thái giả: track này đang nằm trong unconfirmed
    gallery.unconfirmed[tracker_id] = track

    pid = gallery.promote_to_active(track, person_id=None)

    assert pid == 1 # next_id ban đầu là 1
    assert gallery.next_id == 2
    assert 1 in gallery.tracks
    assert gallery.map_id[tracker_id] == 1
    assert tracker_id not in gallery.unconfirmed # Phải bị xóa khỏi unconfirmed

def test_promote_reid(gallery, dummy_frame_info, sample_feat):
    """Test Re-ID: Tracker mới map vào Person cũ."""
    # 1. Tạo Person cũ (PID=5, Tracker=100)
    old_track = TrackInfo(tracker_id=100, bbox=[0,0,0,0], feat=sample_feat, frame_info=dummy_frame_info)
    old_track.person_id = 5
    gallery.tracks[5] = old_track
    gallery.map_id[100] = 5

    # 2. Track mới (Tracker=200) được nhận diện là PID=5
    new_track = TrackInfo(tracker_id=200, bbox=[10,10,20,20], feat=sample_feat, frame_info=dummy_frame_info)

    pid = gallery.promote_to_active(new_track, person_id=5)

    assert pid == 5
    assert 200 in gallery.map_id
    assert gallery.map_id[200] == 5
    assert 100 not in gallery.map_id # Map cũ bị xóa
    
    # Check data update
    assert gallery.tracks[5].tracker_id == 200
    assert gallery.tracks[5].state == TrackState.ACTIVE

def test_cleanup_lifecycle(gallery, dummy_frame_info, sample_feat):
    """Test vòng đời: Active -> Lost -> Dead."""
    # Setup track (PID=1, Tracker=10)
    track = TrackInfo(10, [0,0,0,0], sample_feat, dummy_frame_info)
    track.person_id = 1
    track.state = TrackState.ACTIVE
    gallery.tracks[1] = track
    gallery.map_id[10] = 1

    # 1. Frame này không thấy track -> Chuyển sang LOST
    gallery.clean_up(current_tracker_ids=set(), current_person_ids=set())
    assert gallery.tracks[1].state == TrackState.LOST
    assert gallery.tracks[1].lost_age == 0

    # 2. Vẫn không thấy -> Tăng lost_age
    gallery.clean_up(current_tracker_ids=set(), current_person_ids=set()) # Age = 1
    gallery.clean_up(current_tracker_ids=set(), current_person_ids=set()) # Age = 2
    gallery.clean_up(current_tracker_ids=set(), current_person_ids=set()) # Age = 3 (max_age)
    
    assert 1 in gallery.tracks # Vẫn còn vì age = 3 chưa > 3

    # 3. Lần nữa -> Age = 4 -> Xóa
    gallery.clean_up(current_tracker_ids=set(), current_person_ids=set())
    assert 1 not in gallery.tracks
    assert 10 not in gallery.map_id

# ==========================================
# 5. MAIN BLOCK (Để chạy trực tiếp được như script của bạn)
# ==========================================
if __name__ == "__main__":
    # Cho phép chạy file này bằng lệnh: python tests/test_gallery_final.py
    # Đoạn này giả lập việc chạy pytest bằng code thường để bạn debug nếu cần
    print("Đang chạy test thủ công...")
    
    # Manual setup
    cfg = {"max_age": 3, "min_hits": 3}
    gal = InMemGallery(cfg)
    frame_info = {"id": 1}
    feat = np.zeros(128)
    
    # Test thử 1 hàm
    res = gal.add_or_update_unconfirmed(1, [0,0,0,0], feat, frame_info)
    print(f"Test add unconfirmed (lần 1): {res} (Kỳ vọng: None)")
    
    print("Hoàn tất test thủ công. Để chạy đầy đủ hãy dùng lệnh: pytest tests/test_gallery_final.py")