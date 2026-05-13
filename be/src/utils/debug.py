import os
import sys
import cv2
import yaml
import numpy as np
from glob import glob
from typing import Dict, List, Union

# Thêm thư mục src vào sys.path để Python tìm thấy package 'modules'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.embedder.factory import EmbedderFactory

def calculate_embedding_distance(folder1: str, folder2: str, config_path: str = "configs/sct_config.yaml") -> np.ndarray:
    """
    Tính khoảng cách cosine giữa các features của ảnh trong 2 folder, sử dụng model của dự án.
    
    Args:
        folder1: Đường dẫn đến folder 1
        folder2: Đường dẫn đến folder 2
        config_path: Đường dẫn đến file config để khởi tạo model Embedder
        
    Returns:
        Ma trận khoảng cách cosine kích thước (số ảnh folder1 x số ảnh folder2)
    """
    # 1. Khởi tạo model embedder từ config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    embedder_config = config["TRACK_MANAGER"]["EMBEDDING"]
    embedder = EmbedderFactory(embedder_config).get_embedder()
    
    def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
        images = []
        extensions = ("*.jpg", "*.jpeg", "*.png")
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob(os.path.join(folder_path, ext)))
            image_paths.extend(glob(os.path.join(folder_path, ext.upper())))
            
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
        return images

    # 2. Đọc tất cả hình ảnh trong hai folders
    images1 = load_images_from_folder(folder1)
    images2 = load_images_from_folder(folder2)
            
    if not images1 or not images2:
        print(f"Thiếu ảnh! Folder 1 có {len(images1)} ảnh, Folder 2 có {len(images2)} ảnh.")
        return np.array([])
        
    # 3. Trích xuất đặc trưng (embeddings) với batching
    batch_size = 32
    def extract_in_batches(images):
        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_features = embedder.extract_feature(batch)
            if hasattr(batch_features, "cpu"):
                batch_features = batch_features.detach().cpu().numpy()
            features.extend(batch_features)
        return features

    print(f"Bắt đầu trích xuất đặc trưng cho {len(images1)} ảnh từ {folder1}...")
    features1 = extract_in_batches(images1)
    
    print(f"Bắt đầu trích xuất đặc trưng cho {len(images2)} ảnh từ {folder2}...")
    features2 = extract_in_batches(images2)
        
    # 4. Chuẩn hoá đặc trưng
    features1 = np.array([f / (np.linalg.norm(f) + 1e-8) for f in features1])
    features2 = np.array([f / (np.linalg.norm(f) + 1e-8) for f in features2])
    
    # 5. Tính từng cặp Cosine Distance = 1 - Cosine Similarity
    distances = np.zeros((len(features1), len(features2)))
    for i, f1 in enumerate(features1):
        for j, f2 in enumerate(features2):
            distances[i, j] = 1 - np.dot(f1, f2)
            
    print("-"*40)
    print(f"Thống kê khoảng cách giữa '{folder1}' và '{folder2}':")
    print(f"Khoảng cách nhỏ nhất: {np.min(distances):.4f}")
    print(f"Khoảng cách lớn nhất: {np.max(distances):.4f}")
    print(f"Khoảng cách trung bình: {np.mean(distances):.4f}")
    print("-"*40)
    
    return distances

if __name__ == "__main__":
    # Ví dụ bạn có thể chạy file này trực tiếp từ terminal và test.
    # Nhớ đứng từ thư mục 'be' khi chạy file:
    # `python -m src.utils.debug`
    calculate_embedding_distance("/home/dungnt/People-Tracking-System/be/debug/1", "/home/dungnt/People-Tracking-System/be/debug/4")
