import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rtmlib import RTMPose
from pose import is_full_body

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def calculate_img_area(img: np.ndarray) -> float:
	h, w = img.shape[:2]
	return float(w * h)


def calculate_img_ratio(img: np.ndarray) -> float:
	h, w = img.shape[:2]
	if w <= 0:
		return 0.0
	return float(h / w)


def _summary(values: list[float]) -> dict[str, float]:
	values_np = np.asarray(values, dtype=np.float64)
	return {
		"mean": float(np.mean(values_np)),
		"std": float(np.std(values_np)),
		"min": float(np.min(values_np)),
		"max": float(np.max(values_np)),
		"median": float(np.median(values_np)),
		"p5": float(np.percentile(values_np, 5)),
		"p95": float(np.percentile(values_np, 95)),
	}


def calculate_full_body_img_stats(
	root_dir: str,
	model_path: str,
	input_size: tuple[int, int] = (192, 256),
	device: str = "cpu",
	confidence_threshold: float = 0.5,
) -> dict[str, Any]:
	root_path = Path(root_dir)
	if not root_path.exists() or not root_path.is_dir():
		raise ValueError(f"Invalid root directory: {root_dir}")

	image_paths = [
		path
		for path in root_path.rglob("*")
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
	]
	if not image_paths:
		raise ValueError(f"No images found in directory: {root_dir}")

	pose_estimator = RTMPose(
		onnx_model=model_path,
		model_input_size=input_size,
		device=device,
	)

	selected_ratios: list[float] = []
	selected_areas: list[float] = []
	unreadable_images = 0
	pose_failed_images = 0

	for image_path in image_paths:
		img = cv2.imread(str(image_path))
		if img is None:
			unreadable_images += 1
			continue

		h, w = img.shape[:2]
		if h == 0 or w == 0:
			unreadable_images += 1
			continue

		full_image_box = [[0.0, 0.0, float(w - 1), float(h - 1)]]

		try:
			kpts, scores = pose_estimator(img, full_image_box)
			if len(kpts) == 0:
				pose_failed_images += 1
				continue
			scores_expand = scores[..., np.newaxis]
			kpts_scores = np.concatenate([kpts, scores_expand], axis=-1)
		except Exception:
			pose_failed_images += 1
			continue

		if not is_full_body(kpts_scores[0], confidence_threshold=confidence_threshold):
			continue

		selected_ratios.append(calculate_img_ratio(img))
		selected_areas.append(calculate_img_area(img))

	if not selected_ratios:
		return {
			"img_ratio": None,
			"area": None,
			"num_total_images": len(image_paths),
			"num_selected_full_body_images": 0,
			"num_unreadable_images": unreadable_images,
			"num_pose_failed_images": pose_failed_images,
		}

	return {
		"img_ratio": _summary(selected_ratios),
		"area": _summary(selected_areas),
		"num_total_images": len(image_paths),
		"num_selected_full_body_images": len(selected_ratios),
		"num_unreadable_images": unreadable_images,
		"num_pose_failed_images": pose_failed_images,
	}


def split_images_by_full_body(
	root_dir: str,
	output_dir: str,
	model_path: str,
	input_size: tuple[int, int] = (192, 256),
	device: str = "cpu",
	confidence_threshold: float = 0.5,
) -> dict[str, Any]:
	root_path = Path(root_dir)
	if not root_path.exists() or not root_path.is_dir():
		raise ValueError(f"Invalid root directory: {root_dir}")

	output_path = Path(output_dir)
	full_dir = output_path / "full"
	not_full_dir = output_path / "not_full"
	full_dir.mkdir(parents=True, exist_ok=True)
	not_full_dir.mkdir(parents=True, exist_ok=True)

	image_paths = [
		path
		for path in root_path.rglob("*")
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
	]
	if not image_paths:
		raise ValueError(f"No images found in directory: {root_dir}")

	pose_estimator = RTMPose(
		onnx_model=model_path,
		model_input_size=input_size,
		device=device,
	)

	full_count = 0
	not_full_count = 0
	unreadable_images = 0
	pose_failed_images = 0

	for image_path in image_paths:
		rel_path = image_path.relative_to(root_path)
		img = cv2.imread(str(image_path))
		if img is None:
			unreadable_images += 1
			dst_path = not_full_dir / rel_path
			dst_path.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(image_path, dst_path)
			not_full_count += 1
			continue

		h, w = img.shape[:2]
		if h == 0 or w == 0:
			unreadable_images += 1
			dst_path = not_full_dir / rel_path
			dst_path.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(image_path, dst_path)
			not_full_count += 1
			continue

		full_image_box = [[0.0, 0.0, float(w - 1), float(h - 1)]]
		is_full = False
		try:
			kpts, scores = pose_estimator(img, full_image_box)
			if len(kpts) > 0:
				scores_expand = scores[..., np.newaxis]
				kpts_scores = np.concatenate([kpts, scores_expand], axis=-1)
				is_full = is_full_body(kpts_scores, confidence_threshold=confidence_threshold)
			else:
				pose_failed_images += 1
		except Exception:
			pose_failed_images += 1

		dst_root = full_dir if is_full else not_full_dir
		dst_path = dst_root / rel_path
		dst_path.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(image_path, dst_path)

		if is_full:
			full_count += 1
		else:
			not_full_count += 1

	return {
		"num_total_images": len(image_paths),
		"num_full_images": full_count,
		"num_not_full_images": not_full_count,
		"num_unreadable_images": unreadable_images,
		"num_pose_failed_images": pose_failed_images,
		"full_dir": str(full_dir),
		"not_full_dir": str(not_full_dir),
	}


if __name__ == "__main__":
	root_dir = "/home/dungnt/workspaces/HUST/DATN/be/debug"
	output_dir = "/home/dungnt/workspaces/HUST/DATN/be/debug_split"
	model_path = "/home/dungnt/workspaces/HUST/DATN/be/weights/rtmpose-l_256x192/end2end.onnx"
	input_size = (192, 256)
	device = "cuda"
	confidence_threshold = 0.7

	# stats = calculate_full_body_img_stats(
	# 	root_dir=root_dir,
	# 	model_path=model_path,
	# 	input_size=input_size,
	# 	device=device,
	# 	confidence_threshold=confidence_threshold,
	# )
	# print(json.dumps(stats, indent=2, ensure_ascii=False))

	split_result = split_images_by_full_body(
		root_dir=root_dir,
		output_dir=output_dir,
		model_path=model_path,
		input_size=input_size,
		device=device,
		confidence_threshold=confidence_threshold,
	)
	print(json.dumps(split_result, indent=2, ensure_ascii=False))