from typing import Dict, List
import numpy as np
import cv2
from ultralytics import YOLO
from ..interface import IPoseEsimation

class YoloPose(IPoseEsimation):
    def __init__(self, config: Dict):
        self.device = config["device"]
        self.conf_threshold = config["conf_threshold"]
        self.weight = config["weight"]

        self.model = YOLO(self.weight, task='pose', verbose=False).to(self.device)
    
    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        return img_bgr

    def estimate(self, image: np.ndarray, boxes: List[List[float]]) -> List[np.ndarray]:
        img = self._preprocess(image)
        predicts = self.predict(img, boxes)
        return self._postprocess(predicts)

    def predict(self, image: np.ndarray, boxes: List[List[float]]) -> List[np.ndarray]:
        if len(boxes) == 0:
            return []

        results = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Let the model handle preprocessing internally
            output = self.model(crop, verbose=False)[0]
            if output.keypoints is None:
                continue

            keypoints = output.keypoints.data.cpu().numpy()  # (1, 17, 3)
            if keypoints.shape[0] == 0:
                continue

            for kpts in keypoints:
                # Scale keypoints back to original coordinates
                kpts[:, 0] = kpts[:, 0] * (x2 - x1) / 640 + x1
                kpts[:, 1] = kpts[:, 1] * (y2 - y1) / 640 + y1  # assuming 640 input
                results.append(kpts)

        return results

    def _postprocess(self, preds: List[np.ndarray]) -> List[np.ndarray]:
        return preds

    def predict_bb(self, image: np.ndarray) -> List[np.ndarray]:
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        bboxes = []

        for result in results:
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, score, cls_int = box
                bboxes.append(np.array([x1, y1, x2, y2]))

        return bboxes
