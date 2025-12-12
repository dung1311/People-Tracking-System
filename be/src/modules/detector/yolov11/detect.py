from typing import Dict

from ultralytics import YOLO

from ..base import BaseDetector

class Yolov11Detector(BaseDetector):
    def __init__(self, config: Dict, crop_roi=None):
        """_summary_

        Args:
            config (Dict): _description_
        """
        self.__model = YOLO(model=config["model_path"], task=config["task"])
        self.__imgsz = config["imgsz"]
        self.__conf = config["conf_thres"]
        self.__iou = config["iou_thres"]
        self.__device = config["device"]
        self.__classes = config.get("classes", 0)
        self.__max_det = config["max_det"]
        self.__crop_roi = crop_roi
    
    def _preprocess(self, imgs):
        return imgs
    
    def detect(self, imgs):
        _imgs = self._preprocess(imgs)
        preds = self.__model(
            source=_imgs,
            imgsz=self.__imgsz,
            conf=self.__conf,
            iou=self.__iou,
            device=self.__device,
            classes=self.__classes,
            max_det=self.__max_det,
            verbose=False
        )   
        
        return self._postprocess(preds)
    
    def _postprocess(self, preds):
        boxes = []
        for pred in preds:
            
            for box in pred.boxes.data:
                x1, y1, x2, y2, sc, cl = box.cpu().numpy().tolist()
                if self.__crop_roi:            
                    pass
                boxes.append([x1, y1, x2, y2, sc])
        
        return boxes