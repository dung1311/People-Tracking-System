import numpy as np
import cv2
from modules.inference_engine.onnx_runtime import Network
from .interface import FeatureExtractorBase

class FeatureExtractor(FeatureExtractorBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.network = Network(self.cfg['model_path'], self.cfg['device'])
        self.mean = self.cfg['mean']
        self.std = self.cfg['std']
        self.model_name = self.cfg['model_name']
        self.vector_dim = self.network.output_shape[1]
              
    def preprocess(self, imgs):
        batch_imgs = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.network.input_width, self.network.input_height))
            img = (img - self.mean)/self.std
            img = img.transpose(2, 0, 1)
            batch_imgs.append(img)
        batch_imgs = np.array(batch_imgs).astype(np.float32)
        return batch_imgs
    
    def extract_feature(self, imgs):
        pre_imgs = self.preprocess(imgs)
        outputs = self.network.inference(pre_imgs)[0]  
        return np.array(outputs)
 