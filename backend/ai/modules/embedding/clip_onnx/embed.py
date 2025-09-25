import numpy as np
import cv2
from .onnx_runtime import Network
from ..base import IEmbedding

mean = [127.5, 127.5, 127.5]
std = [127.5, 127.5, 127.5]

class Embedding(IEmbedding):
    def __init__(self, cfg):
        self.cfg = cfg
        self.network = Network(self.cfg['model_path'], self.cfg['device'])
              
    def _preprocess(self, imgs_bgr):
        batch_imgs = []
        for img in imgs_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.network.input_width, self.network.input_height))
            img = (img - mean)/std
            img = img.transpose(2, 0, 1)
            batch_imgs.append(img)
        batch_imgs = np.array(batch_imgs).astype(np.float32)
        return batch_imgs
    
    
    def extract_feature(self, imgs_BRG):
        pre_imgs = self._preprocess(imgs_BRG)
        outputs = self.network.inference(pre_imgs)[0]  
        return np.array(outputs)
        
if __name__ == '__main__':
    # @ ./src$ python -m modules.vectorization.fastreid.vectorization
    import time
    import cv2

    # config
    vec_cfg = {
        "model_path": "/home/materials/models/clipreid/ViTB16e30_h256w128e1280_ccdmmpss.onnx",
        "device": 1 # cuda
    }
    
    vectorizer = Embedding(vec_cfg)
    img = cv2.imread('../materials/media/test.png')
    n = 100
    img_list = [img for i in range(n)]
    st_time = time.time ()
    features = vectorizer.extract_feature(img_list)
    print('FPS: {}'.format(round(n / (time.time() - st_time), 2)))
    print(f"feature shape: {features.shape}")
    assert features.shape[0] == n, 'Num. of vectors must be equal to num. of images'