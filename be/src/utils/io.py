from queue import Queue
from threading import Thread
import datetime

import cv2

class FileVideoStream:
    def __init__(self, path: str, queue_size=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        
        self.Q = Queue(maxsize=queue_size)
    
    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return
            
            if not self.Q.full():
                grabbed, frame = self.stream.read()
                
                if not grabbed:
                    self.stop()
                    return 
            
                self.Q.put(frame)
    
    def read(self):
        return self.Q.get()
    
    def more(self):
        return self.Q.qsize() > 0
    
    def stop(self):
        self.stopped = True


class FPS:
    def __init__(self):
        self.__start = None
        self.__end = None
        self.__num_frames = 0
    
    def start(self):
        self.__start = datetime.datetime.now()
        return self 
    
    def stop(self):
        self.__end = datetime.datetime.now()
    
    def update(self):
        self.__num_frames += 1
    
    def elapsed(self):
        if self.__end is None:
            return (datetime.datetime.now() - self.__start).total_seconds()
        return (self.__end - self.__start).total_seconds()
    
    def fps(self):
        return self.__num_frames / self.elapsed() if self.elapsed() > 0 else 0
