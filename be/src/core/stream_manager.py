from typing import Dict, Optional
import threading

class StreamManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(StreamManager, cls).__new__(cls)
                    cls._instance.pipelines = {}
                    cls._instance.pipeline_configs = {}
        return cls._instance

    def set_config(self, pipeline_config: Dict):
        """Store global pipeline config"""
        self.pipeline_config = pipeline_config

    def get_pipeline(self, camera_id: int, source: str):
        """Get or create a pipeline for the specific camera"""
        if camera_id not in self.pipelines:
            # Need to import here to avoid circular imports if any, 
            # though Pipeline is in pipelines.sct_pipeline
            from pipelines.sct_pipeline import Pipeline
            
            input_config = {"video_path": source}
            # Assuming self.pipeline_config is set. 
            # In a real app, we might load this from a config file or pass it in init.
            # For now, we'll assume it's set or use default if not.
            
            if not hasattr(self, 'pipeline_config'):
                # Fallback default config if not set
                self.pipeline_config = {
                     "DETECTION": "yolov8", # Example defaults
                     "POSE_ESTIMATOR": "yolov8",
                     "TRACKING": "ocsort",
                     "TRACK_MANAGER": {}
                }

            self.pipelines[camera_id] = Pipeline(
                pipeline_config=self.pipeline_config,
                input_config=input_config,
                camera_id=camera_id
            )
        return self.pipelines[camera_id]

    def stop_pipeline(self, camera_id: int):
        if camera_id in self.pipelines:
            pipeline = self.pipelines[camera_id]
            pipeline.stop()
            del self.pipelines[camera_id]
            
    def create_pipeline(self, camera_id: int, source: str):
        # Stop existing pipeline if any
        self.stop_pipeline(camera_id)
        
        from pipelines.sct_pipeline import Pipeline
        
        # Default configuration matching project factories
        pipeline_config = {
            "DETECTION": {
                "name": "yolov11",
                "yolov11": {
                    "model_path": "yolo11n.pt", 
                    "task": "detect",
                    "imgsz": 640,
                    "conf_thres": 0.25,
                    "iou_thres": 0.45,
                    "device": "cpu", # Default to CPU for safety
                    "max_det": 300,
                    "classes": [0] # Person class
                }
            },
            "TRACKING": {
                "name": "ocsort",
                "ocsort": {
                    "det_thresh": 0.4,
                    "max_age": 30,
                    "min_hits": 3,
                    "iou_threshold": 0.3,
                    "delta_t": 3,
                    "asso_func": "iou",
                    "inertia": 0.2,
                    "use_byte": False
                }
            },
            "TRACK_MANAGER": {
                "is_join_track": True,
                "GALLERY": {
                    "max_live_time": 9000,
                    "min_hits": 3,
                    "max_features": 50
                },
                "MATCHING": {
                    "threshold": 0.4
                },
                "EMBEDDING": {
                    "name": "fastreid",
                    "fastreid": "configs/osnet-ain_x1.0-ibn_512_256x192_ccdmmps.yaml"
                }
            }
        }
        
        input_config = {"video_path": source}
        pipeline = Pipeline(pipeline_config, input_config, camera_id)
        self.pipelines[camera_id] = pipeline
        return pipeline

stream_manager = StreamManager()
