import os 
import logging 
import time
from pathlib import Path

import cv2

from modules.detector.factory import DetectorFactory
from modules.tracker_2D.factory import TrackerFactory
from modules.track_manager.single_track_manager import SingleTrackManager
from utils.vis import draw_tracks, setup_video_writer
from utils.io import WebcamVideoStream

logger = logging.getLogger(__name__)

class SCTCameraPipeline:
    def __init__(self, config: dict, video_path: Path | str):
        self.detector = DetectorFactory(config["DETECTION"]).get_detector()
        self.tracker = TrackerFactory(config["TRACKING"]).get_tracker()
        self.track_manager = SingleTrackManager(config["TRACK_MANAGER"])
        
        # Determine video name for output
        if isinstance(video_path, (str, Path)) and os.path.exists(str(video_path)):
             self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        else:
             self.video_name = f"camera_stream"

        self.cap = WebcamVideoStream(src=video_path).start()
        
    def run(self):
        current_frame = 0
        # Use underlying stream for setup as WebcamVideoStream wrapper doesn't expose get()
        writer = setup_video_writer(self.cap.stream, output_path=f'{self.video_name}.mp4')
        
        logger.info("Starting pipeline processing...")
        try:
            while True:
                frame = self.cap.read()
                
                if frame is None:
                    if self.cap.stopped:
                        break
                    # Wait briefly if frame not ready
                    time.sleep(0.01)
                    continue

                current_frame += 1
                
                bboxes = self.detector.detect(frame)
                frame_info = {
                        "cam_id": "1",
                        "frame_id": current_frame,
                        'frame': frame
                    }

                tracks = self.tracker.update(bboxes, frame_info)
                live_tracks = self.track_manager.process(tracks, frame_info)

                annotated_frame = draw_tracks(frame, live_tracks, frame_info)
                writer.write(annotated_frame)

                # Visualize
                cv2.imshow("Camera Stream", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if current_frame % 100 == 0:
                    logger.info("Frame: %d", current_frame)

        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user.")
        except Exception as e:
            logger.exception(f"Error in pipeline: {e}")
        finally:
            self.cap.stop()
            writer.release()
            cv2.destroyAllWindows()
            logger.info("Resources released.")