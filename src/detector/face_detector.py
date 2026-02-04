import cv2
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class FaceDetection:
    bbox: Tuple[int, int, int, int]
    confidence: int
    landmarks: Optional[list[Tuple[int, int]]] = None

class FaceDetector:
    
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    MODEL_PATH = Path("assets/blaze_face_short_range.tflite")

    def __init__(self, min_confidence : float = 0.4):
        
        self.min_confidence = min_confidence
        self._ensure_model()

        # Store latest detection results and frame dimensions
        self._latest_detections = []
        self._frame_dims = (0, 0)
        self._timestamp_ms = 0

        base_options = python.BaseOptions(str(self.MODEL_PATH))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_confidence,
            running_mode=vision.RunningMode.VIDEO
        )

        self.detector = vision.FaceDetector.create_from_options(options)

    def _ensure_model(self):
        "Download the model if it does not exist"
        if not self.MODEL_PATH.exists():
            self.MODEL_PATH.cwd().mkdir(parents=True, exist_ok=True)
            print(f"Downloading face detection model...")
            urllib.request.urlretrieve(self.MODEL_URL, str(self.MODEL_PATH))
            print(f"Model Downloaded to {self.MODEL_PATH}")
    
    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        
        # bgr to rgb
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # mediapipe image
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # detect for video for frame by frame processing
        self._timestamp_ms += 1
        result = self.detector.detect_for_video(mp_frame, self._timestamp_ms)

        detections = []
        for detection in result.detections:
            bbox = detection.bounding_box
            x = bbox.origin_x
            y = bbox.origin_y
            width = bbox.width
            height = bbox.height

            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)

            confidence = detection.categories[0].score if detection.categories else 0.0

            landmarks = []
            if detection.keypoints:
                for kp in detection.keypoints:
                    lx = int(kp.x * w)
                    ly = int(kp.y * h)
                    landmarks.append((lx, ly))
            
            detections.append(FaceDetection(
                bbox=(x, y, width, height),
                confidence=confidence,
                landmarks=landmarks
            ))
        return detections

    
    def count_faces(self, frame: np.ndarray) -> int:
        detections = self.detect(frame)
        return len(detections)

    def draw_detections(self, frame: np.ndarray, detections: list[FaceDetection], 
                        color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        
        """
        Draw's bounding box on the frame
        
        Args: 
            frame: BGR image numpy array
            detections: Detection result of the frame
            color: color of the bounding box
        
        Returns: 
            numpy array of frame with bounding box drawn
        
        """

        output = frame.copy()

        if len(detections) > 0:
        # if there is detections then draw
            for det in detections:
                x, y, w, h = det.bbox
                cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)

                label = f"{det.confidence:.2f}"
                cv2.putText(output, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                for pt in det.landmarks:
                    cv2.circle(output, pt, 1, color, 2)

        # if no detections then return the copy of the frame    
        return output
    
    def release(self):
        if hasattr(self, 'detector'):
            self.detector.close()
    
    def __enter__(self):
        return self

    def __exit__(self):
        self.release()




 


