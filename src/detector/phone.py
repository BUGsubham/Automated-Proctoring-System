from ultralytics import YOLO
from pathlib import Path
from ultralytics.engine.results import Results
import os
import cv2
import numpy as np
class PhoneDetector:

    MODEL_PATH = Path(__file__).parent.parent.parent / "assets" / "yolo26n.pt"

    def __init__(self, min_confidence : float = 0.3):
        self.min_confidence = min_confidence
        self.model = self._ensure_and_return_model()
        print(self.model.info())
    
    def _ensure_and_return_model(self): 
        if not self.MODEL_PATH.exists():
            print(f"Downloading Model .... to {self.MODEL_PATH}")
            os.makedirs(self.MODEL_PATH.parent, exist_ok=True)
            model = YOLO(str(self.MODEL_PATH.parent))
        else:
            print(f"Using model from .. {self.MODEL_PATH}")
            model = YOLO(str(self.MODEL_PATH))

        return model
    

    def detect(self, frame: np.ndarray) -> Results:

        # class 67 = mobile phone
        result = self.model.predict(frame, classes=67, verbose=False)[0]
        return result
        
    def draw_boxes(self, frame: np.ndarray, result: Results) -> np.ndarray:
        if result and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                # xyxy gives top-left (x1, y1) and bottom-right (x2, y2) coordinates
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                conf = float(boxes.conf[i])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return frame
    
    # def __enter__(self):
    #     return self

    # def __exit__():
    #     pass    

if __name__ == "__main__":
    pass
   

