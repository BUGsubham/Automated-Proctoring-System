from src.detector import *
import numpy as np


class ProctorEngine:

    def __init__(self):
        self.phone_detector = PhoneDetector()
        self.face_detector = FaceDetector()
        self.pose_dectector = HeadPoseDetector()
        pass
    
    def detect_face(self, frame: np.ndarray) -> np.ndarray:
        detections = self.face_detector.detect(frame)
        output = self.face_detector.draw_detections(frame, detections)
        return output        
    
    def detect_pose(self, frame: np.ndarray) -> np.ndarray:
        pose = self.pose_dectector.estimate_pose(frame)
        output = self.pose_dectector.draw_pose_info(frame, pose)
        return output

    def detect_phone(self, frame: np.ndarray) -> np.ndarray:
        results = self.phone_detector.detect(frame)
        output = self.phone_detector.draw_boxes(frame, results)
        return output
    
    def detect_all(self, frame: np.ndarray) -> np.ndarray:
        output = self.detect_face(frame)
        output = self.detect_pose(output)  
        output = self.detect_phone(output)
        return output
    
    def release_all(self):
        self.face_detector.release()
        self.pose_dectector.release()
