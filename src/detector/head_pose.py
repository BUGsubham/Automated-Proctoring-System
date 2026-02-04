import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from urllib.request import urlretrieve
from typing import Tuple

@dataclass
class HeadPose:
    "Angle in degrees of rotation of Head"
    yaw: float
    pitch: float
    roll: float
    direction: str 

class HeadPoseDetector:

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = Path("assets/face_landmarker.task")

    LANDMARK_INDICES = {
        'nose_tip': 1,
        'chin': 152,
        'left_eye_outer': 263,
        'right_eye_outer': 33,
        'left_mouth_corner': 287,
        'right_mouth_corner': 57
    }
    
    def __init__(self, min_detection_confidence: float = 0.4, 
                 min_tracking_confidence: float = 0.4) -> None:
        
        self.min_detect_confidence = min_detection_confidence
        self.min_track_confidence = min_tracking_confidence
        self._timestamp_ms = 0 
        self._frame_dims = (0, 0)
        self._ensure_model()

        base_options = python.BaseOptions(model_asset_path=str(self.MODEL_PATH))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # 3D model points for pose estimation (from observations)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye outer corner
            (225.0, 170.0, -135.0),    # Right eye outer corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype=np.float64)

    def _ensure_model(self):
        
        if not self.MODEL_PATH.exists():
            self.MODEL_PATH.cwd().mkdir(parents=True, exist_ok=True)
            print("Downloading Model ...")
            urlretrieve(self.MODEL_URL, str(self.MODEL_PATH))
            print(f"Model downloaded to {str(self.MODEL_PATH)}")
        else:
            print("Using Pre-Downloaded Model")

    def estimate_pose(self, frame: np.ndarray) -> HeadPose:

        h, w, _ = frame.shape
        self._frame_dims = (h, w)
        
        focal_length = w                    # general approximation
        center = (w / 2, h / 2)             # intersection of pricipal axis and the image
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))      # no distortion

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self._timestamp_ms += 1
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
        
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None
        
        landmarks = result.face_landmarks[0]

        image_points = np.array([
            self._get_landmark_point(landmarks, self.LANDMARK_INDICES['nose_tip'], w, h),
            self._get_landmark_point(landmarks, self.LANDMARK_INDICES['chin'], w, h),
            self._get_landmark_point(landmarks, self.LANDMARK_INDICES['left_eye_outer'], w, h),
            self._get_landmark_point(landmarks, self.LANDMARK_INDICES['right_eye_outer'], w, h),
            self._get_landmark_point(landmarks, self.LANDMARK_INDICES['left_mouth_corner'], w, h),
            self._get_landmark_point(landmarks, self.LANDMARK_INDICES['right_mouth_corner'], w, h)
        ], dtype=np.float64)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
        
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        # Compose projection matrix and decompose to get Euler angles
        proj_matrix = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]
        
        # Determine direction
        direction = self._get_direction(yaw, pitch)
        
        return HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            direction=direction
        )
    
    def _get_landmark_point(self, landmarks, idx: int, w: int, h: int) -> Tuple[float, float]:
        """Get 2D point from landmark index."""
        landmark = landmarks[idx]
        return (landmark.x * w, landmark.y * h)
        

    def _get_direction(self, yaw: float, pitch: float) -> str:
        
        if abs(yaw) < 10 and abs(pitch) < 10:
            return "CENTER"
        elif yaw < -15:
            return "RIGHT"
        elif yaw > 15:
            return "LEFT"
        elif pitch < -10:
            return "DOWN"
        elif pitch > 10:
            return "UP"
        else:
            return "CENTER"
        
    def is_looking_away(self, frame: np.ndarray, yaw_threshold: float = 20.0, pitch_threshold: float = 20.0) -> Tuple[bool, str|None]:

        pose = self.estimate_pose(frame)

        if pose is None:
            return False, None
        
        if abs(pose.yaw) > yaw_threshold:
            return True, "Right" if pose.yaw > 0 else "Left"
        
        if abs(pose.pitch) > yaw_threshold:
            return True, "Up" if pose.yaw > 0 else "Down"
        
        return False, None
    
    def draw_pose_info(self, frame: np.ndarray, pose: HeadPose, position: Tuple[int, int] = (30, 30)) -> np.ndarray:

        output = frame.copy()
        x, y = position

        if pose is None:
            cv2.putText(output, "No pose Detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            return output

        lineInfo = [
            f"Yaw: {pose.yaw:1f}",
            f"pitch: {pose.pitch:1f}",
            f"roll: {pose.roll:1f}",
            f"Direction: {pose.direction}"
        ]

        for i, line in enumerate(lineInfo):
            cv2.putText(output, line, (x, y+ i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0.0, 255.0, 0.0), 2)

        return output
    

    def release(self):
        if hasattr(self, "landmarker"):
            self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        # print(exc_type)
        # print(exc_tb)



if __name__ == "__main__":
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()
    
    print("Starting head pose detection. Press 'q' to quit.")
    
    # Initialize detector
    with HeadPoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5) as detector:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Estimate pose
            pose = detector.estimate_pose(frame)
            
            # Draw pose information
            if pose is not None:
                output_frame = detector.draw_pose_info(frame, pose)
                
                # Draw direction indicator
                h, w = frame.shape[:2]
                cv2.putText(output_frame, f"Looking: {pose.direction}", 
                           (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 255), 2)
            else:
                output_frame = frame.copy()
                cv2.putText(output_frame, "No face detected", 
                           (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Head Pose Detection', output_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting...")