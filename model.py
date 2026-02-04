
from engine import ProctorEngine
import cv2
import sys
import time

eng = ProctorEngine()

def cleanup(cap):
    """Release all detector and camera resources"""
    global eng
    eng.release_all()
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting...")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    sys.exit(0)

print("Starting head pose detection. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    start_time = time.time()
    output = eng.detect_all(frame)
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000
    print(f"Frame processing latency: {latency_ms:.2f} ms")
    cv2.imshow('Head Pose Detection', output)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cleanup(cap)


