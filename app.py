from flask import Flask, render_template, Response
import cv2
from engine import ProctorEngine


app = Flask(__name__)


def generate_frames():
    """Generator function that captures video frames, processes them, and yields JPEG-encoded frames."""
    source = 0
    cam = cv2.VideoCapture(source)
    engine = ProctorEngine()  # Create new engine instance for each stream
    try:
        while True:
            has_frame, frame = cam.read()
            if not has_frame:
                print("No frame detected")
                break
            
            # Extract and print the shape of the captured frame
            frame_shape = frame.shape 
            h,w = frame.shape[:2]

            # Process the frame through the proctoring engine
            processed_frame = engine.detect_all(frame)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue

            # Yield the frame in multipart format for streaming
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cam.release()
        engine.release_all()


@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)