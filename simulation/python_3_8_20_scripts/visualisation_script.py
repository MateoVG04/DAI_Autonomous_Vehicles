import cv2
from flask import Flask, Response

from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper

app = Flask(__name__)

def generate_frames():
    while True:
        # Replace this with your simulation output
        camera_width = 600
        camera_height = 800
        shared_memory_filepath = "/dev/shm/carla_shared.dat"
        shared_memory = CarlaWrapper(filename=shared_memory_filepath, image_width=camera_width,
                                     image_height=camera_height)
        latest_image = shared_memory.read_latest_image()

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', latest_image)
        frame_bytes = buffer.tobytes()

        # Yield in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_frames_detected():
    while True:
        # Replace this with your simulation output
        camera_width = 600
        camera_height = 800
        shared_memory_filepath = "/dev/shm/carla_shared.dat"
        shared_memory = CarlaWrapper(filename=shared_memory_filepath, image_width=camera_width,
                                     image_height=camera_height)
        latest_image = shared_memory.read_latest_object_detected()

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', latest_image)
        frame_bytes = buffer.tobytes()

        # Yield in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/object_detected')
def video_feed():
    return Response(generate_frames_detected(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)