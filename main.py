import cv2
import time
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from utils.tts import speak

# ========== Logging Setup ==========
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='debug.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(console)

# ========== Constants ==========
FOCAL_LENGTH = 1000
KNOWN_WIDTH_CM = 7
SAFE_DISTANCE_CM = 100
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30
SPEAK_COOLDOWN = 2

# ========== FastAPI App ==========
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Load YOLO Model ==========
try:
    model = YOLO("yolov8n.pt")
    model.to('cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    exit(1)

last_speak_time = 0

# ========== Helper Functions ==========
def estimate_distance(pixel_width):
    if pixel_width == 0:
        return 0
    return (KNOWN_WIDTH_CM * FOCAL_LENGTH) / pixel_width

def gen_frames():
    global last_speak_time
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.5, imgsz=640, verbose=False)

        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            obj_type = model.names[int(cls)]
            pixel_width = x2 - x1
            distance = estimate_distance(pixel_width)

            if distance < SAFE_DISTANCE_CM:
                if time.time() - last_speak_time > SPEAK_COOLDOWN:
                    speak(f"Warning! {obj_type} too close.")
                    last_speak_time = time.time()
                status = "NOT SAFE"
                color = (0, 0, 255)
            else:
                if time.time() - last_speak_time > SPEAK_COOLDOWN:
                    speak(f"{obj_type} at safe distance.")
                    last_speak_time = time.time()
                status = "SAFE"
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{obj_type}: {int(distance)} cm ({status})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ========== API Routes ==========

@app.get("/")
def root():
    return {"message": "Live Object Detection API is working!"}

@app.get("/video")
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
