import cv2
import time
import logging
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.tts import speak
from ultralytics import YOLO
from io import BytesIO
import base64
from PIL import Image

# Logging
logging.basicConfig(level=logging.DEBUG)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use your Netlify domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model
model = YOLO("yolov8n.pt")

# Constants
FOCAL_LENGTH = 1000
KNOWN_WIDTH_CM = 7
SAFE_DISTANCE_CM = 100
SPEAK_COOLDOWN = 2
last_speak_time = 0

# Schema
class ImageData(BaseModel):
    image: str  # base64 image string

def estimate_distance(pixel_width):
    return (KNOWN_WIDTH_CM * FOCAL_LENGTH) / pixel_width if pixel_width else 0

@app.post("/detect")
async def detect(data: ImageData):
    global last_speak_time

    try:
        image_data = data.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = model(frame, conf=0.5)[0]
        detections = []

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            obj_type = model.names[int(cls)]
            pixel_width = x2 - x1
            distance = estimate_distance(pixel_width)

            status = "SAFE"
            if distance < SAFE_DISTANCE_CM:
                status = "NOT SAFE"
                if time.time() - last_speak_time > SPEAK_COOLDOWN:
                    speak(f"Warning! {obj_type} too close.")
                    last_speak_time = time.time()
            else:
                if time.time() - last_speak_time > SPEAK_COOLDOWN:
                    speak(f"{obj_type} at safe distance.")
                    last_speak_time = time.time()

            detections.append({
                "object": obj_type,
                "distance_cm": int(distance),
                "status": status
            })

        return {"detections": detections}
    except Exception as e:
        return {"error": str(e)}
