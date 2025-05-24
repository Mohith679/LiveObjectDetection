# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np

app = FastAPI()

# Allow CORS from any frontend (e.g., Netlify)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your Netlify URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # or your custom model

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    results = model(img_array)[0]  # YOLOv8 output

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = model.names[cls_id]
        xyxy = box.xyxy[0].tolist()

        detections.append({
            "label": label,
            "confidence": round(confidence, 2),
            "box": {
                "xmin": int(xyxy[0]),
                "ymin": int(xyxy[1]),
                "xmax": int(xyxy[2]),
                "ymax": int(xyxy[3])
            }
        })

    return JSONResponse(content={"detections": detections})
