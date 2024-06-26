import os
import torch
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic authentication credentials (adjust as needed)
USERNAME = "admin"
PASSWORD = "123"

# YOLO models initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_lp = YOLO("model/best_KBS_8n.pt", task='detect')
model_char = YOLO("model/best_char_8n.pt", task='detect')

model_lp.to(device)
model_char.to(device)

CHAR_THRES = 0.7


# Pydantic models for request and response payloads
class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectedObject(BaseModel):
    label: str
    bounding_box: BoundingBox
    chars: List[str]


def format_LP(chars, char_centers):
    if not chars:
        return []

    x = [c[0] for c in char_centers]
    y = [c[1] for c in char_centers]
    y_mean = np.mean(y)

    if max(y) - min(y) < 10:
        return [i for _, i in sorted(zip(x, chars))]

    sorted_chars = [i for _, i in sorted(zip(x, chars))]
    y = [i for _, i in sorted(zip(x, y))]

    first_line = [i for i in range(len(chars)) if y[i] < y_mean]
    second_line = [i for i in range(len(chars)) if y[i] >= y_mean]

    return [sorted_chars[i] for i in first_line] + ['-'] + [sorted_chars[i] for i in second_line]


# Route to serve index.html
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


# Route to process frame
@app.post("/process_frame", response_model=List[DetectedObject])
def process_frame(frame: UploadFile = File(...), username: str = USERNAME, password: str = PASSWORD):
    # Basic authentication
    if username != USERNAME or password != PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Read frame and decode
    frame_np = np.frombuffer(frame.file.read(), np.uint8)
    frame_img = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

    results = model_lp(frame_img)
    detected_objects = []

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            boxes = boxes.to(device)

            for box in boxes:
                x, y, w, h = box.xywh[0].tolist()
                cls = box.cls.item()

                label = model_lp.names[int(cls)]

                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                plate = frame_img[y1:y2, x1:x2]

                results_plate = model_char(plate)
                detected_texts = []
                char_centers = []
                detected_chars = []

                for result_plate in results_plate:
                    boxes_char = result_plate.boxes
                    if boxes_char is not None and len(boxes_char) > 0:
                        boxes_char = boxes_char.to(device)
                        for box_char in boxes_char:
                            x_char, y_char, w_char, h_char = box_char.xywh[0].tolist()
                            cls_char = box_char.cls.item()

                            label_char = model_char.names[int(cls_char)]

                            detected_chars.append(label_char)

                            x1_char = int(x_char - w_char / 2)
                            y1_char = int(y_char - h_char / 2)
                            x2_char = int(x_char + w_char / 2)
                            y2_char = int(y_char + h_char / 2)

                            center_x = (x1_char + x2_char) // 2
                            center_y = (y1_char + y2_char) // 2
                            char_centers.append((center_x, center_y))

                if detected_chars:
                    detected_texts.append(''.join(format_LP(detected_chars, char_centers)))

                obj = DetectedObject(
                    label=label,
                    bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    chars=detected_texts
                )
                detected_objects.append(obj)

    return detected_objects


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=6066)
