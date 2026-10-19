import cv2, numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()
model = YOLO(".\model\ppe-4_results (train-4)\weights\best.pt")                 # loaded once at startup
print(model.names)                      # check your class spellings in the terminal

MIN_CONF = {"person": 0.4, "helmet": 0.5, "vest": 0.5, "gloves": 0.5}
ITEMS = ("helmet", "vest", "gloves")

def center_in(b, p):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return p[0] <= cx <= p[2] and p[1] <= cy <= p[3]

@app.post("/detect")
def detect(file: UploadFile = File(...)):
    frame = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"people": [], "items": []}

    r = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
    persons, items = [], []
    for box, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        name = model.names[int(k)].lower()
        if name not in MIN_CONF or c < MIN_CONF[name]:
            continue                    # ignore unused classes and weak detections
        d = {"name": name, "box": [round(v) for v in box], "conf": round(c, 2)}
        (persons if name == "person" else items).append(d)

    for p in persons:                   # attach each item to the person it sits inside
        found = {i["name"] for i in items if center_in(i["box"], p["box"])}
        for k in ITEMS:
            p[k] = k in found
    return {"people": persons, "items": items}

app.mount("/", StaticFiles(directory="static", html=True), name="static")import cv2, numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()
model = YOLO("best.pt")                 # loaded once at startup
print(model.names)                      # check your class spellings in the terminal

MIN_CONF = {"person": 0.4, "helmet": 0.5, "vest": 0.5, "gloves": 0.5}
ITEMS = ("helmet", "vest", "gloves")

def center_in(b, p):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return p[0] <= cx <= p[2] and p[1] <= cy <= p[3]

@app.post("/detect")
def detect(file: UploadFile = File(...)):
    frame = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"people": [], "items": []}

    r = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
    persons, items = [], []
    for box, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        name = model.names[int(k)].lower()
        if name not in MIN_CONF or c < MIN_CONF[name]:
            continue                    # ignore unused classes and weak detections
        d = {"name": name, "box": [round(v) for v in box], "conf": round(c, 2)}
        (persons if name == "person" else items).append(d)

    for p in persons:                   # attach each item to the person it sits inside
        found = {i["name"] for i in items if center_in(i["box"], p["box"])}
        for k in ITEMS:
            p[k] = k in found
    return {"people": persons, "items": items}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
