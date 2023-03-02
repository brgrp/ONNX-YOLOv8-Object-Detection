####################################### IMPORT #################################
import json
import numpy as np

from fastapi import FastAPI, File, status, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
import cv2
import io
import os
from yolov8 import YOLOv8, utils
from pydantic import BaseModel
from pydantic import parse_obj_as
from typing import List


class Object(BaseModel):
    uid: str
    score: int
    x1: int
    y1: int
    x2: int
    y2: int


class Objects(BaseModel):
    objects: List[Object]


# Initialize yolov8 object detector
model_path = os.getenv('MODEL_PATH', default="models/yolov8x.onnx")
print("Load model from {}".format(model_path))
yolov8_detector = YOLOv8(model_path, conf_thres=0.1, iou_thres=0.1)


def load_image_from_file(file):
    contents = file.file.read()
    nparr = np.fromstring(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


###################### FastAPI Setup #############################
app = FastAPI(
    title="ONNX FastAPI Object Detection"
)

print("###########################################")
print("    Start ONNX FastAPI Object Detection    ")
print("###########################################")


@app.on_event("startup")
def save_openapi_json():
    openapi_data = app.openapi()
    with open("../openapi.json", "w") as file:
        json.dump(openapi_data, file)


# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/info', status_code=status.HTTP_200_OK)
def info():
    return {'model_path': model_path,
            'conf_threshold': yolov8_detector.conf_threshold,
            'iou_threshold': yolov8_detector.iou_threshold}


@app.post("/annotate_image")
def annotate_image(file: UploadFile = File(...)):
    """
         This endpoint returns the annotated input image.
    """
    # Load image
    img = load_image_from_file(file)
    # Detect Objects
    _, _, _ = yolov8_detector(img)

    # Draw detections
    combined_img, results = yolov8_detector.draw_detections(img)

    # Convert OpenCV Image to jpg
    res, im_png = cv2.imencode(".jpg", combined_img)

    print("Result: {}".format(results))
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpg")


@app.post("/annotate_only")
def annotate_only(file: UploadFile = File(...)):
    """
     This endpoint returns the detections as json.
    """
    # Load image
    img = load_image_from_file(file)
    # Detect Objects
    boxes, scores, class_ids = yolov8_detector(img)

    label_scores = {}
    objects = Objects(objects=[])

    for box, score, class_id in zip(boxes, scores, class_ids):
        # Create unique label
        label = utils.class_names[class_id]
        unique_label = utils.create_unique_label(label_scores=label_scores, label=label)
        label_scores[label] = float(score)

        uid = unique_label.split('_')[-1]

        # Add label + score
        x1, y1, x2, y2 = box.astype(int)

        objects.objects.append(Object(uid=unique_label, score=int(score * 100), x1=x1, y1=y1, x2=x2, y2=y2))

    print("Objects: {}".format(objects))
    return JSONResponse(content=objects.dict(), status_code=200)
