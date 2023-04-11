import os
import argparse

import numpy as np
import torch
from torchvision.io import VideoReader, write_video
import cv2
import time
from ultralytics import YOLO
from dotenv import load_dotenv
import mlflow

from model import FrameContext


load_dotenv()

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJ_DIR, 'models')
INPUT_DIR = os.path.join(PROJ_DIR, 'input')
OUTPUT_DIR = os.path.join(PROJ_DIR, 'output')

ap = argparse.ArgumentParser()
ap.add_argument(
    "-w", "--weights", required=True, help="path to yolo-trained weights")
ap.add_argument(
    "-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
ap.add_argument(
    "-s", "--sample", required=True, help="video sample")
ap.add_argument(
    "-o", "--objects", type=str, default="", help="comma-separated list of objects to detect")
args = ap.parse_args()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

model = YOLO(os.path.join(MODEL_DIR, args.weights))
objects = set([v.strip() for v in args.objects.split(',') if v])

print("[INFO] starting video stream")
file_path = os.path.join(INPUT_DIR, args.sample)
reader = VideoReader(file_path, "video")

fps = reader.get_metadata()["video"]["fps"][0]
cap = cv2.VideoCapture(file_path)
frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# obj = FrameContext(("airplane", 14.5, 6), frame_size, (12, 4.92, 2.77), (255, 0, 0))
obj = FrameContext(("person", 0.6, 1.7), frame_size, (4, 7.38, 4.15), (255, 0, 0))
# obj = FrameContext(("car", 2, 1.4), frame_size, (2.8, 5.47, 3.07), (255, 0, 0))

video_array = []
win_name = "camera_frame"
cv2.namedWindow(win_name)
prev_time = time.time()
prev_pts = 0
wait = 1
with mlflow.start_run(run_name=obj.obj["name"]):
    for frame in reader:
        data = cv2.cvtColor(frame["data"].moveaxis(0, 2).numpy(), cv2.COLOR_RGB2BGR)

        curr_time = time.time()
        prev_time = max(curr_time, prev_time + (frame["pts"] - prev_pts))
        prev_pts = frame["pts"]
        time.sleep(prev_time - curr_time)

        result = model.predict(data, conf=args.confidence, verbose=False)[0]

        detections = []
        for i in range(result.boxes.shape[0]):
            obj_cls = int(result.boxes.cls[i].item())
            obj_name = result.names.get(obj_cls)
            if objects and obj_name not in objects:
                continue
            detections.append((result.boxes.xyxy[i].cpu().numpy(), obj_name, result.boxes.conf[i].item()))

        obj.update(detections)
        metrics = obj.draw(data)
        if metrics is not None:
            mlflow.log_metrics(metrics.dict())

        cv2.imshow(win_name, data)
        key = cv2.waitKey(wait) & 0xFF
        if key == ord("q"): break
        if key == ord("w"): wait = 0 if wait > 0 else 1
        if key == ord("e"): wait = 200

        video_array.append(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

    mlflow.log_metric("frames", len(video_array))

    out_filename = os.path.join(OUTPUT_DIR, args.sample)
    write_video(out_filename, torch.tensor(np.stack(video_array)), fps)
    mlflow.log_artifact(out_filename, "arts123")

    cv2.destroyWindow(win_name)
    print("[INFO] video stream closed")

