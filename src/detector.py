import numpy as np
from ultralytics import YOLO

from model import TrackedObject, BBox, SmoothedBBox
from utils import hypot


class Detector:
    def __init__(
            self, model_weights_dir: str, min_conf: float, frame_center: np.array,
            camera_f: float, camera_px_size: np.array, obj_name: str, obj_size: np.array,
            ratiodev: float, smooth: float
    ):
        self.model = YOLO(model_weights_dir)
        self.min_conf = min_conf
        self.frame_center = frame_center
        self.camera_f = camera_f
        self.camera_px_size = camera_px_size
        self.obj_name = obj_name
        self.obj_size = obj_size
        self.ratiodev = ratiodev
        self.smooth = smooth
        self.prev_bbox = None

    def detect(self, frame: np.array) -> tuple[TrackedObject, dict]:
        predict = self.model.predict(frame, conf=self.min_conf, verbose=False)[0]
        min_dist = None
        tracked_obj = None
        for i in range(predict.boxes.shape[0]):
            obj_cls = int(predict.boxes.cls[i].item())
            if self.obj_name != predict.names.get(obj_cls):
                continue

            box_coord = predict.boxes.xyxy[i].cpu().numpy()
            bbox = (
                SmoothedBBox(box_coord, self.obj_size, self.ratiodev, self.smooth, self.prev_bbox)
                if self.smooth < 1. else
                BBox(box_coord, self.obj_size, self.ratiodev)
            )
            if not bbox.is_valid:
                continue

            conf = predict.boxes.conf[i].item()
            curr_tracked = TrackedObject(
                self.frame_center, self.camera_f, self.camera_px_size, self.obj_name, conf, self.obj_size, bbox
            )
            curr_dist = hypot(*curr_tracked.center_shift)
            if min_dist is None or curr_dist < min_dist:
                tracked_obj = curr_tracked
                self.prev_bbox = bbox
                min_dist = curr_dist
        if tracked_obj is None:
            self.prev_bbox = None
        return tracked_obj, {}

