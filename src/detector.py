import numpy as np
from ultralytics import YOLO
from multiprocessing import Process, Queue

from abc import ABC, abstractmethod
from typing import Optional

from model import TrackedObject, BBox
from utils import hypot


class BaseDetector(ABC):
    def __init__(self, frame_center: np.array, camera_f: float, camera_px_size: np.array, obj_name: str, obj_size: np.array):
        self.frame_center = frame_center
        self.camera_f = camera_f
        self.camera_px_size = camera_px_size
        self.obj_name = obj_name
        self.obj_size = obj_size

    @abstractmethod
    def warmup(self, n):
        pass

    @abstractmethod
    def detect(self, frame: np.array, num: int) -> tuple[TrackedObject, dict]:
        pass

    def _handle_detections(self, detections, bbox_sup):
        min_dist = None
        tracked_obj = None
        for i in range(detections.boxes.shape[0]):
            obj_cls = int(detections.boxes.cls[i].item())
            if self.obj_name != detections.names.get(obj_cls):
                continue

            bbox = bbox_sup(detections.boxes.xyxy[i].cpu().numpy())
            if not bbox.is_valid:
                continue

            conf = detections.boxes.conf[i].item()
            curr_tracked = TrackedObject(
                self.frame_center, self.camera_f, self.camera_px_size, self.obj_name, self.obj_size, conf, bbox
            )
            curr_dist = hypot(*curr_tracked.center_shift)
            if min_dist is None or curr_dist < min_dist:
                tracked_obj = curr_tracked
                min_dist = curr_dist
        return tracked_obj


class SimpleDetector(BaseDetector):
    def __init__(self, model_weights_dir: str, min_conf: float, ratiodev: float, smooth: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = YOLO(model_weights_dir)
        self.min_conf = min_conf
        self.ratiodev = ratiodev
        self.smooth = smooth
        self.prev_bbox = None

    def warmup(self, n):
        pass

    def detect(self, frame: np.array, num: int) -> tuple[TrackedObject, dict]:
        bbox_sup = lambda xyxy: BBox(xyxy, self.obj_size, self.ratiodev, self.smooth, self.prev_bbox)
        detections = self.model.predict(frame, conf=self.min_conf, verbose=False)[0]
        tracked_obj = self._handle_detections(detections, bbox_sup)
        self.prev_bbox = None if tracked_obj is None else tracked_obj.obj_bbox
        return tracked_obj, {}


class AsyncDetector(BaseDetector):
    def __init__(self, model_weights_dir: str, min_conf: float, ratiodev: float, smooth: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_weights_dir = model_weights_dir
        self.min_conf = min_conf
        self.ratiodev = ratiodev
        self.smooth = smooth

        self.job_queue = Queue(1)
        self.det_queue = Queue(1)
        self.async_detect = False
        work_proc = Process(target=self._async_detect)
        work_proc.daemon = True
        work_proc.start()

        self.n_hist, self.N_hist = 3, 20
        self.hist_x, self.hist_y, self.hist_s = [], [], []
        self.pred_x, self.pred_y, self.pred_s = [], [], []
        self.coefs = None

    def _async_detect(self):
        model = YOLO(self.model_weights_dir)
        bbox_sup = lambda xyxy: BBox(xyxy, self.obj_size, self.ratiodev)
        while True:
            frame, num = self.job_queue.get()
            detections = model.predict(frame, conf=self.min_conf, verbose=False)[0]
            tracked_obj = super()._handle_detections(detections, bbox_sup)
            self.det_queue.put((tracked_obj, num))

    def warmup(self, n):
        frame = np.random.rand(720, 1280, 3)
        for i in range(n):
            self.job_queue.put((frame, i))
            self.det_queue.get()

    def detect(self, frame: np.array, num: int) -> tuple[TrackedObject, dict]:
        if len(self.hist_x) < self.n_hist:
            assert not self.async_detect, "Async detection not available"
            self.job_queue.put((frame, num))
            result, k = self.det_queue.get()
            if result:
                self.hist_x.append(k)
                self.hist_y.append(result.obj_bbox.xyxy)
            metrics = {"pred_count": 0,
                       "reg_b_x1": 0., "reg_b_y1": 0., "reg_b_x2": 0., "reg_b_y2": 0.,
                       "reg_k_x1": 0., "reg_k_y1": 0., "reg_k_x2": 0., "reg_k_y2": 0.
                       }
            return result, metrics

        if not self.async_detect:
            self.job_queue.put((frame, num))
            self.async_detect = True
            return self.__predict(num, True)

        if self.det_queue.empty():
            return self.__predict(num, False)

        tracked_k, k = self.det_queue.get()
        self.job_queue.put((frame, num))
        if not tracked_k:
            self.__flush_pred(None)
        else:
            self.__flush_pred((k, tracked_k.obj_bbox.xyxy))
        return self.__predict(num, True)

    def __flush_pred(self, refined: Optional[tuple[int, np.array]] = None):
        if refined:
            coefs = AsyncDetector.__estimate_coefs((self.hist_x + [refined[0]])[1:], (self.hist_y + [refined[1]])[1:])
            assert self.pred_x[0] == refined[0], f"{self.pred_x[0]} != {refined[0]}"
            r = [refined[1]]
            for m in self.pred_x[1:]:
                r.append(coefs[0] + coefs[1] * m)
            self.pred_y = r

        # TODO: в hist добавлять только detections
        self.hist_x = (self.hist_x + self.pred_x)[-self.N_hist:]
        self.hist_y = (self.hist_y + self.pred_y)[-self.N_hist:]
        self.pred_x = []
        self.pred_y = []

    def __predict(self, num: int, update_coefs: bool) -> tuple[TrackedObject, dict]:
        if update_coefs:
            assert len(self.pred_x) == 0
            self.coefs = AsyncDetector.__estimate_coefs(self.hist_x, self.hist_y)
        else:
            assert len(self.pred_x) > 0

        xyxy_pred = self.coefs[0] + self.coefs[1] * num
        self.pred_x.append(num)
        self.pred_y.append(xyxy_pred)
        bbox = BBox(xyxy_pred, self.obj_size, self.ratiodev)
        metrics = {
            "pred_count": len(self.pred_x),
            "reg_b_x1": self.coefs[0, 0], "reg_b_y1": self.coefs[0, 1],
            "reg_b_x2": self.coefs[0, 2], "reg_b_y2": self.coefs[0, 3],
            "reg_k_x1": self.coefs[1, 0], "reg_k_y1": self.coefs[1, 1],
            "reg_k_x2": self.coefs[1, 2], "reg_k_y2": self.coefs[1, 3],
        }
        return TrackedObject(
            self.frame_center, self.camera_f, self.camera_px_size, self.obj_name, self.obj_size, -1.0, bbox
        ), metrics

    @staticmethod
    def __estimate_coefs(x, y) -> np.array:
        x = np.vstack(x)
        y = np.vstack(y)
        x_m = x.mean()
        y_m = y.mean(axis=0)
        k = ((x - x_m) * (y - y_m)).sum(axis=0) / ((x - x_m) ** 2).sum()
        b = y_m - x_m * k
        return np.vstack([b, k])

