import numpy as np
from ultralytics import YOLO
from multiprocessing import Process, Queue

from abc import ABC, abstractmethod

from model import TrackParams, TrackedObject, BBox
from utils import hypot, to_whxy, to_xyxy, linreg


class BaseDetector(ABC):
    def __init__(self, params: TrackParams):
        self.params = params

    @abstractmethod
    def warmup(self, n):
        pass

    @abstractmethod
    def detect(self, frame: np.array, num: int) -> tuple[TrackedObject, dict]:
        pass

    def _filter_detections(self, detections, bbox_sup):
        min_dist = None
        tracked_obj = None
        for i in range(detections.boxes.shape[0]):
            obj_cls = int(detections.boxes.cls[i].item())
            if self.params.obj_name != detections.names.get(obj_cls):
                continue

            bbox = bbox_sup(detections.boxes.xyxy[i].detach().cpu().numpy())
            if not bbox.is_valid:
                continue

            conf = detections.boxes.conf[i].item()
            curr_tracked = TrackedObject(self.params, conf, bbox)
            curr_dist = hypot(*curr_tracked.center_shift)
            if min_dist is None or curr_dist < min_dist:
                tracked_obj = curr_tracked
                min_dist = curr_dist
        return tracked_obj


class SimpleDetector(BaseDetector):
    def __init__(self, model_weights_dir: str, min_conf: float, ratiodev: float, smooth: float, params: TrackParams):
        super().__init__(params)
        self.model = YOLO(model_weights_dir)
        self.min_conf = min_conf
        self.ratiodev = ratiodev
        self.smooth = smooth
        self.prev_bbox = None

    def warmup(self, n):
        pass

    def detect(self, frame: np.array, num: int) -> tuple[TrackedObject, dict]:
        bbox_sup = lambda xyxy: BBox(xyxy, self.params.obj_size, self.ratiodev, self.smooth, self.prev_bbox)
        detections = self.model.predict(frame, conf=self.min_conf, verbose=False)[0]
        tracked_obj = self._filter_detections(detections, bbox_sup)
        self.prev_bbox = None if tracked_obj is None else tracked_obj.obj_bbox
        return tracked_obj, {}


class AsyncDetector(BaseDetector):
    def __init__(self, model_weights_dir: str, min_conf: float, ratiodev: float, smooth: float, params: TrackParams):
        super().__init__(params)
        self.model_weights_dir = model_weights_dir
        self.min_conf = min_conf
        self.ratiodev = ratiodev
        self.smooth = smooth
        self.prev_bbox = None

        self.job_queue = Queue(1)
        self.det_queue = Queue(1)
        self.async_detect = False
        work_proc = Process(target=self._async_detect)
        work_proc.daemon = True
        work_proc.start()

        self.hist_x, self.hist_y = [], []
        self.pred_x, self.pred_y = [], []
        self.coefs = None

    def _async_detect(self):
        model = YOLO(self.model_weights_dir)
        bbox_sup = lambda xyxy: BBox(xyxy, self.params.obj_size, self.ratiodev)
        while True:
            frame, num = self.job_queue.get()
            detections = model.predict(frame, conf=self.min_conf, verbose=False)[0]
            tracked_obj = self._filter_detections(detections, bbox_sup)
            self.det_queue.put((tracked_obj, num))

    def warmup(self, n):
        frame = np.random.rand(720, 1280, 3)
        for i in range(n):
            self.job_queue.put((frame, i))
            self.det_queue.get()

    def detect(self, frame: np.array, num: int) -> tuple[TrackedObject, dict]:
        if len(self.hist_x) < 2:
            assert not self.async_detect, "Async detection not available"
            self.job_queue.put((frame, num))
            result, m = self.det_queue.get()
            if result:
                self.hist_x.append(m)
                self.hist_y.append(to_whxy(result.obj_bbox.crop))
                self.prev_bbox = BBox(result.obj_bbox.xyxy, self.params.obj_size, self.ratiodev, self.smooth, self.prev_bbox)
            metrics = {"pred_count": 0,
                       "reg_b_w": 0., "reg_b_h": 0., "reg_b_x": 0., "reg_b_y": 0.,
                       "reg_k_w": 0., "reg_k_h": 0., "reg_k_x": 0., "reg_k_y": 0.
                       }
            return result, metrics

        assert len(self.hist_x) == 2, f"Actual hist size: {len(self.hist_x)}"

        if not self.async_detect:
            self.job_queue.put((frame, num))
            self.async_detect = True
            return self.__predict(num, True)

        if self.det_queue.empty():
            return self.__predict(num, False)

        tracked_m, m = self.det_queue.get()
        self.job_queue.put((frame, num))
        if not tracked_m:
            return self.__predict(num, False)

        assert m in self.pred_x, f"detected frame {m} not in pred_x ({self.pred_x})"
        self.pred_x = []
        self.pred_y = []
        self.hist_x = [np.mean(self.hist_x), m]
        self.hist_y = [np.mean(self.hist_y, axis=0), to_whxy(tracked_m.obj_bbox.crop)]
        return self.__predict(num, True)

    def __predict(self, num: int, update_coefs: bool) -> tuple[TrackedObject, dict]:
        if update_coefs:
            assert len(self.pred_x) == 0
            self.coefs = linreg(self.hist_x, self.hist_y)
        else:
            assert len(self.pred_x) > 0

        whxy_pred = self.coefs[0] + self.coefs[1] * num
        self.pred_x.append(num)
        self.pred_y.append(whxy_pred)
        self.prev_bbox = BBox(to_xyxy(whxy_pred), self.params.obj_size, self.ratiodev, self.smooth, self.prev_bbox)
        metrics = {
            "pred_count": len(self.pred_x),
            "reg_b_w": self.coefs[0, 0], "reg_b_h": self.coefs[0, 1],
            "reg_b_x": self.coefs[0, 2], "reg_b_y": self.coefs[0, 3],
            "reg_k_w": self.coefs[1, 0], "reg_k_h": self.coefs[1, 1],
            "reg_k_x": self.coefs[1, 2], "reg_k_y": self.coefs[1, 3],
        }
        return TrackedObject(self.params, -1.0, self.prev_bbox), metrics

