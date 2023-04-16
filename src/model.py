import os
import numpy as np
import cv2
from dataclasses import dataclass, asdict
from torchvision.io import VideoReader, write_video
import torch
import mlflow
from ultralytics import YOLO

from utils import center_coord, crop_box, hypot


@dataclass
class TrackedMetrics:
    px_size_x: float
    px_size_y: float
    sim_koef_x: float
    sim_koef_y: float
    object_conf: float
    box_size_x: int
    box_size_y: int
    center_dist_x: float
    center_dist_y: float
    center_shift_x: float
    center_shift_y: float
    object_dist_x: float
    object_dist_y: float
    object_dist_avg: float

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


class FrameContext:
    def __init__(
            self, obj: tuple[str, float, float], camera: tuple[float, float, float],
            frame_size: tuple[float, float], ratiodev: float
    ):
        """
        :param obj: object characteristics: (name, width, height), in meters
        :param camera: camera characteristics: (focal length, sensor width, sensor height), in millimeters
        :param frame_size: frame size (width x height), in pixels
        :param ratiodev: maximum box ratio deviance
        :param color: RGB color
        """
        self.obj = {"name": obj[0], "size": np.array([obj[1], obj[2]])}
        self.frame_size = np.array(frame_size, dtype=int)
        self.frame_center = center_coord(np.array([0, 0, *frame_size]))
        self.camera = {"f": camera[0], "px_size": np.array([camera[1] / frame_size[0], camera[2] / frame_size[1]])}
        self.obj_bbox = None
        self.ratiodev = ratiodev
        self.color = (0, 0, 255)

    def update(self, detections: list[tuple[np.array, str, float]]):
        self.obj_bbox = None
        min_dist = None
        for box, name, conf in detections:
            if self.obj["name"] != name:
                continue

            curr_obj_bbox = BBox(self, name, conf, self.ratiodev, box)
            if not curr_obj_bbox.is_valid():
                continue

            curr_dist = hypot(*curr_obj_bbox.center_shift())
            if min_dist is None or curr_dist < min_dist:
                self.obj_bbox = curr_obj_bbox
                min_dist = curr_dist

    def draw(self, frame):
        k = (self.frame_size[1] + 1000 - 1) // 1000
        k5, k15 = k * 5, k * 15
        font_sz, line_sz = 0.6 * k, k * 2
        x_ind, row_h = k5, 20 * k

        cv2.line(frame, self.frame_center + [0, -k15], self.frame_center + [0, k15], self.color, line_sz)
        cv2.line(frame, self.frame_center + [-k15, 0], self.frame_center + [k15, 0], self.color, line_sz)
        cv2.circle(frame, self.frame_center, k5, self.color, -1)

        if self.obj_bbox is None:
            return None

        # bounding box with label
        cv2.rectangle(frame, self.obj_bbox.box_raw[:2], self.obj_bbox.box_raw[-2:], (50, 50, 50), line_sz)
        cv2.rectangle(frame, self.obj_bbox.box_crop[:2], self.obj_bbox.box_crop[-2:], self.color, line_sz)
        y_text = self.obj_bbox.box_crop[1] + (-k5 if self.obj_bbox.box_crop[1] > k5 else k15)
        cv2.putText(frame, self.obj_bbox.obj_name, (self.obj_bbox.box_crop[0], y_text), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        # object stat
        cv2.putText(frame, f"{self.obj_bbox.obj_name}:", (x_ind, 1 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        center_shift_xy = self.obj_bbox.center_shift()
        cv2.putText(frame, f"  X_shift: {center_shift_xy[0]:.2f} m", (x_ind, 2 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        cv2.putText(frame, f"  Y_shift: {center_shift_xy[1]:.2f} m", (x_ind, 3 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        object_dist_avg = self.obj_bbox.object_dist_avg()
        cv2.putText(frame, f"  Distance: {object_dist_avg:.2f} m", (x_ind, 4 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        box_w, box_h = self.obj_bbox.box_crop[-2:] - self.obj_bbox.box_crop[:2]
        cv2.putText(frame, f"  bbox: {box_w}/{box_h}={box_w / box_h:.2f}", (x_ind, 6 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        box_w, box_h = self.obj["size"]
        cv2.putText(frame, f"  real: {box_w}/{box_h}={box_w / box_h:.2f}", (x_ind, 7 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        cv2.putText(frame, f"  conf: {self.obj_bbox.obj_conf:.2f}", (x_ind, 8 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        center_dist_xy = self.obj_bbox.center_dist()
        object_dist_xy = self.obj_bbox.object_dist()
        return TrackedMetrics(
            *self.camera["px_size"], *self.obj_bbox.sim_koef, self.obj_bbox.obj_conf, *self.obj_bbox.box_size,
            *center_dist_xy, *center_shift_xy, *object_dist_xy, object_dist_avg
        )


class BBox:
    def __init__(self, frame_ctx: FrameContext, obj_name: str, obj_conf: float, ratiodev: float, obj_box: np.array):
        self.frame_center = frame_ctx.frame_center
        self.camera = frame_ctx.camera
        self.obj_size = frame_ctx.obj["size"]
        self.obj_name = obj_name
        self.obj_conf = obj_conf
        self.box_raw = obj_box.astype(int)
        self.box_crop = crop_box(self.box_raw, *frame_ctx.obj["size"])
        self.box_size = self.box_crop[-2:] - self.box_crop[:2]
        self.ratiodev = ratiodev
        self.sim_koef = self.__sim_koef(self.camera["px_size"])

    def __sim_koef(self, px_size):
        return self.obj_size / (self.box_size * px_size)

    def is_valid(self):
        max_idx = int(self.box_size[1] > self.box_size[0])
        min_idx = 1 - max_idx
        valid_ratio = self.box_size[max_idx] / self.box_size[min_idx]
        raw_size = self.box_raw[-2:] - self.box_raw[:2]
        raw_ratio = raw_size[max_idx] / raw_size[min_idx]
        return valid_ratio * (1 - self.ratiodev) < raw_ratio < valid_ratio * (1 + self.ratiodev)

    def center_shift(self):
        center_shift_xy = (center_coord(self.box_crop) - self.frame_center) * self.camera["px_size"]
        return center_shift_xy * self.sim_koef

    def object_dist(self):
        obj_shift_xy = np.abs(center_coord(self.box_crop) - self.frame_center) * self.camera["px_size"]
        return hypot(obj_shift_xy, self.camera["f"]) * self.sim_koef

    def object_dist_avg(self):
        px_size_avg = np.mean(self.camera["px_size"])
        sim_koef_avg = np.mean(self.__sim_koef(px_size_avg))
        center_shift_xy = np.abs(center_coord(self.box_crop) - self.frame_center) * px_size_avg
        center_shift_avg = hypot(*center_shift_xy)
        return hypot(center_shift_avg, self.camera["f"]) * sim_koef_avg

    def center_dist(self):
        return self.camera["f"] * self.sim_koef


class Tracker:
    def __init__(self, params: dict, basedir: str, mlflow_tracking_uri: str):
        self.params = params
        self.basedir = basedir
        self.progress_bar = 20
        self.model = YOLO(os.path.join(basedir, "models", params["model"]))
        self.min_conf = params["conf"]
        self.filepath = os.path.join(basedir, "samples", params["sample"])
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    def __metadata(self):
        cap = cv2.VideoCapture(self.filepath)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return frame_size, frames_cnt, fps

    def run(self):
        frame_size, frames_cnt, fps = self.__metadata()
        frame_ctx = FrameContext(
            (self.params["oname"], self.params["owidth"], self.params["oheight"]),
            (self.params["cfocal"], self.params["cwidth"], self.params["cheight"]),
            frame_size, self.params["ratiodev"]
        )

        reader = VideoReader(self.filepath)
        video_array = []
        sample_name = self.params["sample"]
        progress_step = (frames_cnt + self.progress_bar - 1) // self.progress_bar

        mlflow.set_experiment(sample_name)
        with mlflow.start_run(run_name=frame_ctx.obj["name"]):
            mlflow.log_params(self.params)
            for k, frame in enumerate(reader):
                data = cv2.cvtColor(frame["data"].moveaxis(0, 2).numpy(), cv2.COLOR_RGB2BGR)
                predict = self.model.predict(data, conf=self.min_conf, verbose=False)[0]

                detections = []
                for i in range(predict.boxes.shape[0]):
                    obj_cls = int(predict.boxes.cls[i].item())
                    obj_name = predict.names.get(obj_cls)
                    detections.append((predict.boxes.xyxy[i].cpu().numpy(), obj_name, predict.boxes.conf[i].item()))

                frame_ctx.update(detections)
                metrics = frame_ctx.draw(data)
                if metrics is not None:
                    mlflow.log_metrics(metrics.dict(), k)

                video_array.append(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

                print("\r", end="")
                if progress_step <= 0:
                    print("progress:", f"unknown ({(k + 1) / fps:.1f} sec)", end="")
                else:
                    print("progress:", "*" * ((k + progress_step - 1) // progress_step), f"[{int((k + 1) / frames_cnt * 100)}%]", end="")

            mlflow.log_metric("frames", len(video_array))

            print("\n[INFO] saving video ...")
            out_filename = os.path.join(self.basedir, "tmp", sample_name)
            write_video(out_filename, torch.tensor(np.stack(video_array)), fps)
            mlflow.log_artifact(out_filename, "output")
            os.remove(out_filename)
            print(f"[INFO] video saved")

