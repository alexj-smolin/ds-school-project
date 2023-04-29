import os
import numpy as np
import cv2
from torchvision.io import VideoReader, write_video
import torch
import mlflow
from ultralytics import YOLO

from utils import center_coord, hypot
from model import TrackedObject, BBox, SmoothedBBox


class FrameContext:
    def __init__(
            self, obj: tuple[str, float, float], camera: tuple[float, float, float],
            frame_size: tuple[float, float], ratiodev: float, smooth: float
    ):
        """
        :param obj: object characteristics: (name, width, height), in meters
        :param camera: camera characteristics: (focal length, sensor width, sensor height), in millimeters
        :param frame_size: frame size (width x height), in pixels
        :param ratiodev: maximum box ratio deviance
        """
        self.obj = {"name": obj[0], "size": np.array([obj[1], obj[2]])}
        self.frame_size = np.array(frame_size, dtype=int)
        self.frame_center = center_coord(np.array([0, 0, *frame_size]))
        self.camera = {"f": camera[0], "px_size": np.array([camera[1] / frame_size[0], camera[2] / frame_size[1]])}
        self.tracked_obj = None
        self.ratiodev = ratiodev
        self.smooth = smooth
        self.color = (0, 0, 255)

    def update(self, detections: list[tuple[np.array, str, float]]):
        prev_box = None if self.tracked_obj is None else self.tracked_obj.obj_box
        self.tracked_obj = None
        min_dist = None
        for box, name, conf in detections:
            if self.obj["name"] != name:
                continue

            obj_box = (
                SmoothedBBox(box, self.obj["size"], self.ratiodev, self.smooth, prev_box)
                if self.smooth < 1. else
                BBox(box, self.obj["size"], self.ratiodev)
            )
            if not obj_box.is_valid:
                continue

            curr_tracked = TrackedObject(
                self.frame_center, self.camera["f"], self.camera["px_size"], name, conf, self.obj["size"], obj_box
            )
            curr_dist = hypot(*curr_tracked.center_shift)
            if min_dist is None or curr_dist < min_dist:
                self.tracked_obj = curr_tracked
                min_dist = curr_dist

    def draw(self, frame, num):
        k = (self.frame_size[1] + 1000 - 1) // 1000
        k5, k15 = k * 5, k * 15
        font_sz, line_sz = 0.6 * k, k * 2
        x_ind, row_h = k5, 20 * k

        cv2.line(frame, self.frame_center + [0, -k15], self.frame_center + [0, k15], self.color, line_sz)
        cv2.line(frame, self.frame_center + [-k15, 0], self.frame_center + [k15, 0], self.color, line_sz)
        cv2.circle(frame, self.frame_center, k5, self.color, -1)
        cv2.putText(frame, f"frame: {num}", (x_ind, 1 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        if self.tracked_obj is None:
            return None
        metrics = self.tracked_obj.metrics()
        obj_box = self.tracked_obj.obj_box

        # bounding box
        cv2.rectangle(frame, obj_box.orig[:2].astype(int), obj_box.orig[-2:].astype(int), (50, 50, 50), line_sz)
        cv2.rectangle(frame, obj_box.crop[:2], obj_box.crop[-2:], self.color, line_sz)
        y_text = obj_box.crop[1] + (-k5 if obj_box.crop[1] > k5 else k15)
        cv2.putText(frame, self.tracked_obj.obj_name, (obj_box.crop[0], y_text), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        # object metrics
        cv2.putText(frame, f"{self.tracked_obj.obj_name}:", (x_ind, 2 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        cv2.putText(frame, f"  X_shift: {metrics['center_shift_x']:.2f} m", (x_ind, 3 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        cv2.putText(frame, f"  Y_shift: {metrics['center_shift_y']:.2f} m", (x_ind, 4 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        cv2.putText(frame, f"  Distance: {metrics['object_dist_avg']:.2f} m", (x_ind, 5 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        box_w, box_h = obj_box.crop_size
        cv2.putText(frame, f"  bbox: {box_w}/{box_h}={box_w / box_h:.2f}", (x_ind, 7 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        box_w, box_h = self.obj["size"]
        cv2.putText(frame, f"  real: {box_w}/{box_h}={box_w / box_h:.2f}", (x_ind, 8 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        cv2.putText(frame, f"  conf: {metrics['object_conf']:.2f}", (x_ind, 9 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        return metrics


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
            frame_size, self.params["ratiodev"], self.params["smooth"]
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
                metrics = frame_ctx.draw(data, k + 1)
                if metrics is not None:
                    mlflow.log_metrics(metrics, k)

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

