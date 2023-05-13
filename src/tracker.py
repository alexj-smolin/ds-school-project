import os
import numpy as np
import cv2
from torchvision.io import VideoReader, write_video
import torch
import mlflow
import time

from utils import center_coord
from model import TrackedObject
from detector import Detector

# from multiprocessing import Process, Queue


class FrameEnricher:
    def __init__(self, obj_size: np.array, frame_center: np.array):
        self.obj_size = obj_size
        self.frame_center = frame_center
        self.color = (0, 0, 255)

    def draw(self, frame, num, tracked_obj: TrackedObject) -> dict:
        k = (self.frame_center[1] * 2 + 1000 - 1) // 1000
        k5, k15 = k * 5, k * 15
        font_sz, line_sz = 0.6 * k, k * 2
        x_ind, row_h = k5, 20 * k

        cv2.line(frame, self.frame_center + [0, -k15], self.frame_center + [0, k15], self.color, line_sz)
        cv2.line(frame, self.frame_center + [-k15, 0], self.frame_center + [k15, 0], self.color, line_sz)
        cv2.circle(frame, self.frame_center, k5, self.color, -1)
        cv2.putText(frame, f"frame: {num}", (x_ind, 1 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        if tracked_obj is None:
            return {}
        metrics = tracked_obj.metrics()
        obj_box = tracked_obj.obj_box

        # bounding box
        cv2.rectangle(frame, obj_box.orig[:2].astype(int), obj_box.orig[-2:].astype(int), (50, 50, 50), line_sz)
        cv2.rectangle(frame, obj_box.crop[:2], obj_box.crop[-2:], self.color, line_sz)
        y_text = obj_box.crop[1] + (-k5 if obj_box.crop[1] > k5 else k15)
        cv2.putText(frame, tracked_obj.obj_name, (obj_box.crop[0], y_text), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        # object metrics
        cv2.putText(frame, f"{tracked_obj.obj_name}:", (x_ind, 2 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        cv2.putText(frame, f"  X_shift: {metrics['center_shift_x']:.2f} m", (x_ind, 3 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        cv2.putText(frame, f"  Y_shift: {metrics['center_shift_y']:.2f} m", (x_ind, 4 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        cv2.putText(frame, f"  Distance: {metrics['object_dist_avg']:.2f} m", (x_ind, 5 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        box_w, box_h = obj_box.crop_size
        cv2.putText(frame, f"  bbox: {box_w}/{box_h}={box_w / box_h:.2f}", (x_ind, 7 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        box_w, box_h = self.obj_size
        cv2.putText(frame, f"  real: {box_w}/{box_h}={box_w / box_h:.2f}", (x_ind, 8 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        cv2.putText(frame, f"  conf: {metrics['object_conf']:.2f}", (x_ind, 9 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        return metrics


class Tracker:
    def __init__(self, params: dict, basedir: str, mlflow_tracking_uri: str):
        self.params = params
        self.basedir = basedir
        self.progress_bar = 20
        self.sample_path = os.path.join(basedir, "samples", params["sample"])

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # self.work_in_queue = Queue(1)
        # self.work_out_queue = Queue(1)
        # work_proc = Process(target=self.predict, args=(self.work_in_queue, self.work_out_queue, "work_queue"))
        # work_proc.daemon = True
        # work_proc.start()

    def __metadata(self):
        cap = cv2.VideoCapture(self.sample_path)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return frame_size, frames_cnt, fps

    # def predict(self, in_queue, out_queue, msg):
    #     while True:
    #         if not in_queue.empty():
    #             data, i = in_queue.get()
    #             result = self.model.predict(data, conf=self.min_conf, verbose=False)[0]
    #             out_queue.put(result)
    #             # print(msg, i)

    def run(self):
        frame_size, frames_cnt, fps = self.__metadata()
        obj_name = self.params["oname"]
        obj_size = np.array([self.params["owidth"], self.params["oheight"]])
        frame_center = center_coord(np.array([0, 0, *frame_size]))
        camera_f = self.params["cfocal"]
        camera_px_size = np.array([self.params["cwidth"] / frame_size[0], self.params["cheight"] / frame_size[1]])

        frame_enricher = FrameEnricher(obj_size, frame_center)
        weights_dir = os.path.join(self.basedir, "models", self.params["model"])
        detector = Detector(
            weights_dir, self.params["conf"], frame_center, camera_f, camera_px_size,
            obj_name, obj_size, self.params["ratiodev"], self.params["smooth"]
        )

        reader = VideoReader(self.sample_path)
        video_array = []
        sample_name = self.params["sample"]
        progress_step = (frames_cnt + self.progress_bar - 1) // self.progress_bar

        mlflow.set_experiment(sample_name)
        with mlflow.start_run(run_name=obj_name):
            mlflow.log_params(self.params)
            mlflow.log_metrics({"src_width": frame_size[0], "src_height": frame_size[1], "src_fps": fps})
            start_time = time.time()
            stage0 = start_time
            all_metrics = []
            for k, frame in enumerate(reader):
                data = cv2.cvtColor(frame["data"].moveaxis(0, 2).numpy(), cv2.COLOR_RGB2BGR)
                stage_read = time.time()

                tracked_obj, det_metrics = detector.detect(data)
                obj_metrics = frame_enricher.draw(data, k + 1, tracked_obj)
                stage_proc = time.time()

                frame_sleep = start_time + frame["pts"] - stage_proc
                time.sleep(max(0., frame_sleep))

                video_array.append(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

                print("\r", end="")
                if progress_step <= 0:
                    print("progress:", f"unknown ({(k + 1) / fps:.1f} sec)", end="")
                else:
                    print("progress:", "*" * ((k + progress_step - 1) // progress_step), f"[{int((k + 1) / frames_cnt * 100)}%]", end="")

                stage_loop = time.time()
                all_metrics.append(det_metrics | obj_metrics | {
                    "frame_read": stage_read - stage0, "frame_proc": stage_proc - stage_read,
                    "frame_sleep": frame_sleep, "frame_loop": stage_loop - stage0
                })
                stage0 = stage_loop
            for j, m in enumerate(all_metrics):
                mlflow.log_metrics(m, j)

            mlflow.log_metric("src_frames", len(video_array))

            print("\n[INFO] saving video ...")
            out_filename = os.path.join(self.basedir, "tmp", sample_name)
            write_video(out_filename, torch.tensor(np.stack(video_array)), fps)
            mlflow.log_artifact(out_filename, "output")
            os.remove(out_filename)
            print(f"[INFO] video saved")

