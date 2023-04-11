import numpy as np
import cv2
from utils import center_coord, crop_box, hypot
from dataclasses import dataclass, asdict


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
            self, obj: tuple[str, float, float], frame_size: tuple[float, float],
            camera: tuple[float, float, float], color: tuple[int, int, int]
    ):
        """
        :param obj: object characteristics: (name, width, height), in meters
        :param frame_size: frame size (width x height)
        :param camera: camera characteristics: (focal length, sensor width, sensor height), in millimeters
        :param color: RGB color
        """
        self.obj = {"name": obj[0], "size": np.array([obj[1], obj[2]])}
        self.frame_size = np.array(frame_size, dtype=int)
        self.frame_center = center_coord(np.array([0, 0, *frame_size]))
        self.camera = {"f": camera[0], "px_size": np.array([camera[1] / frame_size[0], camera[2] / frame_size[1]])}
        self.color = tuple(color[::-1])
        self.obj_bbox = None

    def update(self, detections: list[tuple[np.array, str, float]]):
        self.obj_bbox = None
        min_dist = None
        for box, name, conf in detections:
            if self.obj["name"] != name:
                continue

            curr_obj_bbox = BBox(self, name, conf, box)
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
        cv2.putText(frame, f"  conf: {self.obj_bbox.obj_conf:.2f}", (x_ind, 2 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        center_shift_xy = self.obj_bbox.center_shift()
        cv2.putText(frame, f"  X_shift: {center_shift_xy[0]:.2f} m", (x_ind, 3 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        cv2.putText(frame, f"  Y_shift: {center_shift_xy[1]:.2f} m", (x_ind, 4 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        object_dist_xy = self.obj_bbox.object_dist()
        center_dist_xy = self.obj_bbox.center_dist()
        cv2.putText(frame, f"  X_dist: {object_dist_xy[0]:.2f} ({center_dist_xy[0]:.2f}) m", (x_ind, 5 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        cv2.putText(frame, f"  Y_dist: {object_dist_xy[1]:.2f} ({center_dist_xy[1]:.2f}) m", (x_ind, 6 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        object_dist_avg = self.obj_bbox.object_dist_avg()
        cv2.putText(frame, f"  M_dist: {object_dist_avg:.2f} m", (x_ind, 7 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        box_w, box_h = self.obj_bbox.box_crop[-2:] - self.obj_bbox.box_crop[:2]
        cv2.putText(frame, f"  crop: {box_w}, {box_h}, {box_w / box_h:.2f}", (x_ind, 8 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)
        box_w, box_h = self.obj["size"]
        cv2.putText(frame, f"  real: {box_w}, {box_h}, {box_w / box_h:.2f}", (x_ind, 9 * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sz, self.color, line_sz)

        return TrackedMetrics(
            *self.camera["px_size"], *self.obj_bbox.sim_koef, self.obj_bbox.obj_conf, *self.obj_bbox.box_size,
            *center_dist_xy, *center_shift_xy, *object_dist_xy, object_dist_avg
        )


class BBox:
    def __init__(self, frame_ctx: FrameContext, obj_name: str, obj_conf: float, obj_box: np.array):
        self.frame_center = frame_ctx.frame_center
        self.camera = frame_ctx.camera
        self.obj_size = frame_ctx.obj["size"]
        self.obj_name = obj_name
        self.obj_conf = obj_conf
        self.box_raw = obj_box.astype(int)
        self.box_crop = crop_box(self.box_raw, *frame_ctx.obj["size"])
        self.box_size = self.box_crop[-2:] - self.box_crop[:2]
        self.sim_koef = self.__sim_koef(self.camera["px_size"])

    def __sim_koef(self, px_size):
        return self.obj_size / (self.box_size * px_size)

    def is_valid(self):
        max_idx = int(self.box_crop[1] > self.box_crop[0])
        min_idx = 1 - max_idx
        valid_ratio = self.box_crop[max_idx] / self.box_crop[min_idx]
        raw_ratio = self.box_raw[max_idx] / self.box_raw[min_idx]
        return valid_ratio * 0.95 < raw_ratio < valid_ratio * 1.05

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


