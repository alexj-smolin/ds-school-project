import numpy as np

from dataclasses import dataclass

from utils import center_coord, crop_box, hypot


@dataclass
class TrackParams:
    frame_center: np.array
    camera_f: float
    camera_px_size: np.array
    obj_name: str
    obj_size: np.array


class BBox:
    def __init__(self, xyxy: np.array, obj_size: np.array, ratiodev: float, alpha: float = 1.0, prev_bbox: "BBox" = None):
        self.raw = xyxy.astype(float)
        self.xyxy = BBox.__smooth_value(prev_bbox, alpha, self.raw)
        self.crop = crop_box(self.xyxy.astype(int), *obj_size)
        self.crop_size = self.crop[-2:] - self.crop[:2]
        self.__add_is_valid(ratiodev)

    def __add_is_valid(self, ratiodev: float):
        max_idx = int(self.crop_size[1] > self.crop_size[0])
        min_idx = 1 - max_idx
        valid_ratio = self.crop_size[max_idx] / self.crop_size[min_idx]
        raw_size = self.xyxy[-2:] - self.xyxy[:2]
        raw_ratio = raw_size[max_idx] / raw_size[min_idx]
        self.is_valid = valid_ratio * (1 - ratiodev) < raw_ratio < valid_ratio * (1 + ratiodev)

    def box_metrics(self) -> dict:
        return {
            "box_crop_x1": self.crop[0], "box_crop_y1": self.crop[1],
            "box_crop_x2": self.crop[2], "box_crop_y2": self.crop[3],
            "box_raw_x1": self.raw[0], "box_raw_y1": self.raw[1],
            "box_raw_x2": self.raw[2], "box_raw_y2": self.raw[3]
        }

    @staticmethod
    def __smooth_value(prev_bbox: "BBox", alpha: float, curr_xyxy: np.array) -> np.array:
        if alpha >= 1.0:
            return curr_xyxy
        if prev_bbox is None:
            assert curr_xyxy is not None, "No current xyxy"
            return curr_xyxy
        return prev_bbox.xyxy + alpha * (prev_bbox.raw - prev_bbox.xyxy)


class TrackedObject:
    def __init__(
            self, params: TrackParams, obj_conf: float, obj_bbox: BBox
    ):
        self.params = params
        self.obj_conf = obj_conf
        self.obj_bbox = obj_bbox
        self.sim_coef = self.__sim_coef(params.camera_px_size)
        self.__add_distances()
        self.__basic_metrics = [
            "px_size_x", "px_size_y", "sim_coef_x", "sim_coef_y", "object_conf", "center_dist_x", "center_dist_y",
            "center_shift_x", "center_shift_y", "object_dist_x", "object_dist_y", "object_dist_avg"
        ]

    def __sim_coef(self, px_size):
        return self.params.obj_size / (self.obj_bbox.crop_size * px_size)

    def __add_distances(self):
        # axis distances
        center_shift_xy = (center_coord(self.obj_bbox.crop) - self.params.frame_center) * self.params.camera_px_size
        self.center_dist = self.params.camera_f * self.sim_coef
        self.center_shift = center_shift_xy * self.sim_coef
        self.object_dist = hypot(center_shift_xy, self.params.camera_f) * self.sim_coef

        # average object distance
        px_size_avg = np.mean(self.params.camera_px_size)
        sim_coef_avg = np.mean(self.__sim_coef(px_size_avg))
        center_shift_avg = hypot(*((center_coord(self.obj_bbox.crop) - self.params.frame_center) * px_size_avg))
        self.object_dist_avg = hypot(center_shift_avg, self.params.camera_f) * sim_coef_avg

    def metrics(self) -> dict:
        result = dict(zip(self.__basic_metrics, [
            *self.params.camera_px_size, *self.sim_coef, self.obj_conf, *self.center_dist,
            *self.center_shift, *self.object_dist, self.object_dist_avg
        ]))
        return result | self.obj_bbox.box_metrics()

