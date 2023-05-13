import numpy as np

from utils import center_coord, crop_box, hypot


class BBox:
    def __init__(self, box: np.array, obj_size: np.array, ratiodev: float):
        self.orig = box.astype(float)
        self.crop = crop_box(box.astype(int), *obj_size)
        self.crop_size = self.crop[-2:] - self.crop[:2]
        self.__add_is_valid(ratiodev)

    def box_metrics(self):
        return {"box_x1": self.crop[0], "box_y1": self.crop[1], "box_x2": self.crop[2], "box_y2": self.crop[3]}

    def __add_is_valid(self, ratiodev):
        max_idx = int(self.crop_size[1] > self.crop_size[0])
        min_idx = 1 - max_idx
        valid_ratio = self.crop_size[max_idx] / self.crop_size[min_idx]
        raw_size = self.orig[-2:] - self.orig[:2]
        raw_ratio = raw_size[max_idx] / raw_size[min_idx]
        self.is_valid = valid_ratio * (1 - ratiodev) < raw_ratio < valid_ratio * (1 + ratiodev)


class SmoothedBBox(BBox):
    def __init__(self, box: np.array, obj_size: np.array, ratiodev: float, alpha: float, prev_bbox = None):
        super().__init__(SmoothedBBox.__smooth_value(box, prev_bbox, alpha), obj_size, ratiodev)
        self.raw = box.astype(int)

    def box_metrics(self):
        return super().box_metrics() | {
            "box_raw_x1": self.raw[0], "box_raw_y1": self.raw[1],
            "box_raw_x2": self.raw[2], "box_raw_y2": self.raw[3]
        }

    @staticmethod
    def __smooth_value(box: np.array, prev_bbox: BBox, alpha: float) -> np.array:
        if prev_bbox is None:
            return box
        if not isinstance(prev_bbox, SmoothedBBox):
            raise Exception("Expected object of type SmoothedBBox")
        return prev_bbox.orig + alpha * (prev_bbox.raw - prev_bbox.orig)


class SmoothedBBox2(BBox):
    def __init__(self, box: np.array, obj_size: np.array, ratiodev: float, alpha: float, beta: float, prev_bbox = None):
        super().__init__(SmoothedBBox2.__smooth_value(box, prev_bbox, alpha), obj_size, ratiodev)
        self.trend = SmoothedBBox2.__trend_value(box, self.orig, prev_bbox, beta)
        self.raw = box.astype(int)

    def box_metrics(self):
        return super().box_metrics() | {
            "box_raw_x1": self.raw[0], "box_raw_y1": self.raw[1],
            "box_raw_x2": self.raw[2], "box_raw_y2": self.raw[3]
        }

    @staticmethod
    def __smooth_value(box: np.array, prev_bbox: BBox, alpha: float) -> np.array:
        if prev_bbox is None:
            return box
        if not isinstance(prev_bbox, SmoothedBBox2):
            raise Exception("Expected object of type SmoothedBBox2")
        if prev_bbox.trend is None:
            s = prev_bbox.orig + (box - prev_bbox.raw)
        else:
            s = prev_bbox.orig + prev_bbox.trend
        return s + alpha * (prev_bbox.raw - s)

    @staticmethod
    def __trend_value(box: np.array, curr_orig: np.array, prev_bbox: BBox, beta: float) -> np.array:
        if prev_bbox is None:
            return None
        if not isinstance(prev_bbox, SmoothedBBox2):
            raise Exception("Expected object of type SmoothedBBox2")
        t = prev_bbox.trend
        if t is None:
            t = box - prev_bbox.raw
        return t + beta * (curr_orig - prev_bbox.orig - t)


class TrackedObject:
    def __init__(
            self, frame_center: np.array, camera_f: float, camera_px_size: np.array,
            obj_name: str, obj_conf: float, obj_size: np.array, obj_box: BBox
    ):
        self.frame_center = frame_center
        self.camera_f = camera_f
        self.camera_px_size = camera_px_size
        self.obj_name = obj_name
        self.obj_conf = obj_conf
        self.obj_size = obj_size
        self.obj_box = obj_box
        self.sim_koef = self.__sim_koef(camera_px_size)
        self.__add_distances()
        self.__basic_metrics = [
            "px_size_x", "px_size_y", "sim_koef_x", "sim_koef_y", "object_conf", "center_dist_x", "center_dist_y",
            "center_shift_x", "center_shift_y", "object_dist_x", "object_dist_y", "object_dist_avg"
        ]

    def __sim_koef(self, px_size):
        return self.obj_size / (self.obj_box.crop_size * px_size)

    def __add_distances(self):
        # axis distances
        center_shift_xy = (center_coord(self.obj_box.crop) - self.frame_center) * self.camera_px_size
        self.center_dist = self.camera_f * self.sim_koef
        self.center_shift = center_shift_xy * self.sim_koef
        self.object_dist = hypot(center_shift_xy, self.camera_f) * self.sim_koef

        # average object distance
        px_size_avg = np.mean(self.camera_px_size)
        sim_koef_avg = np.mean(self.__sim_koef(px_size_avg))
        center_shift_avg = hypot(*((center_coord(self.obj_box.crop) - self.frame_center) * px_size_avg))
        self.object_dist_avg = hypot(center_shift_avg, self.camera_f) * sim_koef_avg

    def metrics(self) -> dict:
        result = dict(zip(self.__basic_metrics, [
            *self.camera_px_size, *self.sim_koef, self.obj_conf, *self.center_dist,
            *self.center_shift, *self.object_dist, self.object_dist_avg
        ]))
        return result | self.obj_box.box_metrics()

