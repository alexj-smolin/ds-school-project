import numpy as np


def center_coord(xyxy: np.array) -> np.array:
    return np.round((xyxy[:2] + xyxy[-2:]) / 2).astype(int)


def hypot(d1, d2):
    return (d1 ** 2 + d2 ** 2) ** 0.5


def crop_box(box: np.array, ref_w, ref_h):
    adj_box = box.copy()
    box_w, box_h = box[-2:] - box[:2]

    if ref_w / ref_h < box_w / box_h:
        # crop box_w
        adj_w = ref_w / ref_h * box_h
        crop = (box_w - adj_w) / 2
        adj_box[0] = np.round(box[0] + crop)
        adj_box[2] = np.round(box[2] - crop)
    else:
        # crop box_h
        adj_h = ref_h / ref_w * box_w
        crop = (box_h - adj_h) / 2
        adj_box[1] = np.round(box[1] + crop)
        adj_box[3] = np.round(box[3] - crop)
    return adj_box


def to_whxy(xyxy: np.array):
    return np.hstack([np.abs(xyxy[-2:] - xyxy[:2]), (xyxy[:2] + xyxy[-2:]) / 2.])


def to_xyxy(whxy: np.array):
    return np.hstack([whxy[-2:] - whxy[:2] / 2., whxy[-2:] + whxy[:2] / 2.])


def linreg(x: list[float], y: list[np.array]) -> np.array:
    x = np.vstack(x)
    y = np.vstack(y)
    x_m = x.mean()
    y_m = y.mean(axis=0)
    k = ((x - x_m) * (y - y_m)).sum(axis=0) / ((x - x_m) ** 2).sum()
    k = np.clip(k, -10, 10)
    b = y_m - x_m * k
    return np.vstack([b, k])
