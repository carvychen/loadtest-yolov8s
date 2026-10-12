"""量化实验共用: COCO val2017 切分、letterbox 预处理、类别映射。

校准 (prep_calib.py) 与评测 (eval_map.py) 必须走同一份预处理, 否则校准出的
scale 与实际推理时的激活分布对不上, 量化精度会被错误地拉低。
"""
import json
import os

import cv2
import numpy as np

COCO_DIR = os.environ.get("COCO_DIR", "datasets/coco")
IMG_DIR = os.path.join(COCO_DIR, "val2017")
ANN = os.path.join(COCO_DIR, "annotations", "instances_val2017.json")
IMGSZ = 640

# Ultralytics 80 类索引 -> COCO 官方 category_id (1..90, 有空号)
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
assert len(COCO80_TO_91) == 80


def split_ids():
    """确定性切分: 按 image_id 排序, 每 10 张取 1 张做校准 (500), 其余 4500 张做评测。
    校准集与评测集不重叠, 避免量化后 mAP 被高估。"""
    with open(ANN) as f:
        ids = sorted(im["id"] for im in json.load(f)["images"])
    calib = ids[::10]
    held = set(calib)
    return calib, [i for i in ids if i not in held]


def img_path(image_id):
    return os.path.join(IMG_DIR, f"{image_id:012d}.jpg")


def letterbox(img):
    """与 Ultralytics LetterBox(auto=False, center=True) 一致: 等比缩放 + 灰边 114 居中填充到 640x640。"""
    h, w = img.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = (IMGSZ - nh) / 2, (IMGSZ - nw) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    return img, r, (left, top)


def preprocess(path):
    """返回 (CHW float32 /255 RGB 张量, 缩放比, (pad_x, pad_y), 原图 (h, w))。"""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    lb, r, pad = letterbox(bgr)
    x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
    return x, r, pad, bgr.shape[:2]
