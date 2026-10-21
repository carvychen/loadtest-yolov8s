"""生成 ModelOpt 校准数据: 500 张 COCO val2017 (与评测集不重叠) -> (500,C,H,W) float32 .npy, 路径见 models.py。

TensorRT 建议 CNN 校准样本 >= 500。预处理与评测完全一致:
  yolov8s  quant_common.preprocess (letterbox 640, /255)          -> (500,3,640,640) ~2.5GB
  swin_t   swin.preprocess (timm: resize + center crop 224, 归一化) -> (500,3,224,224) ~0.3GB
COCO 图片对分类模型只是 "自然图片", 校准只统计激活范围, 不需要 ImageNet 标签。已存在则跳过。
用法 (仓库根目录下): python quant/prep_calib.py [--model M]
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import models
from quant_common import img_path, split_ids

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolov8s")
m = models.get(ap.parse_args().model)
if os.path.exists(m.calib):
    print(f"已存在 {m.calib}, 跳过")
    raise SystemExit
if m.name.startswith("swin_t"):   # swin_t / swin_t_tanh 共用一份 (预处理与 GELU 写法无关)
    import swin
    tf = swin.transform(swin.build_model())
    prep = lambda i: swin.preprocess(tf, img_path(i))
else:
    from quant_common import preprocess
    prep = lambda i: preprocess(img_path(i))[0]
calib, _ = split_ids()
with ThreadPoolExecutor(8) as ex:
    arr = np.stack(list(ex.map(prep, calib)))
np.save(m.calib, arr)
print(f"校准集 -> {m.calib}  shape={arr.shape} dtype={arr.dtype}")
