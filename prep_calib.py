"""生成 ModelOpt 校准数据: 500 张 COCO val2017 (与评测集不重叠) -> (500,3,640,640) float32 .npy (~2.5GB)。

TensorRT 建议 CNN 校准样本 >= 500。预处理与评测完全一致 (quant_common.preprocess)。
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from quant_common import img_path, preprocess, split_ids

out = sys.argv[1] if len(sys.argv) > 1 else "datasets/calib_500.npy"
calib, _ = split_ids()
with ThreadPoolExecutor(8) as ex:
    arr = np.stack(list(ex.map(lambda i: preprocess(img_path(i))[0], calib)))
np.save(out, arr)
print(f"校准集 -> {out}  shape={arr.shape} dtype={arr.dtype}")
