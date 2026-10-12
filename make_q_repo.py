"""为每个精度变体生成 Triton 模型目录 model_repository_q/yolov8s_<variant>/config.pbtxt。

以 model_repository/yolov8s/config.pbtxt 为模板 (动态批处理等参数与一期完全一致, 保证同口径),
只改模型名和 I/O 数据类型: I/O 类型从实际 ONNX 读取 (stronglyTyped 引擎的 I/O 类型 = ONNX 声明类型),
避免量化工具改了 I/O 精度而 Triton 配置没跟上导致加载失败。
用法: python make_q_repo.py fp16 fp8 int8
"""
import os
import re
import sys

import onnx

TEMPLATE = "model_repository/yolov8s/config.pbtxt"
ONNX_TO_TRITON = {onnx.TensorProto.FLOAT: "TYPE_FP32", onnx.TensorProto.FLOAT16: "TYPE_FP16"}


def onnx_for(variant):
    return "yolov8s.onnx" if variant == "fp16" else f"yolov8s.{variant}.onnx"


def io_types(path):
    g = onnx.load(path, load_external_data=False).graph
    return (ONNX_TO_TRITON[g.input[0].type.tensor_type.elem_type],
            ONNX_TO_TRITON[g.output[0].type.tensor_type.elem_type])


with open(TEMPLATE) as f:
    tpl = f.read()

for v in sys.argv[1:]:
    t_in, t_out = io_types(onnx_for(v))
    cfg = tpl.replace('name: "yolov8s"', f'name: "yolov8s_{v}"', 1)
    types = iter([t_in, t_out])                      # 模板里先 input 后 output
    cfg = re.sub(r"data_type: TYPE_\w+", lambda m: f"data_type: {next(types)}", cfg)
    d = f"model_repository_q/yolov8s_{v}"
    os.makedirs(f"{d}/1", exist_ok=True)
    with open(f"{d}/config.pbtxt", "w") as f:
        f.write(cfg)
    print(f"{d}: input {t_in}, output {t_out}")
