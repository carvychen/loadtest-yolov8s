"""yolov8s.pt -> yolov8s.onnx (动态 batch, 三卡通用)。一次性, 由 setup.sh 调用。

精度不在此固定为 FP16——交给 TensorRT 在各卡构建引擎时用 --fp16 处理,
这样 ONNX 保持通用, 三张卡各自编译出针对本硬件优化的 FP16 引擎。
"""
from ultralytics import YOLO

model = YOLO("yolov8s.pt")          # 首次自动下载权重 (~22MB)
model.export(
    format="onnx",
    imgsz=640,
    opset=17,
    dynamic=True,                    # 动态 batch 维, 供 Triton 动态批处理
    simplify=True,
    half=False,                      # 精度交给 TRT (--fp16)
)
print("exported: yolov8s.onnx")
