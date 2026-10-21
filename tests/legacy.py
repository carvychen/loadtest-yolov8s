"""golden 样本录于目录重组与多模型之前 (脚本都在根目录, 生成物在根目录)。比对时按这张表换成现在的路径, 其余逐字不变。"""

RENAMES = [
    ("--onnx_path=yolov8s.", "--onnx_path=artifacts/yolov8s."),
    ("--output_path=yolov8s.", "--output_path=artifacts/yolov8s."),
    ("--onnx=yolov8s.", "--onnx=artifacts/yolov8s."),
    ("model_repository_q", "artifacts/model_repository_q"),
    ("python fetch_coco.py", "python quant/fetch_coco.py"),
    ("python prep_calib.py", "python quant/prep_calib.py"),
    ("python eval_map.py", "python quant/eval_map.py"),
    ("python analyze_quant.py", "python quant/analyze_quant.py"),
    ("bash run_sweep.sh", "bash serving/run_sweep.sh"),
    # 多模型之后: AutoCast 样本按模型命名
    ("datasets/autocast_calib.npz", "datasets/autocast_calib_yolov8s.npz"),
    ("yolo-quant:modelopt-0.47.0", "lt-quant:modelopt-0.47.0-timm"),   # 镜像加装 timm 后改名, 旧镜像不会被误用
]


def to_new_layout(text):
    for old, new in RENAMES:
        text = text.replace(old, new)
    return text
