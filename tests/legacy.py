"""golden 样本录于目录重组之前 (脚本都在根目录, 生成物在根目录)。比对时按这张表换成现在的路径, 其余逐字不变。"""

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
]


def to_new_layout(text):
    for old, new in RENAMES:
        text = text.replace(old, new)
    return text
