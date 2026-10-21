"""在量化容器内运行: 为每个精度变体准备 ONNX 与 Triton 模型目录。由 run_quant.sh 调用。
  1. FP16 基线 ONNX: ModelOpt AutoCast -> artifacts/<模型>.fp16.onnx (TensorRT 11 用; TRT 10 直接用原 FP32 ONNX + --fp16)
  2. 量化: ModelOpt ONNX PTQ (或 torch 路径) -> artifacts/<模型>.<quant_name>.onnx (与 GPU 无关, 已存在则跳过)
  3. Triton 配置: artifacts/model_repository_q/<模型>_<变体>/config.pbtxt
各变体的量化命令 / ONNX 文件见 variants.Variant, 各模型的文件名 / 校准集见 models.py。
用法 (仓库根目录下): python quant/prepare_variants.py [--model M] <变体...>   (环境变量 FP8_CALIB / INT8_CALIB 可覆盖校准方法)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

from variants import parse_all

LOGS = Path("results/quant")


def tee(cmd, log):
    """跑命令, 输出同时打到终端和 log (相当于 `cmd 2>&1 | tee log`); 返回是否成功。"""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p, \
            open(log, "w") as f:
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
    return p.returncode == 0


def quantize(variants, runner=tee):
    """生成 AutoCast FP16 ONNX 与各变体的 Q/DQ ONNX。量化命令退出码非 0 或没生成输出文件都算失败。"""
    LOGS.mkdir(parents=True, exist_ok=True)
    m = variants[0].model
    # 已知: 不管 TRT 版本、是否有 fp16 变体, 都会生成 AutoCast FP16 ONNX (只多花一次 CPU 转换)
    if not Path(m.fp16_onnx).exists():
        if not Path(m.autocast_calib).exists():   # AutoCast 判断数值范围用, 取校准集前 16 张即可
            import numpy as np   # 同上, 只在真正生成时需要
            a = np.load(m.calib, mmap_mode="r")
            np.savez(m.autocast_calib, **{m.input_name: np.ascontiguousarray(a[:16])})
        log = LOGS / f"autocast-{m.name}.log"
        # --keep_io_types: I/O 保持 FP32, 与其它变体同一份 Triton 配置; 数值 >512 的节点 (如 YOLO 检测头解码) AutoCast 自动留 FP32
        ok = runner([sys.executable, "-m", "modelopt.onnx.autocast", f"--onnx_path={m.source_onnx}",
                     f"--output_path={m.fp16_onnx}", "--low_precision_type=fp16", "--keep_io_types",
                     f"--calibration_data={m.autocast_calib}"], log)
        if not ok or not Path(m.fp16_onnx).exists():
            sys.exit(f"AutoCast 失败, 见 {log}")

    methods = {"fp8": os.environ.get("FP8_CALIB"), "int8": os.environ.get("INT8_CALIB")}
    for v in variants:
        if v.quant_name is None:
            continue
        onnx = v.onnx_path()
        if Path(onnx).exists():
            print(f"  已存在 {onnx}")
            continue
        cmd = v.quantize_cmd(calibration_method=methods[v.precision])
        print(f"  [{v.quant_name}] {' '.join(cmd)}")
        log = LOGS / f"quant-{Path(onnx).stem}.log"
        ok = runner([sys.executable, *cmd], log)
        if not ok or not Path(onnx).exists():
            sys.exit(f"[{v.quant_name}] 量化失败, 见 {log}")


def read_io_types(onnx_path):
    import onnx   # 用到时才导入: 本地测试不装 onnx 也能跑
    to_triton = {onnx.TensorProto.FLOAT: "TYPE_FP32", onnx.TensorProto.FLOAT16: "TYPE_FP16"}
    g = onnx.load(onnx_path, load_external_data=False).graph
    return (to_triton[g.input[0].type.tensor_type.elem_type],
            to_triton[g.output[0].type.tensor_type.elem_type])


def write_configs(variants, io_types=read_io_types):
    """以一期 config.pbtxt 为模板 (动态批处理等参数与一期一致, 保证同口径), 只改模型名和 I/O 数据类型。
    I/O 类型从实际 ONNX 读取 (stronglyTyped 引擎的 I/O 类型 = ONNX 声明类型), 避免量化工具改了 I/O 精度而配置没跟上。"""
    for v in variants:
        tpl = Path(v.model.triton_template).read_text()
        # 此时还不知道 TRT 版本; fp16 的两份 ONNX (原 FP32 / AutoCast --keep_io_types) I/O 一样都是 FP32, 取哪份都行
        t_in, t_out = io_types(v.onnx_path(trt_major=10))
        cfg = tpl.replace(f'name: "{v.model.name}"', f'name: "{v.triton_name}"', 1)
        types = iter([t_in, t_out])                      # 模板里先 input 后 output
        cfg = re.sub(r"data_type: TYPE_\w+", lambda m: f"data_type: {next(types)}", cfg)
        Path(v.model_dir, "1").mkdir(parents=True, exist_ok=True)
        Path(v.model_dir, "config.pbtxt").write_text(cfg)
        print(f"{v.model_dir}: input {t_in}, output {t_out}")


if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(line_buffering=True)   # docker 不带 -t 时 stdout 是管道, 否则 ModelOpt 进度会攒成大块才显示
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s")
    ap.add_argument("variants", nargs="+")
    a = ap.parse_args()
    variants = parse_all(a.variants, a.model)
    quantize(variants)
    write_configs(variants)
