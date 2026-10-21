"""在 Triton server 容器内运行 (与 serving 同一 TensorRT 版本, 只用标准库): 为每个精度变体构建引擎,
再用 trtexec 做两层测速 + 逐层 profile。由 run_quant.sh 调用。

两层测速, 用来把 "FP8 提升不足" 拆成 GPU 内 vs GPU 外 (参数随 TRT 版本不同, 见 trtexec.measure_flags):
  compute   只有 GPU kernel 时间 = 量化在算子层的真实收益上限
  transfer  再加 H2D/D2H (FP32 输入 4.7MB + 输出 2.7MB / 张): 加上 PCIe 搬运后还剩多少
再往外一层 (gRPC / 预处理 / 动态批处理) 由 Triton + perf_analyzer 测, 见 run_quant.sh。
各变体的 ONNX 与精度参数见 variants.Variant; 一律 --profilingVerbosity=detailed (不影响性能, 逐层报告需要它)。

用法: python3 bench_trtexec.py <gpu-label> <变体...>   (环境变量 BATCHES / PROFILE_BATCH / DURATION 可调)
产物 (results/quant/): <label>-trtexec.csv, <label>-env.txt, 每个变体 / batch / 层的 .log, -profile.json, -layers.json
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import trtexec
from variants import parse_all

OUT = Path("results/quant")
CSV_HEADER = "variant,batch,mode,infer_per_s,img_per_s,gpu_compute_mean_ms,gpu_compute_p99_ms"
SHAPE = "images:{}x3x640x640"
# 可用环境变量覆盖 (空值按默认); run_quant.sh 用 docker -e 透传
BATCHES = (os.environ.get("BATCHES") or "1 8 32").split()
PROFILE_BATCH = os.environ.get("PROFILE_BATCH") or "8"
DURATION = os.environ.get("DURATION") or "15"           # 每个测点秒数


def find_trtexec():
    for cand in (shutil.which("trtexec"), "/usr/src/tensorrt/bin/trtexec", "/opt/tensorrt/bin/trtexec"):
        if cand and os.access(cand, os.X_OK):
            return cand
    sys.exit("找不到 trtexec, 请确认在 Triton server 镜像内运行")


def run(cmd, log):
    """跑一条命令, stdout+stderr 写进 log; 返回是否成功。"""
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode == 0


def tail(log, n):
    print("".join(Path(log).read_text(errors="replace").splitlines(keepends=True)[-n:]), end="")


def fail(msg, log, n):
    print(msg)
    tail(log, n)
    sys.exit(1)


def main():
    sys.stdout.reconfigure(line_buffering=True)   # docker 不带 -t 时 stdout 是管道, 否则进度会攒成大块才显示
    p = argparse.ArgumentParser()
    p.add_argument("label")
    p.add_argument("variants", nargs="+")
    a = p.parse_args()
    variants = parse_all(a.variants)

    exe = find_trtexec()
    help_text = subprocess.run([exe, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    ver = trtexec.parse_version(help_text)
    major = ver // 10000
    print(f"TensorRT v{ver} (major {major})")

    OUT.mkdir(parents=True, exist_ok=True)
    env_txt = OUT / f"{a.label}-env.txt"
    env_lines = [f"label={a.label}"]
    try:
        env_lines += subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout.splitlines()
    except FileNotFoundError:
        pass
    if "TensorRT v" in help_text:
        env_lines.append(f"TensorRT v{ver}")
    env_txt.write_text("".join(l + "\n" for l in env_lines))

    csv = OUT / f"{a.label}-trtexec.csv"
    csv.write_text(CSV_HEADER + "\n")

    env_has_build_ver = False
    for v in variants:
        plan = f"{v.model_dir}/1/model.plan"
        onnx = v.onnx_path(major)
        flags = v.build_flags(major)
        if not Path(onnx).exists():
            sys.exit(f"缺少 {onnx}, 先跑量化步骤")
        Path(plan).parent.mkdir(parents=True, exist_ok=True)
        prefix = OUT / f"{a.label}-{v.name}"

        print(f"== [{v.name}] 构建引擎 ({onnx} {' '.join(flags)}) ==")
        build_log = f"{prefix}-build.log"
        if not run([exe, f"--onnx={onnx}", f"--saveEngine={plan}", *flags,
                    f"--minShapes={SHAPE.format(1)}", f"--optShapes={SHAPE.format(8)}",
                    f"--maxShapes={SHAPE.format(32)}",
                    "--memPoolSize=workspace:4096", "--profilingVerbosity=detailed"], build_log):
            fail(f"[{v.name}] 构建失败, 见 {build_log}", build_log, 20)
        print(f"== [{v.name}] 引擎已生成 ({Path(plan).stat().st_size / 2**20:.0f}M), 开始测速 ==")
        built = trtexec.parse_build_version(Path(build_log).read_text(errors="replace"))
        if built and not env_has_build_ver:   # 所有变体同一 TensorRT, 只记一次
            with open(env_txt, "a") as f:
                f.write(f"TensorRT version: {built}\n")
            env_has_build_ver = True

        for b in BATCHES:
            for mode in ("compute", "transfer"):
                rlog = f"{prefix}-b{b}-{mode}.log"
                if not run([exe, f"--loadEngine={plan}", f"--shapes={SHAPE.format(b)}",
                            "--warmUp=2000", f"--duration={DURATION}", *trtexec.measure_flags(major, mode)], rlog):
                    fail(f"!! [{v.name} b{b} {mode}] trtexec 运行失败, 日志 {rlog} 末尾:", rlog, 25)
                try:
                    m = trtexec.parse_metrics(Path(rlog).read_text(errors="replace"), int(b))
                except ValueError as e:
                    fail(f"!! [{v.name} b{b} {mode}] {e}, 日志 {rlog} 末尾:", rlog, 25)
                row = ",".join([v.name, b, mode, m["infer_per_s"], m["img_per_s"],
                                m["gpu_compute_mean_ms"], m["gpu_compute_p99_ms"]])
                print(row)
                with open(csv, "a") as f:
                    f.write(row + "\n")

        print(f"== [{v.name}] 逐层 profile (batch {PROFILE_BATCH}) ==")
        plog = f"{prefix}-profile.log"
        if not run([exe, f"--loadEngine={plan}", f"--shapes={SHAPE.format(PROFILE_BATCH)}",
                    *trtexec.profile_flags(major), "--warmUp=1000", "--duration=5",
                    "--profilingVerbosity=detailed", "--dumpProfile",
                    f"--exportProfile={prefix}-profile.json", f"--exportLayerInfo={prefix}-layers.json"], plog):
            print(f"!! [{v.name}] 逐层 profile 失败 (不影响测速结果), 见 {plog}")
            tail(plog, 15)
    print(f"trtexec 完成 -> {csv}")


if __name__ == "__main__":
    main()
