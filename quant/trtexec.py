"""trtexec 的版本差异与日志解析 (纯计算, 只用标准库; 在 Triton server 容器内由 bench_trtexec.py 调用)。"""
import re


def parse_metrics(log, batch):
    """测速日志 -> CSV 指标 (保留日志原文数字)。
    坑: 日志末尾的指标说明段也有 "GPU Compute Time: the GPU latency...", 只认 "GPU Compute Time: min" 那行;
    "Throughput: X qps" 是每秒推理次数, img/s = X × batch。"""
    tput = re.findall(r"Throughput: ([0-9.]+)", log)
    if not tput:
        raise ValueError("解析不到 Throughput")
    gline = ([l for l in log.splitlines() if "GPU Compute Time: min" in l] or [""])[-1]
    mean = re.search(r"mean = ([0-9.]+)", gline)
    p99 = re.search(r"percentile\(99%\) = ([0-9.]+)", gline)
    return {"infer_per_s": tput[-1], "img_per_s": f"{float(tput[-1]) * batch:.1f}",
            "gpu_compute_mean_ms": mean.group(1) if mean else "",
            "gpu_compute_p99_ms": p99.group(1) if p99 else ""}


def parse_version(text):
    """`trtexec --help` 输出 -> 版本号整数, 如 101302 (10.13.2) / 110300 (11.3.0)。找不到时按 TRT 10 处理。"""
    m = re.search(r"TensorRT v([0-9]+)", text)
    return int(m.group(1)) if m else 100000


def measure_flags(major, mode):
    """两层测速的计时参数。compute = 只有 GPU kernel (量化在算子层的收益上限); transfer = 再加 PCIe H2D/D2H。
    TRT 11 默认已开 --useCudaGraph / --useSpinWait / --noDataTransfers (旧参数已弃用), 含传输改用 --includeDataTransfers。"""
    if major >= 11:
        return {"compute": [], "transfer": ["--includeDataTransfers"]}[mode]
    return {"compute": ["--useCudaGraph", "--useSpinWait", "--noDataTransfers"],
            "transfer": ["--useCudaGraph", "--useSpinWait"]}[mode]


def profile_flags(major):
    """逐层 profile 的计时参数: 逐层计时与整体计时分开跑 (TRT 10 用 --separateProfileRun; TRT 11 默认如此)。
    已知: TRT 10 逐层计时不开 CUDA Graph, TRT 11 默认开着 CUDA Graph, 两版本逐层耗时口径不完全一致 (只作同版本内相对比较)。"""
    return [] if major >= 11 else ["--noDataTransfers", "--separateProfileRun"]


def parse_build_version(log):
    """构建日志里的 "TensorRT version: 10.13.2" -> "10.13.2"; 找不到返回 None。"""
    m = re.search(r"TensorRT version: ([0-9.]+)", log)
    return m.group(1) if m else None
