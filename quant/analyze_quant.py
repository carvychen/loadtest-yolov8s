"""汇总量化实验: 精度 (mAP) × 三层吞吐 (GPU 纯计算 / 含 PCIe 传输 / Triton 端到端 @SLA) × 逐层归因。

用法 (仓库根目录下): python quant/analyze_quant.py --label rtx6000-quarter-nc24 [--sla-ms 50]
读取 results/quant/<label>-*, 输出 Markdown 到 stdout (run_quant.sh 会 tee 到 <label>-summary.md)。

核心读法: 同一变体的三层加速比逐层下降, 下降发生在哪一层, 瓶颈就在哪一层。
  - GPU 纯计算加速比就不到 1.5x      -> 瓶颈在量化覆盖 / kernel (看逐层: 低精度卷积占比、reformat 占比)
  - 纯计算 >1.5x 但含传输/端到端掉下来 -> 瓶颈在 GPU 外 (FP32 I/O 搬运、gRPC、预处理), 与量化无关
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serving"))   # 复用一期 analyze.py 的 QPS@SLA 计算
from analyze import load as load_sweep, summarize as summarize_sweep
from layer_report import summarize as summarize_layers

D = "results/quant"


def fmt_x(v, base):
    return f"{v / base:.2f}×" if base else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--sla-ms", type=float, default=50.0)
    args = ap.parse_args()
    L = args.label

    te = pd.read_csv(f"{D}/{L}-trtexec.csv")
    variants = list(dict.fromkeys(te["variant"]))      # 保持运行顺序, fp16 在前
    batches = sorted(te["batch"].unique())

    def img(v, b, mode):
        r = te[(te.variant == v) & (te.batch == b) & (te["mode"] == mode)]
        return float(r["img_per_s"].iloc[0]) if len(r) else None

    maps, qps = {}, {}
    for v in variants:
        p = f"{D}/{L}-{v}-map.json"
        if os.path.exists(p):
            maps[v] = json.load(open(p))
        p = f"{D}/{L}-{v}.csv"
        if os.path.exists(p):
            qps[v] = summarize_sweep(v, load_sweep(p), args.sla_ms, None).get("qps_at_sla")

    env = open(f"{D}/{L}-env.txt").read().strip().replace("\n", " · ") if os.path.exists(f"{D}/{L}-env.txt") else ""
    print(f"# YOLOv8s 量化实验: {L}\n\n环境: {env}\n")

    # ── 1. 精度 ──
    print("## 1. 精度 (COCO val2017 评测集 4500 张, 与校准集不重叠)\n")
    print("| 变体 | mAP50-95 | ΔmAP50-95 vs FP16 | mAP50 |\n|---|---|---|---|")
    base_map = maps.get("fp16", {}).get("mAP50_95")
    for v in variants:
        m = maps.get(v)
        if not m:
            print(f"| {v} | — | — | — |"); continue
        d = f"{(m['mAP50_95'] - base_map) * 100:+.2f} pt" if base_map is not None else "—"
        print(f"| {v} | {m['mAP50_95']:.4f} | {d} | {m['mAP50']:.4f} |")

    # ── 2. 三层吞吐 ──
    print("\n## 2. 三层吞吐与加速比 (img/s; 括号内为相对 FP16)\n")
    hdr = ["变体"] + [f"GPU纯计算 b{b}" for b in batches] + [f"含传输 b{b}" for b in batches] \
        + [f"Triton QPS@p99≤{args.sla_ms:g}ms"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for v in variants:
        cells = [v]
        for mode in ("compute", "transfer"):
            for b in batches:
                x, x0 = img(v, b, mode), img("fp16", b, mode)
                cells.append(f"{x:.0f} ({fmt_x(x, x0)})" if x else "—")
        q, q0 = qps.get(v), qps.get("fp16")
        cells.append(f"{q:.1f} ({fmt_x(q, q0)})" if q else "—")
        print("| " + " | ".join(cells) + " |")

    # ── 3. 逐层归因 ──
    print("\n## 3. 逐层归因 (trtexec --dumpProfile, batch 见 PROFILE_BATCH, 默认 8)\n")
    print("| 变体 | 层数 | GPU 总耗时 ms | 低精度卷积占比 (个数) | 卷积按精度计数 | reformat 时间占比 | 未融合 Q/DQ 时间占比 |")
    print("|---|---|---|---|---|---|---|")
    layer_sums = {}
    for v in variants:
        prefix = f"{D}/{L}-{v}"
        if not (os.path.exists(prefix + "-layers.json") and os.path.exists(prefix + "-profile.json")):
            continue
        s = layer_sums[v] = summarize_layers(prefix)
        ts = s["time_share"]
        print(f"| {v} | {s['layers']} | {s['total_ms']:.2f} | {s['low_prec_conv_ratio']:.0%} | "
              f"{s['conv_count_by_prec']} | {ts.get('reformat', 0):.1%} | {ts.get('qdq_unfused', 0):.1%} |")
    for v, s in layer_sums.items():
        print(f"\n<details><summary>{v} 最耗时 10 层</summary>\n\n| 层 | 类型 | 精度 | ms |\n|---|---|---|---|")
        for name, kind, prec, ms in s["top10"]:
            print(f"| `{name}` | {kind} | {prec} | {ms} |")
        print("\n</details>")

    # ── 4. 归因结论 (自动判读, 供人工复核) ──
    print("\n## 4. 瓶颈判读\n")
    bref = 8 if 8 in batches else batches[len(batches) // 2]
    if img("fp16", bref, "compute") is None:
        print("- 没有 fp16 基线 (VARIANTS 未含 fp16), 跳过加速比判读。")
        return
    for v in variants:
        if v == "fp16":
            continue
        kc, kt = img(v, bref, "compute"), img(v, bref, "transfer")
        sc = kc / img("fp16", bref, "compute") if kc else None
        st = kt / img("fp16", bref, "transfer") if kt else None
        se = qps[v] / qps["fp16"] if qps.get(v) and qps.get("fp16") else None
        parts = [f"GPU纯计算 {sc:.2f}×" if sc else "", f"含传输 {st:.2f}×" if st else "",
                 f"Triton 端到端 {se:.2f}×" if se else ""]
        print(f"- **{v}** (b{bref}): " + " → ".join(p for p in parts if p))
        if sc and sc < 1.5:
            ls = layer_sums.get(v, {})
            print(f"  - GPU 内核层面加速已不足 1.5×: 瓶颈在量化覆盖 / kernel。"
                  f"低精度卷积占比 {ls.get('low_prec_conv_ratio', float('nan')):.0%}, "
                  f"reformat+未融合 Q/DQ 占 {ls.get('time_share', {}).get('reformat', 0) + ls.get('time_share', {}).get('qdq_unfused', 0):.1%} GPU 时间。")
        if sc and se and sc >= 1.2 and se < sc * 0.8:
            print(f"  - 端到端收益明显小于内核收益: 量化省下的 GPU 时间被 GPU 外开销 (FP32 I/O 搬运 / gRPC / 批处理排队) 稀释。")


if __name__ == "__main__":
    main()
