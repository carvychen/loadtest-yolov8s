"""汇总三卡 perf_analyzer 结果 -> QPS@SLA 与每美元性能 (QPS/$)。

用法:
    python analyze.py --sla-ms 50
读取 results/*.csv (文件名即 gpu 标签), 输出对比表 + results/curve.png。
"""
import argparse
import glob
import os

import pandas as pd

# 每小时价格 USD/hr = Azure 门户月价 / 730。填你各 SKU 的实际月价。
PRICES_PER_HOUR = {
    "rtx6000-quarter": 762.21 / 730,   # NC36lds_xl_RTXPRO6000BSE_v6; 省钱可换 NC24lds=692.92/730
    "t4":  None,                        # TODO: Standard_NC16as_T4_v3  月价/730
    "a10": None,                        # TODO: Standard_NV36ads_A10_v5 月价/730
}


def load(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"Inferences/Second": "qps", "p99 latency": "p99_us"})
    df["p99_ms"] = df["p99_us"] / 1000.0
    return df.sort_values("Concurrency")


def summarize(label, df, sla_ms):
    ok = df[df["p99_ms"] <= sla_ms]
    price = PRICES_PER_HOUR.get(label)
    if ok.empty:
        return {"label": label, "qps_at_sla": 0.0,
                "note": f"无并发点满足 p99<={sla_ms}ms (最低 p99={df['p99_ms'].min():.1f}ms)"}
    best = ok.loc[ok["qps"].idxmax()]
    row = {
        "label": label,
        "qps_at_sla": round(best["qps"], 1),
        "p99_ms": round(best["p99_ms"], 1),
        "concurrency": int(best["Concurrency"]),
        "usd_per_hr": round(price, 3) if price else None,
        "qps_per_usd_hr": round(best["qps"] / price, 1) if price else None,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sla-ms", type=float, default=50.0, help="p99 延迟 SLA (ms)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--plot", default="results/curve.png")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results, "*.csv")))
    if not paths:
        print(f"{args.results}/ 下没有 csv, 先在各机器跑 run.sh"); return

    rows, curves = [], {}
    for path in paths:
        label = os.path.splitext(os.path.basename(path))[0]
        df = load(path)
        curves[label] = df
        rows.append(summarize(label, df, args.sla_ms))

    out = pd.DataFrame(rows)
    print(f"\n=== YOLOv8s 三卡对比 (p99 SLA ≤ {args.sla_ms} ms) ===")
    print(out.to_string(index=False))

    if "qps_per_usd_hr" in out.columns and out["qps_per_usd_hr"].notna().any():
        best = out.loc[out["qps_per_usd_hr"].idxmax()]
        print(f"\n每美元性能最优: {best['label']}  →  {best['qps_per_usd_hr']} QPS per $/hr")
        print("提示: 完整 RTX PRO 6000 ≈ 4× quarter, 比较整卡方案时 quarter 的吞吐 ×4、价格 ×4。")
    else:
        print("\n(在 analyze.py 顶部填 t4 / a10 的价格后可得每美元性能排名)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        for label, df in curves.items():
            plt.plot(df["qps"], df["p99_ms"], marker="o", label=label)
        plt.axhline(args.sla_ms, ls="--", c="gray", label=f"SLA {args.sla_ms}ms")
        plt.xlabel("Throughput (QPS)")
        plt.ylabel("p99 latency (ms)")
        plt.title("YOLOv8s: throughput vs p99 latency")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(args.plot, dpi=130, bbox_inches="tight")
        print(f"曲线已保存: {args.plot}")
    except ImportError:
        print("(pip install matplotlib 可出延迟-吞吐曲线图)")


if __name__ == "__main__":
    main()
