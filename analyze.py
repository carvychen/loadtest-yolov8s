"""汇总三卡 perf_analyzer 结果 -> QPS@SLA 与每美元性能 (QPS/$)。

用法:
    python analyze.py --sla-ms 50
读取 results/*.csv (文件名即 gpu 标签), 输出对比表 + results/curve.png。
"""
import argparse
import glob
import os

import pandas as pd

# 每小时价格 USD/hr —— Azure 公开零售价 (Consumption / 按需 / Linux / 非 Spot)。
# 来源: Azure Retail Prices API, 查询于 2026-09; 若有 EA/预留折扣价请自行替换。
PRICES_PER_HOUR = {
    "rtx6000-quarter-nc24": 1.13,   # Standard_NC24lds_xl_RTXPRO6000BSE_v6  24 vCPU/72GB  West US 2 (Spot≈0.21) — 实测主机
    "t4":                   1.276,  # Standard_NC16as_T4_v3                 Sweden Central (Spot=0.3619)
    "a10":                  4.160,  # Standard_NV36ads_A10_v5 (整块A10大VM)  Sweden Central (Spot=0.7688)
}

# 同一块 RTX PRO 6000 1/4 切片也能搭别的主机 SKU: GPU 吞吐相同, 仅主机 vCPU/内存/单价不同。
# 实测跑在 NC24; 下面用同一份 rtx6000-quarter-nc24.csv 的吞吐, 换价格再折算一行 QPS/$
# (NC36 的 vCPU 更多, 不会降低 GPU-bound 吞吐, 用 NC24 吞吐估它的 QPS/$ 偏保守)。
ALT_HOST_SKUS = {
    "rtx6000-quarter-nc24": [
        ("rtx6000-quarter-nc36", 1.243),  # Standard_NC36lds_xl_RTXPRO6000BSE_v6  36 vCPU/72GB  West US 2
    ],
    # 注: NC4as/NC8as_T4_v3 (更便宜的单卡 T4 主机) 未纳入对比 —— 那是向更少 vCPU 折算的
    # 未实测推算 (客户端能否喂满存疑), 本报告只用实测/保守口径。RTX NC36 是向更多 vCPU
    # 折算, 对 GPU-bound 吞吐安全, 故保留。
}


def load(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"Inferences/Second": "qps", "p99 latency": "p99_us"})
    df["p99_ms"] = df["p99_us"] / 1000.0
    return df.sort_values("Concurrency")


def summarize(label, df, sla_ms, price):
    ok = df[df["p99_ms"] <= sla_ms]
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
        # 面向客户的成本指标: 每百万次推理机时费 (USD), 越低越省。
        # = 单价 / (QPS * 3600秒) * 1e6, 与 qps_per_usd_hr 互为倒数, 排名一致。
        "usd_per_1m_inf": round(price / best["qps"] / 3600 * 1e6, 2) if price else None,
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
        rows.append(summarize(label, df, args.sla_ms, PRICES_PER_HOUR.get(label)))
        # 同一 GPU 切片的其它主机 SKU: 复用同一份吞吐, 换价格再算一行 QPS/$
        for alt_label, alt_price in ALT_HOST_SKUS.get(label, []):
            rows.append(summarize(alt_label, df, args.sla_ms, alt_price))

    out = pd.DataFrame(rows)
    print(f"\n=== YOLOv8s 三卡对比 (p99 SLA ≤ {args.sla_ms} ms) ===")
    print(out.to_string(index=False))

    if "usd_per_1m_inf" in out.columns and out["usd_per_1m_inf"].notna().any():
        best = out.loc[out["usd_per_1m_inf"].idxmin()]
        print(f"\n单位推理成本最低: {best['label']}  →  ${best['usd_per_1m_inf']} / 百万次推理"
              f"  ({best['qps_per_usd_hr']} QPS per $/hr)")
        print("提示: 完整 RTX PRO 6000 ≈ 4× quarter, 比较整卡方案时 quarter 的吞吐 ×4、价格 ×4。")
    else:
        print("\n(在 analyze.py 顶部填 t4 / a10 的价格后可得每美元性能排名)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 图例用 Azure SKU 名 (与报告口径一致); 未命中的标签退回原名
        sku_names = {
            "a10": "Standard_NV36ads_A10_v5",
            "t4": "Standard_NC16as_T4_v3",
            "rtx6000-quarter-nc24": "Standard_NC24lds_xl_RTXPRO6000BSE_v6",
            "rtx6000-quarter-nc36": "Standard_NC36lds_xl_RTXPRO6000BSE_v6",
        }
        plt.figure(figsize=(9, 6.5))
        for label, df in curves.items():
            d = df.sort_values("Concurrency")
            plt.plot(d["qps"], d["p99_ms"], marker="o", label=sku_names.get(label, label))
            # 标注每条曲线的并发起点/终点, 说明每个点是一档并发
            c0, c1 = d.iloc[0], d.iloc[-1]
            plt.annotate(f"c={int(c0['Concurrency'])}", (c0["qps"], c0["p99_ms"]),
                         fontsize=7, xytext=(3, -8), textcoords="offset points")
            plt.annotate(f"c={int(c1['Concurrency'])}", (c1["qps"], c1["p99_ms"]),
                         fontsize=7, xytext=(3, 4), textcoords="offset points")
        plt.axhline(args.sla_ms, ls="--", c="gray", label=f"SLA {args.sla_ms}ms")
        plt.xlabel("Throughput (QPS)")
        plt.ylabel("p99 latency (ms)")
        plt.title("YOLOv8s: throughput vs p99 latency")
        # 图例放左上空白区, 避开 T4 在 ~225 QPS 处的竖直爬升
        plt.legend(loc="upper left", framealpha=0.9)
        plt.grid(alpha=0.3)
        # 并发扫描说明挪到图下方备注, 让标题精简
        plt.figtext(0.5, -0.03,
                    "Note: each curve is a concurrency sweep 1->32; each marker = one concurrency level.",
                    ha="center", fontsize=8, color="gray")
        plt.savefig(args.plot, dpi=130, bbox_inches="tight")
        print(f"曲线已保存: {args.plot}")
    except ImportError:
        print("(pip install matplotlib 可出延迟-吞吐曲线图)")


if __name__ == "__main__":
    main()
