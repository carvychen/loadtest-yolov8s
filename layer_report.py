"""把 trtexec 导出的逐层信息 (layers.json) 与逐层耗时 (profile.json) 合并, 回答两个问题:
  1. 量化覆盖率: 卷积有多少真的跑在 FP8/INT8 上, 多少退回 FP16?
  2. 额外开销: reformat (精度/布局转换) 与未融合 Q/DQ 吃掉了多少时间?

用法: python layer_report.py results/quant/<label>-<variant>   (自动读 -layers.json / -profile.json)
"""
import json
import re
import sys
from collections import defaultdict


def precision_of(fmt):
    f = (fmt or "").lower()
    for key, pat in (("FP8", r"fp8|e4m3"), ("INT8", r"int8"), ("FP16", r"fp16|half"),
                     ("FP32", r"fp32|float")):
        if re.search(pat, f):
            return key
    return "?"


def kind_of(layer):
    name, ltype = layer.get("Name", ""), layer.get("LayerType", "")
    if ltype == "Reformat" or "reformat" in name.lower():
        return "reformat"
    # Myelin 编译的层 (FP8 stronglyTyped 常见, 名如 /model_1/conv/Conv_myl0_4) LayerType 不叫 Conv, 要看名字
    is_conv = "conv" in ltype.lower() or "Conv" in name
    if re.search(r"QuantizeLinear|DequantizeLinear", name) and not is_conv:
        return "qdq_unfused"
    if is_conv:
        return "conv"
    if "myelin" in ltype.lower() or ltype == "ForeignNode":
        return "myelin"
    return "other"


def load(prefix):
    with open(f"{prefix}-layers.json") as f:
        info = json.load(f)
    layers = info["Layers"] if isinstance(info, dict) else info
    with open(f"{prefix}-profile.json") as f:
        prof = [p for p in json.load(f) if "name" in p]
    t = {p["name"]: p.get("averageMs", 0.0) for p in prof}
    rows = []
    for L in layers:
        if isinstance(L, str):          # profilingVerbosity 不是 detailed 时只有名字
            rows.append({"name": L, "kind": "?", "prec": "?", "ms": t.get(L, 0.0)})
            continue
        ins = L.get("Inputs") or [{}]
        rows.append({"name": L.get("Name", ""), "kind": kind_of(L),
                     "prec": precision_of(ins[0].get("Format/Datatype")),
                     "ms": t.get(L.get("Name", ""), 0.0), "tactic": L.get("TacticName", "")})
    return rows


def summarize(prefix):
    rows = load(prefix)
    total = sum(r["ms"] for r in rows) or 1e-9
    by_kind, conv_prec, conv_ms = defaultdict(float), defaultdict(int), defaultdict(float)
    for r in rows:
        by_kind[r["kind"]] += r["ms"]
        if r["kind"] == "conv":
            conv_prec[r["prec"]] += 1
            conv_ms[r["prec"]] += r["ms"]
    n_conv = sum(conv_prec.values()) or 1
    low = conv_prec.get("FP8", 0) + conv_prec.get("INT8", 0)
    return {
        "layers": len(rows),
        "total_ms": round(total, 3),
        "conv_count_by_prec": dict(conv_prec),
        "conv_ms_by_prec": {k: round(v, 3) for k, v in conv_ms.items()},
        "low_prec_conv_ratio": round(low / n_conv, 3),
        "time_share": {k: round(v / total, 3) for k, v in by_kind.items()},
        "top10": [(r["name"][:90], r["kind"], r["prec"], round(r["ms"], 4))
                  for r in sorted(rows, key=lambda r: -r["ms"])[:10]],
    }


if __name__ == "__main__":
    s = summarize(sys.argv[1])
    print(json.dumps(s, indent=2, ensure_ascii=False))
