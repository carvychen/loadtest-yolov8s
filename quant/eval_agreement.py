"""分类模型 (swin_t) 的量化精度: Triton 上各变体的真实引擎对 COCO 评测集 4500 张 (与校准集不重叠) 推理,
与 FP16 引擎逐张比对输出, 回答 "FP8 有没有把模型算坏"。ImageNet-22K 没有官方验证集, 不测绝对准确率。

指标: top1_agree (top-1 类别相同的比例), top5_overlap (top-5 集合平均重合率), cos_mean / cos_min (logits 余弦相似度)。
用法 (仓库根目录下): python quant/eval_agreement.py --label L --model swin_t fp16 fp8 ...   (第一个变体为参照, 须为 fp16)
产物: results/quant/<label>-<变体>-agree.json
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class Agreement:
    """分批累加的一致性统计 (ref = 参照变体的 logits)。"""

    def __init__(self):
        self.n, self.top1, self.top5, self.cos = 0, 0, 0.0, []

    def add(self, ref, got):
        ref, got = ref.astype(np.float64), got.astype(np.float64)
        self.n += len(ref)
        self.top1 += int((ref.argmax(1) == got.argmax(1)).sum())
        r5, g5 = np.argsort(-ref, 1)[:, :5], np.argsort(-got, 1)[:, :5]
        self.top5 += sum(len(set(a) & set(b)) / 5 for a, b in zip(r5, g5))
        self.cos += list((ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1)))

    def result(self):
        return {"n_images": self.n, "top1_agree": round(self.top1 / self.n, 4),
                "top5_overlap": round(self.top5 / self.n, 4),
                "cos_mean": round(float(np.mean(self.cos)), 5), "cos_min": round(float(np.min(self.cos)), 5)}


def main():
    import tritonclient.grpc as grpcclient

    import swin
    from quant_common import img_path, split_ids

    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="swin_t")
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 张 (调试用, 0=全部 4500)")
    ap.add_argument("variants", nargs="+")
    a = ap.parse_args()
    if a.variants[0] != "fp16":
        raise SystemExit("第一个变体须为 fp16 (一致性的参照)")

    _, ids = split_ids()
    ids = ids[:a.limit] if a.limit else ids
    client = grpcclient.InferenceServerClient(a.url)
    names = {v: f"{a.model}_{v}" for v in a.variants}
    dtypes = {v: client.get_model_metadata(n).inputs[0].datatype for v, n in names.items()}   # FP32 / FP16 (torch 路径)
    stats = {v: Agreement() for v in a.variants}
    tf = swin.transform(swin.build_model())

    def infer(v, x):
        inp = grpcclient.InferInput("input", list(x.shape), dtypes[v])
        inp.set_data_from_numpy(x.astype(np.float16 if dtypes[v] == "FP16" else np.float32))
        return client.infer(names[v], [inp]).as_numpy("logits")

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for k in range(0, len(ids), a.batch):
            x = np.stack(list(ex.map(lambda i: swin.preprocess(tf, img_path(i)), ids[k:k + a.batch])))
            ref = infer("fp16", x)
            for v in a.variants:
                stats[v].add(ref, ref if v == "fp16" else infer(v, x))
            if (k // a.batch) % 50 == 0:
                print(f"  {k + len(x)}/{len(ids)}", flush=True)
    for v in a.variants:
        res = {"model": names[v], "reference": names["fp16"], **stats[v].result(),
               "total_seconds_all_variants": round(time.time() - t0, 1)}
        with open(f"results/quant/{a.label}-{v}-agree.json", "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
