"""对 Triton 上某个精度变体的引擎跑 COCO mAP (评测集 4500 张, 与校准集不重叠)。

测的就是线上同一份 TRT 引擎, 不是 ONNX 近似。后处理与 Ultralytics DetectionValidator 对齐:
conf=0.001, iou=0.7, multi-label NMS, max_det=300; 唯一差异是固定 640x640 letterbox
(Ultralytics val 用 rect 批), 所以绝对 mAP 会比官方 44.9 略低, 但各精度同口径, 相对差值可比。

用法 (yolo-quant 容器内, --net host):
    python quant/eval_map.py --model yolov8s_fp8 --out results/quant/<label>-fp8-map.json
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchvision
import tritonclient.grpc as grpcclient
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from quant_common import ANN, COCO80_TO_91, img_path, preprocess, split_ids


def postprocess(pred, r, pad, shape, conf=0.001, iou=0.7, max_det=300, max_nms=30000):
    """pred: (84, 8400) = 4 个 cxcywh + 80 类 sigmoid 分数。返回原图坐标下的检测。"""
    p = torch.from_numpy(np.ascontiguousarray(pred.T, dtype=np.float32))
    boxes, scores = p[:, :4], p[:, 4:]
    i, j = (scores > conf).nonzero(as_tuple=True)          # multi-label: 每个 (框, 类) 都是候选
    if i.numel() == 0:
        return []
    s = scores[i, j]
    if s.numel() > max_nms:
        top = s.topk(max_nms).indices
        i, j, s = i[top], j[top], s[top]
    b = boxes[i]
    xyxy = torch.cat([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)
    keep = torchvision.ops.batched_nms(xyxy, s, j, iou)[:max_det]
    xyxy, s, j = xyxy[keep], s[keep], j[keep]
    xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - pad[0]) / r).clamp(0, shape[1])
    xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - pad[1]) / r).clamp(0, shape[0])
    return [(xyxy[k].tolist(), float(s[k]), int(j[k])) for k in range(len(keep))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Triton 模型名, 如 yolov8s_fp16")
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 张 (调试用, 0=全部 4500)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _, eval_ids = split_ids()
    if args.limit:
        eval_ids = eval_ids[:args.limit]
    client = grpcclient.InferenceServerClient(args.url)
    in_dtype = client.get_model_metadata(args.model).inputs[0].datatype   # FP32 / FP16
    np_dtype = np.float16 if in_dtype == "FP16" else np.float32

    dets, t0 = [], time.time()
    with ThreadPoolExecutor(8) as ex:
        for k in range(0, len(eval_ids), args.batch):
            ids = eval_ids[k:k + args.batch]
            items = list(ex.map(lambda i: preprocess(img_path(i)), ids))
            x = np.stack([it[0] for it in items]).astype(np_dtype)
            inp = grpcclient.InferInput("images", list(x.shape), in_dtype)
            inp.set_data_from_numpy(x)
            out = client.infer(args.model, [inp]).as_numpy("output0")
            for image_id, (_, r, pad, shape), pred in zip(ids, items, out):
                for (x1, y1, x2, y2), score, c in postprocess(pred, r, pad, shape):
                    dets.append({"image_id": image_id, "category_id": COCO80_TO_91[c],
                                 "bbox": [round(x1, 3), round(y1, 3),
                                          round(x2 - x1, 3), round(y2 - y1, 3)],
                                 "score": round(score, 5)})
            if (k // args.batch) % 50 == 0:
                print(f"  {args.model}: {k + len(ids)}/{len(eval_ids)}", flush=True)
    secs = time.time() - t0
    if not dets:   # 量化把输出打坏时会一个框都没有, loadRes 会直接报错
        raise SystemExit(f"{args.model}: 0 个检测框 (conf>0.001), 引擎输出异常, 检查量化/构建日志")

    gt = COCO(ANN)
    E = COCOeval(gt, gt.loadRes(dets), "bbox")
    E.params.imgIds = eval_ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    res = {"model": args.model, "n_images": len(eval_ids), "n_dets": len(dets),
           "mAP50_95": round(float(E.stats[0]), 4), "mAP50": round(float(E.stats[1]), 4),
           "seconds": round(secs, 1)}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
