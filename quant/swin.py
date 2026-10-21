"""Swin-T (三期): timm 建模并加载官方 ImageNet-22K 权重、预处理、ONNX 导出。在量化容器内运行 (CPU)。

用 timm 而不是官方仓库代码: timm 的 swin_tiny_patch4_window7_224.ms_in22k 正是这份 .pth (checkpoint_filter_fn
负责改名), 计算与官方实现相同; ModelOpt 官方的 torch 量化示例也基于 timm, 两条量化路径可用同一个模型对比。
"""
import os
import urllib.request

os.environ.setdefault("TIMM_FUSED_ATTN", "0")   # 关掉 SDPA, 注意力写成显式 MatMul+Softmax (与官方代码同构, ModelOpt 才认得 MHA)

CKPT = "artifacts/swin_tiny_patch4_window7_224_22k.pth"
CKPT_URL = "https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth"
TIMM_NAME = "swin_tiny_patch4_window7_224.ms_in22k"
NUM_CLASSES = 21841


def build_model(gelu="none"):
    """timm Swin-T + 官方 22K 权重 (本地没有就从官方 release 下载), eval 模式, FP32, CPU。
    gelu="tanh": 把 GELU 换成 tanh 近似 (同一权重, swin_t_tanh 模型用)。"""
    import timm
    import torch
    from timm.models.swin_transformer import checkpoint_filter_fn

    if not os.path.exists(CKPT):
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        print(f"下载 {CKPT_URL}")
        urllib.request.urlretrieve(CKPT_URL, CKPT)
    model = timm.create_model(TIMM_NAME, pretrained=False, num_classes=NUM_CLASSES)
    sd = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(checkpoint_filter_fn(sd.get("model", sd), model))
    if gelu != "none":
        for mod in model.modules():
            if isinstance(mod, torch.nn.GELU):
                mod.approximate = gelu
    return model.eval()


def transform(model):
    """timm 的推理预处理 (该权重的 pretrained_cfg: 224, bicubic, crop_pct 0.9, ImageNet 均值方差)。
    校准与评测必须走同一份, 否则校准出的 scale 与推理时的激活分布对不上。"""
    import timm
    cfg = timm.data.resolve_model_data_config(model)
    return timm.data.create_transform(**cfg, is_training=False)


def preprocess(tf, path):
    """图片 -> (3,224,224) float32 numpy。"""
    from PIL import Image
    return tf(Image.open(path).convert("RGB")).numpy()


GELU = {"swin_t": "none", "swin_t_tanh": "tanh"}   # 模型名 -> GELU 写法


def export_onnx(name="swin_t"):
    """导出 FP32 ONNX (opset 17, 动态 batch), 并用 onnxruntime 对比 PyTorch 输出确认导出无误。"""
    import numpy as np
    import onnxruntime as ort
    import torch

    out = f"artifacts/{name}.onnx"
    model = build_model(GELU[name])
    x = torch.randn(2, 3, 224, 224)
    torch.onnx.export(model, x, out, input_names=["input"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    with torch.no_grad():
        ref = model(x).numpy()
    got = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    err = float(np.abs(got - ref).max())
    print(f"导出 {out}  onnxruntime 与 PyTorch 最大误差 {err:.2e}")
    if err > 1e-3:
        raise SystemExit(f"ONNX 导出结果与 PyTorch 不一致 (最大误差 {err})")


if __name__ == "__main__":
    # 用法 (仓库根目录下): python quant/swin.py export [--model swin_t|swin_t_tanh]   (已存在则跳过)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export"])
    ap.add_argument("--model", default="swin_t", choices=list(GELU))
    a = ap.parse_args()
    if os.path.exists(f"artifacts/{a.model}.onnx"):
        print(f"已存在 artifacts/{a.model}.onnx, 跳过")
    else:
        export_onnx(a.model)
