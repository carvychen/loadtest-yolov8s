"""fp8-torch 变体: 走 NVIDIA 推荐的 Swin FP8 路径 —— modelopt.torch 在 PyTorch 里量化, 再导出 Q/DQ ONNX。

与 ONNX PTQ (modelopt.onnx.quantization) 的区别 (沿用 ModelOpt 0.47.0 示例 examples/torch_onnx/torch_quant_to_onnx.py):
  - FP8 预设 + "MHA-aware" 配置: 量化 LayerNorm 输出 (Q/K/V/FC 共用一组 Q/DQ, 便于 TRT 融合),
    导出时再给 Softmax 输出插固定 1/448 scale 的 Q/DQ —— 移位窗口块的注意力也在其中
  - 不量化 patch_embed (3 通道首层卷积, Blackwell 上没有 FP8 tactic) 与校准出 amax<=0 的量化器
差别只在量化方式: 模型、权重、校准集 (datasets/calib_swin_t_500.npy) 与 ONNX PTQ 变体完全相同。
在量化容器内运行 (CPU 校准)。用法见 variants.Variant.quantize_cmd。
"""
import argparse
import os
import re

import numpy as np
import torch

import swin

ap = argparse.ArgumentParser()
ap.add_argument("--calibration_data_path", required=True)
ap.add_argument("--output_path", required=True)
ap.add_argument("--calib_batch", type=int, default=16)
a = ap.parse_args()

import modelopt.torch.quantization as mtq                      # noqa: E402  (导入较慢, 参数检查之后再导)
from modelopt.recipe import load_config                        # noqa: E402
from modelopt.recipe.presets import MODEL_QUANT_PRESET_DIR     # noqa: E402
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata   # noqa: E402
from modelopt.torch.quantization.config import QuantizeConfig  # noqa: E402

# ── 以下三段配置照抄 ModelOpt 0.47.0 示例 (Apache-2.0), 保持与官方路径一致 ──
FP8_MHA_OVERRIDE = [
    {"parent_class": "nn.LayerNorm", "quantizer_name": "*output_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
    {"parent_class": "nn.LayerNorm", "quantizer_name": "*input_quantizer", "enable": False},
]
EXCLUDE = re.compile(r".*(patch_embed|pos_embed|global_pool|downsample\.reduction).*")


def dead(q):
    amax = getattr(q, "amax", None)
    return torch.is_tensor(amax) and (torch.any(torch.isnan(amax)) or torch.all(amax <= 0))


model = swin.build_model()
config = load_config(f"{MODEL_QUANT_PRESET_DIR}/fp8", schema_type=QuantizeConfig).model_dump()
config["quant_cfg"].extend(FP8_MHA_OVERRIDE)
calib = torch.from_numpy(np.load(a.calibration_data_path))
print(f"校准: {tuple(calib.shape)}, 每批 {a.calib_batch}")


def forward_loop(m):
    with torch.no_grad():
        for i in range(0, len(calib), a.calib_batch):
            m(calib[i:i + a.calib_batch])


model = mtq.quantize(model, config, forward_loop=forward_loop)
mtq.disable_quantizer(model, lambda n: EXCLUDE.match(n) is not None)
for _, mod in model.named_modules():
    for attr in ("input_quantizer", "output_quantizer", "weight_quantizer"):
        q = getattr(mod, attr, None)
        if q is not None and q.is_enabled and dead(q):
            q.disable()
mtq.print_quant_summary(model)

# 导出: 权重存 FP16 (示例默认), I/O 类型由 prepare_variants 从 ONNX 读出写进 Triton 配置。
# ModelOpt 按 forward 形参命名 I/O (timm 为 x / out), 动态 batch 要用这两个名字声明, 导出后再改成与其它变体一致的 input / logits
onnx_bytes, _ = get_onnx_bytes_and_metadata(
    model, (torch.randn(1, 3, 224, 224),), model_name=os.path.basename(a.output_path)[:-len(".onnx")],
    dynamic_axes={"x": {0: "batch"}, "out": {0: "batch"}}, weights_dtype="fp16")
OnnxBytes.from_bytes(onnx_bytes).write_to_disk(os.path.dirname(a.output_path) or ".", clean_dir=False)

import onnx   # noqa: E402
m = onnx.load(a.output_path)
renames = {m.graph.input[0].name: "input", m.graph.output[0].name: "logits"}
for t in (m.graph.input[0], m.graph.output[0]):
    t.name = renames[t.name]
for n in m.graph.node:
    n.input[:] = [renames.get(x, x) for x in n.input]
    n.output[:] = [renames.get(x, x) for x in n.output]
dims = [d.dim_param or d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
if dims[0] != "batch":
    raise SystemExit(f"导出的 batch 维不是动态的: {dims}")
onnx.save(m, a.output_path)
print(f"导出 {a.output_path}  输入 {dims}")
