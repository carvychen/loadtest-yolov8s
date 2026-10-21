# Glossary

二期 FP8 量化实验的领域术语。代码里的名字与这里一致。

**变体 (Variant)**：一次实验对象 = 一种精度 + 若干标签，写成 `<精度>[-标签...]`，如 `fp8-nohead-opt5`。每个变体对应一个 Triton 模型 `yolov8s_<变体>` 和一组结果文件 `<label>-<变体>-*`。规则全部在 `quant/variants.py`。
_Avoid_：模型（与 Triton 模型、YOLO 模型混淆）、配置。

**精度 (Precision)**：变体的低精度类型，`fp16` / `fp8` / `int8`。`fp16` 是基线，不量化；`fp8`、`int8` 走 ModelOpt PTQ。

**量化标签 (Quant tag)**：改变 Q/DQ ONNX 的标签，如 `nohead`（检测头 model.22 不量化）、`conv`（只量化 Conv）。不能用在 `fp16` 上。

**构建标签 (Build tag)**：只改 trtexec 构建参数、复用同一份 ONNX 的标签，如 `opt5`（`--builderOptimizationLevel=5`）。

**量化模型名 (Quant name)**：精度 + 量化标签（固定顺序），决定 Q/DQ ONNX 文件 `yolov8s.<quant_name>.onnx` 与量化日志；构建标签不同的变体共用同一个量化模型名。

**两层测速 (compute / transfer)**：trtexec 测速的两种口径。compute 只计 GPU kernel 时间，transfer 再加 PCIe H2D/D2H。第三层（端到端）是 Triton + perf_analyzer。

**TRT 主版本 (TRT major)**：TensorRT 大版本号（10 / 11）。11 移除了弱类型（无 `--fp16`），并默认开启 CUDA Graph、关闭数据传输，所以构建与测速参数都按它分支。
