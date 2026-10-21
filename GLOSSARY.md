# Glossary

二期 FP8 量化实验的领域术语。代码里的名字与这里一致。

**模型 (Model)**：被测网络（`yolov8s` 检测 / `swin_t` 分类），决定产物文件名 `artifacts/<模型>.*`、输入张量、校准集、Triton 模板和可用的量化标签。定义在 `quant/models.py`，各脚本用 `--model` 选，默认 `yolov8s`。

**变体 (Variant)**：一次实验对象 = 一种精度 + 若干标签，写成 `<精度>[-标签...]`，如 `fp8-nohead-opt5`。变体属于某个模型，对应 Triton 模型 `<模型>_<变体>` 和一组结果文件 `<label>-<变体>-*`。规则全部在 `quant/variants.py`。
_Avoid_：模型（与 Triton 模型、YOLO 模型混淆）、配置。

**精度 (Precision)**：变体的低精度类型，`fp16` / `fp8` / `int8`。`fp16` 是基线，不量化；`fp8`、`int8` 走 ModelOpt PTQ。

**量化标签 (Quant tag)**：改变 Q/DQ ONNX 的标签。YOLO：`nohead`（检测头 model.22 不量化）、`conv`（只量化 Conv）；Swin：`nomha`（注意力 BMM 不量化）、`torch`（改走 modelopt.torch 量化后导出）。不能用在 `fp16` 上，也只能用在所属模型上。

**构建标签 (Build tag)**：只改 trtexec 构建参数、复用同一份 ONNX 的标签，如 `opt5`（`--builderOptimizationLevel=5`）。

**量化模型名 (Quant name)**：精度 + 量化标签（固定顺序），决定 Q/DQ ONNX 文件 `yolov8s.<quant_name>.onnx` 与量化日志；构建标签不同的变体共用同一个量化模型名。

**输出一致性 (Agreement)**：分类模型的量化精度口径——同一批图片上，变体与 FP16 引擎输出的 top-1 一致率、top-5 重合率、logits 余弦相似度。

**两层测速 (compute / transfer)**：trtexec 测速的两种口径。compute 只计 GPU kernel 时间，transfer 再加 PCIe H2D/D2H。第三层（端到端）是 Triton + perf_analyzer。

**TRT 主版本 (TRT major)**：TensorRT 大版本号（10 / 11）。11 移除了弱类型（无 `--fp16`），并默认开启 CUDA Graph、关闭数据传输，所以构建与测速参数都按它分支。
