#!/usr/bin/env bash
# onnx -> TensorRT 引擎 (FP16, 动态 shape)。在 Triton 容器内运行, 每张卡各自构建。
# 引擎硬件绑定, 不可跨卡复用。
set -euo pipefail

ONNX="${ONNX:-yolov8s.onnx}"
OUT="model_repository/yolov8s/1/model.plan"

if [ ! -f "$ONNX" ]; then
  echo "找不到 $ONNX —— 请先在本机跑: bash setup.sh"; exit 1
fi
mkdir -p "$(dirname "$OUT")"

# 动态 batch: min=1 / opt=8 / max=32, 与 config.pbtxt 的 max_batch_size 一致
trtexec \
  --onnx="$ONNX" \
  --saveEngine="$OUT" \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:8x3x640x640 \
  --maxShapes=images:32x3x640x640 \
  --memPoolSize=workspace:4096 \
  | tee "build_$(hostname).log"

echo "engine -> $OUT"
