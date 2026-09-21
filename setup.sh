#!/usr/bin/env bash
# 一次性环境准备: 拉镜像 + 下权重 + 导出 ONNX(+ 可选 COCO128)。三台机器各跑一次。
set -euo pipefail

TAG="${TRITON_TAG:-25.08}"
SERVER="nvcr.io/nvidia/tritonserver:${TAG}-py3"
SDK="nvcr.io/nvidia/tritonserver:${TAG}-py3-sdk"
ULTRA="ultralytics/ultralytics:latest-cpu"

echo "[setup 1/3] 拉取 Triton 镜像 (server + sdk), tag=${TAG} ..."
docker pull "$SERVER"
docker pull "$SDK"

echo "[setup 2/3] 导出 yolov8s.onnx (ultralytics CPU 镜像, 首次自动下载权重)..."
if [ -f yolov8s.onnx ]; then
  echo "  已存在 yolov8s.onnx, 跳过"
else
  docker pull "$ULTRA"
  docker run --rm -v "$PWD":/work -w /work "$ULTRA" python export_onnx.py
fi

if [ "${1:-}" = "--coco128" ]; then
  echo "[setup 3/3] 拉取 COCO128 (~7MB, 可选真实图片/精度抽检)..."
  docker run --rm -v "$PWD":/work -w /work "$ULTRA" \
    python -c "from ultralytics.utils.downloads import safe_download; safe_download('https://ultralytics.com/assets/coco128.zip', dir='datasets')"
else
  echo "[setup 3/3] 跳过 COCO128 (加 --coco128 可拉取)"
fi

echo "[setup] 完成. 现在跑:  bash run.sh <gpu-label>   (rtx6000-quarter | t4 | a10)"
