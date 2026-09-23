#!/usr/bin/env bash
# 一条命令跑完本机全流程: 建引擎 -> 起 Triton -> perf_analyzer 并发扫描 -> results/<label>.csv
set -euo pipefail

LABEL="${1:?用法: bash run.sh <gpu-label>   例如 rtx6000-quarter | t4 | a10}"
TAG="${TRITON_TAG:-25.08}"
SERVER="nvcr.io/nvidia/tritonserver:${TAG}-py3"
SDK="nvcr.io/nvidia/tritonserver:${TAG}-py3-sdk"

cleanup() { docker rm -f triton-lt >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "[1/4] 构建本卡 TensorRT 引擎 (FP16)..."
docker run --gpus all --rm -v "$PWD":/work -w /work "$SERVER" bash build_engine.sh

echo "[2/4] 启动 Triton..."
cleanup
docker run --gpus all -d --rm --name triton-lt --net host \
  -v "$PWD/model_repository":/models "$SERVER" \
  tritonserver --model-repository=/models >/dev/null

echo -n "      等待 READY"
for i in $(seq 1 60); do
  if curl -sf localhost:8000/v2/health/ready >/dev/null 2>&1; then echo " ✓"; break; fi
  echo -n "."; sleep 2
  if [ "$i" = 60 ]; then echo " 超时"; docker logs triton-lt | tail -40; exit 1; fi
done

echo "[3/4] perf_analyzer 并发扫描 (1→32)..."
# 透传扫描参数进容器 (-e 不带值即取宿主机同名变量): 小内存机型 (如 NC4as_T4_v3 28GiB)
# 必须能从外部压低 REQUESTS/WARMUP, 否则容器内用默认 2500/1000 会 OOM。
docker run --rm --net host -v "$PWD":/work -w /work \
  -e CONCURRENCY -e REQUESTS -e WARMUP \
  "$SDK" bash run_sweep.sh "$LABEL"

echo "[4/4] 停止 Triton"
cleanup
echo "完成 → results/${LABEL}.csv  (收齐三份后跑: python analyze.py)"
