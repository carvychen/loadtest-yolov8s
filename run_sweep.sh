#!/usr/bin/env bash
# perf_analyzer 并发扫描 (在线延迟场景)。在 Triton SDK 容器内运行。
# 扫并发 1→32, 每点测 p99 延迟与吞吐, 输出 results/<label>.csv。
set -euo pipefail

LABEL="${1:?need <gpu-label>}"
CONCURRENCY="${CONCURRENCY:-1:32:1}"      # 起:止:步长
INTERVAL="${INTERVAL:-5000}"              # 每个并发点测量窗口(ms)
mkdir -p results

perf_analyzer \
  -m yolov8s \
  -u localhost:8001 -i grpc \
  --concurrency-range "$CONCURRENCY" \
  --percentile=99 \
  --measurement-interval "$INTERVAL" \
  --shape images:3,640,640 \
  -f "results/${LABEL}.csv"

echo "wrote results/${LABEL}.csv"
