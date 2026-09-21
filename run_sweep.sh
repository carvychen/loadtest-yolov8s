#!/usr/bin/env bash
# perf_analyzer 并发扫描 (在线延迟场景)。在 Triton SDK 容器内运行。
# 逐档单独调用 perf_analyzer: 每档进程跑完即退出、释放内存, 避免单进程累积
# 所有响应 (YOLOv8s 输出 output0 ~2.8MB/请求) 在高并发档吃光内存被 OOM kill。
set -uo pipefail

LABEL="${1:?need <gpu-label>}"
CONCURRENCY="${CONCURRENCY:-1:32:1}"      # 起:止:步长
INTERVAL="${INTERVAL:-5000}"              # 每个并发点测量窗口(ms)
STABILITY="${STABILITY:-15}"              # 稳定化阈值(%), 放宽让饱和档更快收敛
mkdir -p results

IFS=: read -r START END STEP <<< "$CONCURRENCY"
STEP="${STEP:-1}"

OUT="results/${LABEL}.csv"
TMP="results/.${LABEL}.tmp.csv"
: > "$OUT"

for c in $(seq "$START" "$STEP" "$END"); do
  echo "== concurrency $c =="
  if ! perf_analyzer \
        -m yolov8s \
        -u localhost:8001 -i grpc \
        --concurrency-range "${c}:${c}:1" \
        --percentile=99 \
        --measurement-interval "$INTERVAL" \
        --stability-percentage "$STABILITY" \
        --shape images:3,640,640 \
        -f "$TMP"; then
    echo "并发 $c 失败, 停止扫描 (已保留前面 $OUT 的结果)"; break
  fi
  if [ ! -s "$OUT" ]; then
    cat "$TMP" >> "$OUT"          # 第一档: 连表头一起写
  else
    tail -n +2 "$TMP" >> "$OUT"   # 之后各档: 只追加数据行
  fi
  rm -f "$TMP"
done
rm -f "$TMP"

echo "wrote $OUT"
