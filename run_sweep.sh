#!/usr/bin/env bash
# perf_analyzer 并发扫描 (在线延迟场景)。在 Triton SDK 容器内运行。
#
# 内存安全 (关键): YOLOv8s 输出 output0 [84,8400] FP32 ≈ 2.8MB/请求, perf_analyzer
# 会把一档内每个响应都留在客户端内存里不释放。默认的"稳定化"测量 (跑到连续 3 个
# 窗口 p99+吞吐收敛) 在拐点/饱和档可能一直收不敛 -> 无限跑窗口 -> 单档就攒到几十万
# 个响应 -> ~450GB -> OOM kill。
# 因此这里两道保险:
#   1) 每档发固定 REQUESTS 个请求就停 (--request-count), 跳过稳定化循环 ->
#      单档内存写死 = REQUESTS × 2.8MB (4000 档约 11GB, 离 OOM 天远);
#   2) 每档单独起进程, 跑完即退出释放内存, 跨档不累积。
set -uo pipefail

LABEL="${1:?need <gpu-label>}"
CONCURRENCY="${CONCURRENCY:-1:32:1}"      # 起:止:步长
REQUESTS="${REQUESTS:-4000}"              # 每个并发点固定发多少请求 (决定单档内存与 p99 样本量)
mkdir -p results

IFS=: read -r START END STEP <<< "$CONCURRENCY"
STEP="${STEP:-1}"

OUT="results/${LABEL}.csv"
TMP="results/.${LABEL}.tmp.csv"
: > "$OUT"

for c in $(seq "$START" "$STEP" "$END"); do
  echo "== concurrency $c  (request-count $REQUESTS) =="
  if ! perf_analyzer \
        -m yolov8s \
        -u localhost:8001 -i grpc \
        --concurrency-range "${c}:${c}:1" \
        --request-count "$REQUESTS" \
        --percentile=99 \
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
