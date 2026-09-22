#!/usr/bin/env bash
# perf_analyzer 并发扫描 (在线延迟场景)。在 Triton SDK 容器内运行。
#
# 测量口径 (科学性):
#   - 每档先发 WARMUP 个预热请求 (--warmup-request-count): 只跑不计入统计, 剔除
#     冷启动 / 填满在途管道那段抖动, 保证测的是稳态。
#   - 再发固定 REQUESTS 个请求 (--request-count) 作为样本: 样本量写死, p99 才稳
#     (8000 请求 → 尾部约 80 个样本估计 p99, 够紧)。
#   - 三台卡同脚本同参数, 口径完全一致, 保证横向对比公平。
#
# 内存安全: perf_analyzer 会把一档内每个"请求+响应"都留在客户端内存不释放。YOLOv8s
#   单请求 ≈ 7.7MB = 输入 images[3,640,640] FP32 (4.9MB) + 输出 output0[84,8400]
#   FP32 (2.8MB)。固定请求数 → 单档内存写死 = (WARMUP+REQUESTS) × 7.7MB, 与并发范围
#   无关 (每档单独进程, 跨档释放)。默认 9000×7.7MB ≈ 69GB → 在 NC24 (70GiB, Swap=0)
#   会 OOM! 内存小的机器请降 REQUESTS (如 2500 → 约 23GB, 稳)。这也避免了旧稳定化模式
#   在饱和档不收敛 → 无限跑窗口 → OOM kill。
set -uo pipefail

LABEL="${1:?need <gpu-label>}"
CONCURRENCY="${CONCURRENCY:-1:32:1}"      # 起:止:步长
REQUESTS="${REQUESTS:-8000}"              # 每档计入统计的请求数 (决定 p99 样本量与单档内存)
WARMUP="${WARMUP:-1000}"                  # 每档预热请求数 (只跑不计入, 剔除冷启动)
mkdir -p results

IFS=: read -r START END STEP <<< "$CONCURRENCY"
STEP="${STEP:-1}"

OUT="results/${LABEL}.csv"
TMP="results/.${LABEL}.tmp.csv"
: > "$OUT"

for c in $(seq "$START" "$STEP" "$END"); do
  echo "== concurrency $c  (warmup $WARMUP + measure $REQUESTS) =="
  if ! perf_analyzer \
        -m yolov8s \
        -u localhost:8001 -i grpc \
        --concurrency-range "${c}:${c}:1" \
        --warmup-request-count "$WARMUP" \
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
