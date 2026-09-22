#!/usr/bin/env bash
# perf_analyzer 并发扫描 (在线延迟场景)。在 Triton SDK 容器内运行。
#
# 测量口径 (科学性):
#   - 每档先发 WARMUP 个预热请求 (--warmup-request-count): 只跑不计入统计, 剔除
#     冷启动 / 填满在途管道那段抖动, 保证测的是稳态。
#   - 再发固定 REQUESTS 个请求 (--request-count) 作为样本: 样本量写死, p99 才稳
#     (2500 请求 → 尾部约 25 个样本估计 p99, 够用; 内存宽裕可调高更紧)。
#   - 三台卡同脚本同参数, 口径完全一致, 保证横向对比公平。
#
# 内存安全: perf_analyzer 会把一档内每个"请求+响应"都留在客户端内存不释放。YOLOv8s
#   单请求张量 ≈ 7.4MB (输入 images[3,640,640] FP32 4.7MB + 输出 output0[84,8400]
#   FP32 2.7MB), 但 gRPC 会同时留序列化+反序列化两份副本+protobuf 开销, 实测单请求
#   实际吃 ~12MB。固定请求数 → 单档内存写死 ≈ (WARMUP+REQUESTS) × 12MB, 与并发范围
#   无关 (每档单独进程, 跨档释放)。NC24 (标称72GiB, free -h 仅 ~70Gi 可用, Swap=0) 实测
#   6000 请求(5000+1000)就吃满 70Gi → OOM! 故默认压到 2500 (~42GB, 稳)。内存宽裕的机器
#   可调高: T4 ~108Gi → 5000; A10 440GiB → 8000 (已实测)。样本量只影响 p99 精度不改真值,
#   各机器不同 REQUESTS 不影响横向公平。
set -uo pipefail

LABEL="${1:?need <gpu-label>}"
CONCURRENCY="${CONCURRENCY:-1:32:1}"      # 起:止:步长
REQUESTS="${REQUESTS:-2500}"              # 每档计入统计的请求数 (决定 p99 样本量与单档内存; 默认取全机型都安全的值)
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
