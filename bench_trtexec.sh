#!/usr/bin/env bash
# 在 Triton server 容器内运行 (与 serving 同一 TensorRT 版本): 为每个精度变体构建引擎,
# 再用 trtexec 做两层测速 + 逐层 profile。由 run_quant.sh 调用。
#
# 两层测速, 用来把 "FP8 提升不足" 拆成 GPU 内 vs GPU 外:
#   compute   --noDataTransfers + CUDA Graph: 只有 GPU kernel 时间 = 量化在算子层的真实收益上限
#   transfer  带 H2D/D2H (FP32 输入 4.7MB + 输出 2.7MB / 张): 加上 PCIe 搬运后还剩多少
# 再往外一层 (gRPC / 预处理 / 动态批处理) 由 Triton + perf_analyzer 测, 见 run_quant.sh。
#
# 构建口径:
#   fp16           原始 FP32 ONNX + --fp16 (与一期 build_engine.sh 完全一致)
#   fp8 / int8 ... ModelOpt 显式 Q/DQ ONNX + --stronglyTyped (TRT 10 推荐; 精度完全由 Q/DQ 决定)
#   一律 --profilingVerbosity=detailed (不影响性能, 逐层报告需要它)
set -euo pipefail
trap 'echo "!! bench_trtexec.sh 第 $LINENO 行失败 (退出码 $?)"' ERR

LABEL="${1:?need <gpu-label>}"
VARIANTS="${VARIANTS:-fp16 fp8 int8}"
BATCHES="${BATCHES:-1 8 32}"
PROFILE_BATCH="${PROFILE_BATCH:-8}"
DURATION="${DURATION:-15}"                    # 每个测点秒数
OUT=results/quant
mkdir -p "$OUT"

TRTEXEC="$(command -v trtexec || true)"
for cand in /usr/src/tensorrt/bin/trtexec /opt/tensorrt/bin/trtexec; do
  [ -z "$TRTEXEC" ] && [ -x "$cand" ] && TRTEXEC="$cand"
done
: "${TRTEXEC:?找不到 trtexec, 请确认在 Triton server 镜像内运行}"

{ echo "label=$LABEL"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  "$TRTEXEC" --help 2>&1 | grep -m1 -oE "TensorRT v[0-9]+" || true; } > "$OUT/${LABEL}-env.txt" || true

CSV="$OUT/${LABEL}-trtexec.csv"
echo "variant,batch,mode,infer_per_s,img_per_s,gpu_compute_mean_ms,gpu_compute_p99_ms" > "$CSV"

for v in $VARIANTS; do
  PLAN="model_repository_q/yolov8s_${v}/1/model.plan"
  if [ "$v" = fp16 ]; then ONNX=yolov8s.onnx; PREC=(--fp16); else ONNX="yolov8s.${v}.onnx"; PREC=(--stronglyTyped); fi
  [ -f "$ONNX" ] || { echo "缺少 $ONNX, 先跑量化步骤"; exit 1; }

  echo "== [$v] 构建引擎 ($ONNX ${PREC[*]}) =="
  "$TRTEXEC" --onnx="$ONNX" --saveEngine="$PLAN" "${PREC[@]}" \
    --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:32x3x640x640 \
    --memPoolSize=workspace:4096 --profilingVerbosity=detailed \
    > "$OUT/${LABEL}-${v}-build.log" 2>&1 \
    || { echo "[$v] 构建失败, 见 $OUT/${LABEL}-${v}-build.log"; tail -20 "$OUT/${LABEL}-${v}-build.log"; exit 1; }
  echo "== [$v] 引擎已生成 ($(du -h "$PLAN" | cut -f1)), 开始测速 =="
  grep -m1 -oE "TensorRT version: [0-9.]+" "$OUT/${LABEL}-${v}-build.log" >> "$OUT/${LABEL}-env.txt" || true

  for b in $BATCHES; do
    for mode in compute transfer; do
      extra=(); [ "$mode" = compute ] && extra=(--noDataTransfers)
      RLOG="$OUT/${LABEL}-${v}-b${b}-${mode}.log"
      if ! "$TRTEXEC" --loadEngine="$PLAN" --shapes=images:${b}x3x640x640 \
             --useCudaGraph --useSpinWait --warmUp=2000 --duration="$DURATION" "${extra[@]}" \
             > "$RLOG" 2>&1; then
        echo "!! [$v b$b $mode] trtexec 运行失败, 日志 $RLOG 末尾:"; tail -25 "$RLOG"; exit 1
      fi
      # 不用 grep -P / 不让未匹配的 grep 触发 set -e: 解析失败时明确报错并打印日志
      tput=$(grep -oE "Throughput: [0-9.]+" "$RLOG" | tail -1 | grep -oE "[0-9.]+$" || true)
      gline=$(grep "GPU Compute Time:" "$RLOG" | tail -1 || true)
      mean=$(grep -oE "mean = [0-9.]+" <<< "$gline" | grep -oE "[0-9.]+$" || true)
      p99=$(grep -oE "percentile\(99%\) = [0-9.]+" <<< "$gline" | grep -oE "[0-9.]+$" || true)
      if [ -z "$tput" ]; then
        echo "!! [$v b$b $mode] 解析不到 Throughput, 日志 $RLOG 末尾:"; tail -25 "$RLOG"; exit 1
      fi
      img=$(awk -v t="$tput" -v b="$b" 'BEGIN{printf "%.1f", t*b}')   # trtexec 的 qps = 每秒推理次数, 每次 b 张
      echo "$v,$b,$mode,$tput,$img,$mean,$p99" | tee -a "$CSV"
    done
  done

  echo "== [$v] 逐层 profile (batch $PROFILE_BATCH) =="
  # 逐层计时不能开 CUDA Graph; --separateProfileRun 让计时与逐层 profile 分开跑, 互不干扰
  "$TRTEXEC" --loadEngine="$PLAN" --shapes=images:${PROFILE_BATCH}x3x640x640 \
    --noDataTransfers --warmUp=1000 --duration=5 --profilingVerbosity=detailed \
    --dumpProfile --separateProfileRun \
    --exportProfile="$OUT/${LABEL}-${v}-profile.json" \
    --exportLayerInfo="$OUT/${LABEL}-${v}-layers.json" > "$OUT/${LABEL}-${v}-profile.log" 2>&1 \
    || { echo "!! [$v] 逐层 profile 失败 (不影响测速结果), 见 $OUT/${LABEL}-${v}-profile.log"; tail -15 "$OUT/${LABEL}-${v}-profile.log"; }
done
echo "trtexec 完成 -> $CSV"
