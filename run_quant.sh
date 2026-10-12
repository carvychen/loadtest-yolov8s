#!/usr/bin/env bash
# 量化实验一条命令 (二期): 自建 FP8/INT8 量化 + 精度评测 + 三层测速, 量化 "FP8 比 FP16 实际快多少、卡在哪"。
#   COCO 准备 -> ModelOpt 量化 -> 各精度建引擎 + trtexec 两层测速 + 逐层 profile
#   -> Triton 上测 mAP -> perf_analyzer 并发扫描 -> results/quant/<label>-summary.md
# 前置: 已跑过 setup.sh (有 yolov8s.onnx 和 Triton 镜像)。FP8 需 sm89+ (Ada/Hopper/Blackwell), 在 RTX PRO 6000 上跑。
set -euo pipefail
trap 'echo "!! run_quant.sh 第 $LINENO 行失败 (退出码 $?), 往上翻看具体报错; 各步骤日志在 results/quant/*.log"' ERR

LABEL="${1:?用法: bash run_quant.sh <gpu-label>   例如 rtx6000-quarter-nc24}"
TAG="${TRITON_TAG:-25.08}"
SERVER="nvcr.io/nvidia/tritonserver:${TAG}-py3"
SDK="nvcr.io/nvidia/tritonserver:${TAG}-py3-sdk"
QIMG="yolo-quant:modelopt-0.47.0"
# 变体名 = <精度>[-标签...]。量化标签 (改 Q/DQ ONNX): nohead = 检测头 model.22 不量化, conv = 只量化 Conv (不量化 Add);
# 构建标签 (同一 ONNX, 只改 trtexec 参数): opt5 = --builderOptimizationLevel=5。例: fp8-nohead, fp8-opt5, int8-nohead
export VARIANTS="${VARIANTS:-fp16 fp8 int8}"
OUT=results/quant
CALIB=datasets/calib_500.npy
mkdir -p "$OUT"

[ -f yolov8s.onnx ] || { echo "缺少 yolov8s.onnx, 先跑 bash setup.sh"; exit 1; }
cleanup() { docker rm -f triton-q >/dev/null 2>&1 || true; }
trap cleanup EXIT
qrun() { docker run --rm -v "$PWD":/work -w /work "$@"; }

echo "[1/7] 量化/评测镜像 $QIMG"
docker image inspect "$QIMG" >/dev/null 2>&1 || docker build -t "$QIMG" -f Dockerfile.quant .

echo "[2/7] COCO val2017 + 校准集 (500 张, 与 4500 张评测集不重叠)"
qrun "$QIMG" python fetch_coco.py
[ -f "$CALIB" ] || qrun "$QIMG" python prep_calib.py "$CALIB"

echo "[3/7] ModelOpt 量化 -> yolov8s.<variant>.onnx (Q/DQ ONNX 与 GPU 无关, 已存在则跳过)"
# FP16 基线: TensorRT 11 移除了弱类型 (--fp16), 需要先用 ModelOpt AutoCast 转成 FP16 ONNX (I/O 保持 FP32);
# TensorRT 10 仍用原 FP32 ONNX + --fp16 (与一期一致)。两份都备好, 由 bench_trtexec.sh 按 TRT 版本选。
if [ ! -f yolov8s.fp16.onnx ]; then
  [ -f datasets/autocast_calib.npz ] || qrun "$QIMG" python -c \
    "import numpy as np; a=np.load('$CALIB', mmap_mode='r'); np.savez('datasets/autocast_calib.npz', images=np.ascontiguousarray(a[:16]))"
  qrun "$QIMG" python -m modelopt.onnx.autocast --onnx_path=yolov8s.onnx --output_path=yolov8s.fp16.onnx \
    --low_precision_type=fp16 --keep_io_types --calibration_data=datasets/autocast_calib.npz \
    2>&1 | tee "$OUT/autocast-fp16.log"
  [ -f yolov8s.fp16.onnx ] || { echo "AutoCast 失败, 见 $OUT/autocast-fp16.log"; exit 1; }
fi
for v in $VARIANTS; do
  q="${v//-opt5/}"                                         # 构建标签不影响 ONNX
  [ "${q%%-*}" = fp16 ] && continue
  [ -f "yolov8s.${q}.onnx" ] && { echo "  已存在 yolov8s.${q}.onnx"; continue; }
  mode="${q%%-*}"                                          # fp8-nohead -> fp8
  # FP8 用 max (E4M3 动态范围大, NVIDIA 推荐); INT8 用 entropy (TensorRT 经典 CNN 口径)
  if [ "$mode" = fp8 ]; then method="${FP8_CALIB:-max}"; else method="${INT8_CALIB:-entropy}"; fi
  extra=()
  [[ "$q" == *-nohead* ]] && extra+=(--nodes_to_exclude '/model\.22/.*')
  [[ "$q" == *-conv* ]] && extra+=(--op_types_to_quantize Conv)
  echo "  [$q] quantize_mode=$mode calibration_method=$method ${extra[*]}"
  qrun "$QIMG" python -m modelopt.onnx.quantization \
    --onnx_path=yolov8s.onnx --quantize_mode="$mode" \
    --calibration_data_path="$CALIB" --calibration_method="$method" \
    --calibration_eps cpu --output_path="yolov8s.${q}.onnx" "${extra[@]}" \
    2>&1 | tee "$OUT/quant-${q}.log"
  [ -f "yolov8s.${q}.onnx" ] || { echo "[$q] 量化失败, 见 $OUT/quant-${q}.log"; exit 1; }
done

echo "[4/7] 建引擎 + trtexec 两层测速 + 逐层 profile"
qrun "$QIMG" python make_q_repo.py $VARIANTS
docker run --gpus all --rm -v "$PWD":/work -w /work \
  -e VARIANTS -e BATCHES -e PROFILE_BATCH -e DURATION \
  "$SERVER" bash bench_trtexec.sh "$LABEL"

echo "[5/7] 启动 Triton (加载全部精度变体)"
cleanup
docker run --gpus all -d --rm --name triton-q --net host \
  -v "$PWD/model_repository_q":/models "$SERVER" \
  tritonserver --model-repository=/models >/dev/null
echo -n "      等待 READY"
for i in $(seq 1 90); do
  if curl -sf localhost:8000/v2/health/ready >/dev/null 2>&1; then echo " ✓"; break; fi
  echo -n "."; sleep 2
  if [ "$i" = 90 ]; then echo " 超时"; docker logs triton-q | tail -40; exit 1; fi
done
for v in $VARIANTS; do
  curl -sf "localhost:8000/v2/models/yolov8s_${v}/ready" >/dev/null \
    || { echo "yolov8s_${v} 未就绪"; docker logs triton-q | tail -40; exit 1; }
done

if [ "${SKIP_MAP:-0}" != 1 ]; then
  echo "[6/7] COCO mAP (每个变体 4500 张)"
  for v in $VARIANTS; do
    docker run --rm --net host -v "$PWD":/work -w /work "$QIMG" \
      python eval_map.py --model "yolov8s_${v}" --out "$OUT/${LABEL}-${v}-map.json"
  done
else
  echo "[6/7] 跳过 mAP (SKIP_MAP=1)"
fi

if [ "${SKIP_SWEEP:-0}" != 1 ]; then
  echo "[7/7] perf_analyzer 并发扫描 (与一期同口径: 1→32, p99 SLA)"
  for v in $VARIANTS; do
    docker run --rm --net host -v "$PWD":/work -w /work \
      -e CONCURRENCY -e REQUESTS -e WARMUP -e MODEL="yolov8s_${v}" -e OUTDIR="$OUT" \
      "$SDK" bash run_sweep.sh "${LABEL}-${v}"
  done
else
  echo "[7/7] 跳过并发扫描 (SKIP_SWEEP=1)"
fi

cleanup
qrun "$QIMG" python analyze_quant.py --label "$LABEL" | tee "$OUT/${LABEL}-summary.md"
echo "完成 → $OUT/${LABEL}-summary.md"
