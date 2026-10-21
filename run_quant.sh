#!/usr/bin/env bash
# 量化实验一条命令 (二期): 自建 FP8/INT8 量化 + 精度评测 + 三层测速, 量化 "FP8 比 FP16 实际快多少、卡在哪"。
#   COCO 准备 -> ModelOpt 量化 -> 各精度建引擎 + trtexec 两层测速 + 逐层 profile
#   -> Triton 上测 mAP -> perf_analyzer 并发扫描 -> results/quant/<label>-summary.md
# 前置: 已跑过 setup.sh (有 artifacts/yolov8s.onnx 和 Triton 镜像)。FP8 需 sm89+ (Ada/Hopper/Blackwell), 在 RTX PRO 6000 上跑。
set -euo pipefail
trap 'echo "!! run_quant.sh 第 $LINENO 行失败 (退出码 $?), 往上翻看具体报错; 各步骤日志在 results/quant/*.log"' ERR

LABEL="${1:?用法: bash run_quant.sh <gpu-label>   例如 rtx6000-quarter-nc24}"
TAG="${TRITON_TAG:-25.08}"
SERVER="nvcr.io/nvidia/tritonserver:${TAG}-py3"
SDK="nvcr.io/nvidia/tritonserver:${TAG}-py3-sdk"
QIMG="yolo-quant:modelopt-0.47.0"
# 变体名 = <精度>[-标签...], 例: fp8-nohead, fp8-opt5。规则全部在 quant/variants.py (量化标签改 ONNX, 构建标签只改 trtexec 参数)
VARIANTS="${VARIANTS:-fp16 fp8 int8}"
OUT=results/quant
CALIB=datasets/calib_500.npy
mkdir -p "$OUT"

[ -f artifacts/yolov8s.onnx ] || { echo "缺少 artifacts/yolov8s.onnx, 先跑 bash setup.sh"; exit 1; }
cleanup() { docker rm -f triton-q >/dev/null 2>&1 || true; }
trap cleanup EXIT
qrun() { docker run --rm -v "$PWD":/work -w /work "$@"; }

echo "[1/7] 量化/评测镜像 $QIMG"
docker image inspect "$QIMG" >/dev/null 2>&1 || docker build -t "$QIMG" quant
qrun "$QIMG" python quant/variants.py check $VARIANTS        # 变体名拼错几秒内报错, 不等到量化 / 建引擎时

echo "[2/7] COCO val2017 + 校准集 (500 张, 与 4500 张评测集不重叠)"
qrun "$QIMG" python quant/fetch_coco.py
[ -f "$CALIB" ] || qrun "$QIMG" python quant/prep_calib.py "$CALIB"

echo "[3/7] ModelOpt 量化 -> artifacts/yolov8s.<variant>.onnx + Triton 配置 (Q/DQ ONNX 与 GPU 无关, 已存在则跳过)"
qrun -e FP8_CALIB -e INT8_CALIB "$QIMG" python quant/prepare_variants.py $VARIANTS

echo "[4/7] 建引擎 + trtexec 两层测速 + 逐层 profile"
docker run --gpus all --rm -v "$PWD":/work -w /work \
  -e BATCHES -e PROFILE_BATCH -e DURATION \
  "$SERVER" python3 quant/bench_trtexec.py "$LABEL" $VARIANTS

echo "[5/7] 启动 Triton (加载全部精度变体)"
cleanup
docker run --gpus all -d --rm --name triton-q --net host \
  -v "$PWD/artifacts/model_repository_q":/models "$SERVER" \
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
      python quant/eval_map.py --model "yolov8s_${v}" --out "$OUT/${LABEL}-${v}-map.json"
  done
else
  echo "[6/7] 跳过 mAP (SKIP_MAP=1)"
fi

if [ "${SKIP_SWEEP:-0}" != 1 ]; then
  echo "[7/7] perf_analyzer 并发扫描 (与一期同口径: 1→32, p99 SLA)"
  for v in $VARIANTS; do
    docker run --rm --net host -v "$PWD":/work -w /work \
      -e CONCURRENCY -e REQUESTS -e WARMUP -e MODEL="yolov8s_${v}" -e OUTDIR="$OUT" \
      "$SDK" bash serving/run_sweep.sh "${LABEL}-${v}"
  done
else
  echo "[7/7] 跳过并发扫描 (SKIP_SWEEP=1)"
fi

cleanup
qrun "$QIMG" python quant/analyze_quant.py --label "$LABEL" | tee "$OUT/${LABEL}-summary.md"
echo "完成 → $OUT/${LABEL}-summary.md"

if [ "${PUSH_RESULTS:-0}" = 1 ]; then   # 结果直接经 GitHub 回传 (VM 需有仓库写权限, 如 deploy key)
  # pathspec 加引号由 git 展开: 只收本 label 的产物, .gitignore 忽略的 .log 自动跳过
  git add -- "$OUT/${LABEL}-*"
  if git diff --cached --quiet; then
    echo "没有新结果可提交"
  else
    git commit -qm "结果: ${LABEL}" && git pull -q --rebase && git push -q && echo "已推送 → $(git rev-parse --short HEAD)"
  fi
fi
