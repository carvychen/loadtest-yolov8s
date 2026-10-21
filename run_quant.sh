#!/usr/bin/env bash
# 量化实验一条命令: 自建 FP8/INT8 量化 + 精度评测 + 测速, 量化 "FP8 比 FP16 实际快多少、卡在哪"。
#   COCO 准备 -> ModelOpt 量化 -> 各精度建引擎 + trtexec 两层测速 + 逐层 profile
#   -> Triton 上测精度 -> (仅 YOLO) perf_analyzer 并发扫描 -> results/quant/<label>-summary.md
# MODEL=yolov8s (默认, 二期) / swin_t (三期, 精度看 FP8 与 FP16 输出一致性, 不做并发扫描), 见 quant/models.py。
# 前置: 已跑过 setup.sh (Triton 镜像; YOLO 还需 artifacts/yolov8s.onnx)。FP8 需 sm89+ (Ada/Hopper/Blackwell), 在 RTX PRO 6000 上跑。
set -euo pipefail
trap 'echo "!! run_quant.sh 第 $LINENO 行失败 (退出码 $?), 往上翻看具体报错; 各步骤日志在 results/quant/*.log"' ERR

LABEL="${1:?用法: bash run_quant.sh <gpu-label>   例如 rtx6000-quarter-nc24}"
TAG="${TRITON_TAG:-25.08}"
SERVER="nvcr.io/nvidia/tritonserver:${TAG}-py3"
SDK="nvcr.io/nvidia/tritonserver:${TAG}-py3-sdk"
QIMG="lt-quant:modelopt-0.47.0-timm"
MODEL="${MODEL:-yolov8s}"
# 变体名 = <精度>[-标签...], 例: fp8-nohead, fp8-opt5。规则全部在 quant/variants.py (量化标签改 ONNX, 构建标签只改 trtexec 参数)
VARIANTS="${VARIANTS:-fp16 fp8 int8}"
OUT=results/quant
mkdir -p "$OUT"

[ "$MODEL" != yolov8s ] || [ -f artifacts/yolov8s.onnx ] || { echo "缺少 artifacts/yolov8s.onnx, 先跑 bash setup.sh"; exit 1; }
cleanup() { docker rm -f triton-q >/dev/null 2>&1 || true; }
trap cleanup EXIT
qrun() { docker run --rm -v "$PWD":/work -w /work "$@"; }

echo "[1/7] 量化/评测镜像 $QIMG"
docker image inspect "$QIMG" >/dev/null 2>&1 || docker build -t "$QIMG" quant
qrun "$QIMG" python quant/variants.py check --model "$MODEL" $VARIANTS   # 变体名拼错几秒内报错, 不等到量化 / 建引擎时

echo "[2/7] COCO val2017 + 校准集 (500 张, 与 4500 张评测集不重叠)"
qrun "$QIMG" python quant/fetch_coco.py
# Swin: 官方 22K 权重 (缺则下载) -> FP32 ONNX, 导出后用 onnxruntime 对比 PyTorch 输出
[[ "$MODEL" != swin_t* ]] || qrun "$QIMG" python quant/swin.py export --model "$MODEL"
qrun "$QIMG" python quant/prep_calib.py --model "$MODEL"     # 已存在则跳过

echo "[3/7] ModelOpt 量化 -> artifacts/${MODEL}.<variant>.onnx + Triton 配置 (Q/DQ ONNX 与 GPU 无关, 已存在则跳过)"
qrun -e FP8_CALIB -e INT8_CALIB "$QIMG" python quant/prepare_variants.py --model "$MODEL" $VARIANTS

echo "[4/7] 建引擎 + trtexec 两层测速 + 逐层 profile"
docker run --gpus all --rm -v "$PWD":/work -w /work \
  -e BATCHES -e PROFILE_BATCH -e DURATION \
  "$SERVER" python3 quant/bench_trtexec.py --model "$MODEL" "$LABEL" $VARIANTS

# 构建失败的变体 (bench_trtexec.py 记在 <label>-build-failed.txt) 不加载、不评测
BUILT=""
for v in $VARIANTS; do grep -qx -- "$v" "$OUT/${LABEL}-build-failed.txt" 2>/dev/null || BUILT+=" $v"; done
[ -n "${BUILT// /}" ] || { echo "!! 所有变体都构建失败, 见 $OUT/${LABEL}-*-build.log"; exit 1; }
[ ! -s "$OUT/${LABEL}-build-failed.txt" ] || echo "!! 跳过构建失败的变体: $(tr '\n' ' ' < "$OUT/${LABEL}-build-failed.txt")"

echo "[5/7] 启动 Triton (加载全部精度变体)"
cleanup
# 只加载本次的变体: 模型目录里还留着别的模型 / 别的 TRT 版本建的引擎, 默认全量加载会因版本不符而起不来
LOADS=(); for v in $BUILT; do LOADS+=("--load-model=${MODEL}_${v}"); done
docker run --gpus all -d --rm --name triton-q --net host \
  -v "$PWD/artifacts/model_repository_q":/models "$SERVER" \
  tritonserver --model-repository=/models --model-control-mode=explicit "${LOADS[@]}" >/dev/null
echo -n "      等待 READY"
for i in $(seq 1 90); do
  if curl -sf localhost:8000/v2/health/ready >/dev/null 2>&1; then echo " ✓"; break; fi
  echo -n "."; sleep 2
  if [ "$i" = 90 ]; then echo " 超时"; docker logs triton-q | tail -40; exit 1; fi
done
for v in $BUILT; do
  curl -sf "localhost:8000/v2/models/${MODEL}_${v}/ready" >/dev/null \
    || { echo "${MODEL}_${v} 未就绪"; docker logs triton-q | tail -40; exit 1; }
done

if [ "${SKIP_MAP:-0}" = 1 ]; then
  echo "[6/7] 跳过精度评测 (SKIP_MAP=1)"
elif [ "$MODEL" = yolov8s ]; then
  echo "[6/7] COCO mAP (每个变体 4500 张)"
  for v in $BUILT; do
    docker run --rm --net host -v "$PWD":/work -w /work "$QIMG" \
      python quant/eval_map.py --model "yolov8s_${v}" --out "$OUT/${LABEL}-${v}-map.json"
  done
elif [[ "$BUILT " == *" fp16 "* ]]; then
  echo "[6/7] 输出一致性 (4500 张, 各变体 vs FP16)"
  others=(); for v in $BUILT; do [ "$v" = fp16 ] || others+=("$v"); done   # fp16 作参照放第一个
  docker run --rm --net host -v "$PWD":/work -w /work "$QIMG" \
    python quant/eval_agreement.py --label "$LABEL" --model "$MODEL" fp16 "${others[@]}"
else
  echo "[6/7] 跳过输出一致性 (VARIANTS 里没有 fp16 参照)"
fi

if [ "$MODEL" != yolov8s ]; then
  echo "[7/7] 跳过并发扫描 ($MODEL 只测 GPU 侧)"
elif [ "${SKIP_SWEEP:-0}" != 1 ]; then
  echo "[7/7] perf_analyzer 并发扫描 (与一期同口径: 1→32, p99 SLA)"
  for v in $BUILT; do
    docker run --rm --net host -v "$PWD":/work -w /work \
      -e CONCURRENCY -e REQUESTS -e WARMUP -e MODEL="yolov8s_${v}" -e OUTDIR="$OUT" \
      "$SDK" bash serving/run_sweep.sh "${LABEL}-${v}"
  done
else
  echo "[7/7] 跳过并发扫描 (SKIP_SWEEP=1)"
fi

cleanup
qrun "$QIMG" python quant/analyze_quant.py --model "$MODEL" --label "$LABEL" | tee "$OUT/${LABEL}-summary.md"
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
