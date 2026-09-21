# loadtest-yolov8s

YOLOv8s 在线推理性能 & 成本对比：**RTX PRO 6000 Blackwell 1/4 切片 vs T4 vs A10**（Azure VM）。
Triton + TensorRT，延迟敏感场景，最终折算每美元性能（QPS/$）用于选型。

## 三步跑完

```bash
git clone <this-repo> && cd loadtest-yolov8s
bash setup.sh                 # 一次性: 拉镜像 + 下权重 + 导 ONNX（+ 可选 COCO128）
bash run.sh <gpu-label>       # 一条命令: 建引擎 → 起 Triton → 并发扫描 → results/<label>.csv
```

三台机器各自 `run.sh` 用不同标签：`rtx6000-quarter` / `t4` / `a10`。
收齐三份 CSV 后，本机跑 `python analyze.py --sla-ms 50` 出对比报告。

## 目标设备

| 标签 | VM | SKU | 区域 | 月价 (USD) |
|---|---|---|---|---|
| `rtx6000-quarter` | rtx6000-quarter | `Standard_NC36lds_xl_RTXPRO6000BSE_v6` | West US 2 | 762.21 |
| `t4` | t4 | `Standard_NC16as_T4_v3` | Sweden Central | 待填 |
| `a10` | CN-SkillRouter-A10 | `Standard_NV36ads_A10_v5` | Sweden Central | 待填 |

> Azure 把 1/4 切片当成 `GPU × 1` 暴露，无需手动配 MIG，三台机器脚本完全相同。
> 省钱：`NC24lds`（$692.92）通常够用（推理 GPU-bound），比 `NC36lds` 每月省 ~$70。

## 压测数据

- **扫描默认用合成固定 shape 张量**（perf_analyzer 默认）——对纯延迟/吞吐这是标准做法，YOLOv8 前向 FLOPs 与像素内容无关。
- 检测 benchmark 的**标准参考数据集是 COCO val2017**（MLPerf Inference、Ultralytics 官方口径）。完整版 ~1GB 不入库；`setup.sh --coco128` 拉 Ultralytics 官方 ~7MB 子集，供可选真实图片端到端 / mAP 抽检。

## 前置要求（每台 GPU VM）

- Docker + NVIDIA Container Toolkit（`docker run --gpus all` 可用）
- 能访问 `nvcr.io`（NGC 公开镜像，无需登录即可 pull tritonserver）
- 进 A10（NV 系列）先 `nvidia-smi` 确认是标准数据中心驱动而非 vGPU 驱动

## 方法论要点

- 三卡同权重 / 同 640×640 / 同 **FP16** / 同 Triton 动态批处理配置。
- **TRT 引擎硬件绑定**：不可跨卡复用，每卡各自 `build_engine.sh` 构建。
- **在同一 Triton 容器内构建引擎并 serving**，避免 TRT 版本不匹配。
- 默认测模型前向（不烘焙 NMS）——对选型/成本，加速器前向吞吐才是三卡核心差异，NMS 近似 GPU 无关。

## 跑完清理（省钱）

```bash
az vm deallocate -g <rg> -n <vm>
```

## 可调参数（环境变量）

- `TRITON_TAG`（默认 `25.08`）：选一个 TensorRT 支持你最新 GPU 的 tag（Blackwell 需 2025+）。
- 分辨率 / batch 范围 / 并发范围 / SLA：见 `build_engine.sh`、`run_sweep.sh`、`analyze.py` 顶部。
