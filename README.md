# loadtest-yolov8s

YOLOv8s 在线推理性能 & 成本对比：**RTX PRO 6000 Blackwell 1/4 切片 vs T4 vs A10**（Azure VM）。
Triton + TensorRT，延迟敏感场景，最终折算每美元性能（QPS/$）用于选型。

## 三步跑完

```bash
git clone <this-repo> && cd loadtest-yolov8s
bash setup.sh                 # 一次性: 拉镜像 + 下权重 + 导 ONNX（+ 可选 COCO128）
bash run.sh <gpu-label>       # 一条命令: 建引擎 → 起 Triton → 并发扫描 → results/<label>.csv
```

三台机器各自 `run.sh` 用不同标签：`rtx6000-quarter-nc24` / `t4` / `a10`。
收齐三份 CSV 后，本机跑 `python analyze.py --sla-ms 50` 出对比报告。

## 目标设备

价格为 Azure 公开零售价（Consumption / 按需 / Linux / 非 Spot，查询于 2026-09）；QPS/$ 用 $/hr 计算。

| 标签 | SKU | GPU · 显存 | vCPU / 内存 | 区域 | 按需 $/hr | Spot $/hr |
|---|---|---|---|---|---|---|
| `rtx6000-quarter-nc24` | `Standard_NC24lds_xl_RTXPRO6000BSE_v6` | RTX PRO 6000 Blackwell 1/4 · **24 GB** GDDR7 | 24 vCPU / 72 GiB | West US 2 | **1.13** | ~0.21 |
| `rtx6000-quarter-nc36` | `Standard_NC36lds_xl_RTXPRO6000BSE_v6` | RTX PRO 6000 Blackwell 1/4 · **24 GB** GDDR7 | 36 vCPU / 72 GiB | West US 2 | **1.243** | 0.2297 |
| `t4` | `Standard_NC16as_T4_v3` | T4 · **16 GB** GDDR6 | 16 vCPU / 110 GiB | Sweden Central | **1.276** | 0.3619 |
| `a10` | `Standard_NV36ads_A10_v5` | A10 · **24 GB** GDDR6 | 36 vCPU / 440 GiB | Sweden Central | **4.160** | 0.7688 |

> 显存：RTX 1/4 切片与 A10 均 24 GB，T4 仅 16 GB。YOLOv8s 引擎 + batch≤32 激活远小于 16 GB，**三卡显存对本基准都不构成瓶颈**；此列供参考（换更大模型或更大 batch 时才受限）。

> **两行同一块 GPU 切片**：`rtx6000-quarter-nc24` 与 `-nc36` 是同一块 RTX PRO 6000 1/4 切片（`nvidia-smi` 确认为真 MIG 实例：MIG 已启用 / GI·CI / ~46 SM / 24 GB），只差主机 vCPU·内存·单价。实测只跑 NC24 一次；**GPU 吞吐与主机无关**，NC36 用同一份吞吐按 $1.243 折算 QPS/$（vCPU 更多不会降低 GPU-bound 吞吐，估值偏保守）。MIG 模式下 `nvidia-smi` 顶层 `GPU-Util` 显示 `N/A` 属正常，不代表空闲。
> RTX 1/4 切片单价（NC24 **$1.13** / NC36 $1.243/hr）与整块 T4（$1.276/hr）同一量级，是 QPS/$ 最可能拉开差距处；NC24 更便宜，QPS/$ 最优。
> `NV36ads_A10_v5` 是整块 A10 的大 VM（36 vCPU/440GB），$4.16/hr 偏贵；如需公平可用小切片档 `NV6/12/18ads_A10_v5`。
> 有 EA/预留折扣价时，改 `analyze.py` 顶部的 `PRICES_PER_HOUR` 即可。

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

## 内存安全 · 每档请求数（REQUESTS）

perf_analyzer 会把**一档并发内每个「请求+响应」都留在客户端内存不释放**。YOLOv8s 单请求张量 ≈ 7.4 MB（输入 `images[3,640,640]` FP32 4.7 MB + 输出 `output0[84,8400]` FP32 2.7 MB），但 gRPC 会同时保留**序列化 + 反序列化两份副本**加 protobuf 开销，**实测单请求实际吃 ~12 MB**（NC24 上 6000 个请求 = 5000+1000 预热就吃满了 70 Gi）。故单档峰值 ≈（`WARMUP` + `REQUESTS`）× 12 MB，**与并发范围无关**（每档是独立进程，跨档释放）。

内存小且 `Swap=0` 的机器（RTX 两台标称 72 GiB，`free -h` 实测 total 仅 **~70 Gi 可用**，扣固件/内核预留）必须据此压低 `REQUESTS`，否则 OOM killer 会直接杀掉进程（还会短暂冻住整台机器）。**样本量只影响 p99 估计的精度，不改变 p99 真值**——所以各机器用不同 `REQUESTS` 不影响横向公平；真正要对齐的是**并发范围**与**充分预热**（预热只要够到稳态即可，500/1000 都行，不必逐台相等）。

| 机器 | 可用内存 | 建议 | 单档峰值 |
|---|---|---|---|
| `rtx6000-quarter-nc24` / `-nc36` | ~70 Gi · Swap 0 | `REQUESTS=2500 WARMUP=1000` | ~42 GB（留 ~28 Gi 余量） |
| `t4` | ~108 Gi | `REQUESTS=5000 WARMUP=1000` | ~72 GB（留 ~36 Gi 余量） |
| `a10` | ~438 Gi | `REQUESTS=8000 WARMUP=1000` | ~108 GB（内存宽裕，已实测通过） |

```bash
# RTX（NC24/NC36，~70Gi·Swap0 → 压到 2500）
CONCURRENCY=1:32:1 REQUESTS=2500 WARMUP=1000 bash run.sh rtx6000-quarter-nc24
# T4（~108Gi → 5000）
CONCURRENCY=1:32:1 REQUESTS=5000 WARMUP=1000 bash run.sh t4
# A10（440GiB 宽裕，8000 已实测通过）
CONCURRENCY=1:32:1 REQUESTS=8000 WARMUP=1000 bash run.sh a10
```

> 之前按 7.4 MB/请求估算是错的：gRPC 双副本让**实际约 12 MB/请求**。5000 请求（+1000 预热）× 12 MB ≈ 70 GB 正好吃满 NC24（`free -h` 显示 available 只剩 50 Mi），这就是撞墙点；降到 2500 峰值 ~42 GB，留足余量，p99 尾部仍有 ~25 个样本，够用。跑之前 `watch -n0.1 free -h` 盯着最稳。

## 跑完清理（省钱）

```bash
az vm deallocate -g <rg> -n <vm>
```

## 可调参数（环境变量）

- `TRITON_TAG`（默认 `25.08`）：选一个 TensorRT 支持你最新 GPU 的 tag（Blackwell 需 2025+）。
- 分辨率 / batch 范围 / 并发范围 / SLA：见 `build_engine.sh`、`run_sweep.sh`、`analyze.py` 顶部。
