# Swin-T 量化实验: rtx6000-quarter-nc24-trt11-swin-r2

环境: label=rtx6000-quarter-nc24-trt11-swin-r2 · NVIDIA RTX Pro 6000 Blackwell DC-1-24Q, 595.58.03, [Insufficient Permissions] · TensorRT v110300 · TensorRT version: 11.3.0

## 1. 精度: 与 FP16 引擎的输出一致性 (COCO val2017 4500 张, 与校准集不重叠)

| 变体 | top-1 一致率 | top-5 重合率 | logits 余弦 (均值 / 最小) |
|---|---|---|---|
| fp16 | — | — | — |
| fp8 | 88.00% | 90.59% | 0.98909 / 0.88165 |
| fp8-head | 87.80% | 90.40% | 0.98855 / 0.88144 |

## 2. 三层吞吐与加速比 (img/s; 括号内为相对 FP16)

| 变体 | GPU纯计算 b1 | GPU纯计算 b8 | GPU纯计算 b32 | 含传输 b1 | 含传输 b8 | 含传输 b32 | Triton QPS@p99≤50ms |
|---|---|---|---|---|---|---|---|
| fp16 | 930 (1.00×) | 3685 (1.00×) | 3322 (1.00×) | 941 (1.00×) | 3682 (1.00×) | 3299 (1.00×) | — |
| fp8 | 1212 (1.30×) | 4428 (1.20×) | 4233 (1.27×) | 1199 (1.27×) | 4418 (1.20×) | 4212 (1.28×) | — |
| fp8-head | 1312 (1.41×) | 4603 (1.25×) | 4266 (1.28×) | 1298 (1.38×) | 4593 (1.25×) | 4245 (1.29×) | — |

## 3. 逐层归因 (trtexec --dumpProfile, batch 见 PROFILE_BATCH, 默认 8)

| 变体 | 层数 | GPU 总耗时 ms | 低精度卷积占比 (个数) | 卷积按精度计数 | reformat 时间占比 | 未融合 Q/DQ 时间占比 |
|---|---|---|---|---|---|---|
| fp16 | 170 | 2.53 | 0% | {'FP16': 1} | 0.0% | 0.0% |
| fp8 | 174 | 2.23 | 0% | {'FP16': 1} | 0.0% | 0.0% |
| fp8-head | 175 | 2.15 | 0% | {'FP16': 1} | 0.0% | 0.0% |

<details><summary>fp16 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `/head/fc/Gemm_myl0_168` | other | FP16 | 0.1286 |
| `_gemm_mha_v2_myl0_24` | other | FP16 | 0.0668 |
| `__myl_Fc_myl0_27` | other | FP16 | 0.0654 |
| `__myl_Fc_myl0_14` | other | FP16 | 0.0646 |
| `/layers/layers.2/blocks/blocks.5/mlp/fc2/MatMul_myl0_137` | other | FP32 | 0.059 |
| `__myl_Fc_myl0_55` | other | FP16 | 0.0488 |
| `__myl_Fc_myl0_42` | other | FP16 | 0.0485 |
| `__myl_Fc_myl0_164` | other | FP16 | 0.0447 |
| `__myl_Fc_myl0_151` | other | FP16 | 0.0445 |
| `__mye180880+__mye180868+__mye180856_myl0_5` | other | FP16 | 0.0413 |

</details>

<details><summary>fp8 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `/head/fc/Gemm_myl0_172` | other | FP16 | 0.1409 |
| `_gemm_mha_v2_myl0_24` | other | FP16 | 0.0666 |
| `__mye187700+__mye187688+__mye187676_myl0_18` | other | FP8 | 0.0437 |
| `__mye187503+__mye187491+__mye187479_myl0_5` | other | FP8 | 0.0426 |
| `_gemm_mha_v2_myl0_53` | other | FP16 | 0.04 |
| `/layers/layers.0/blocks/blocks.0/mlp/fc1/MatMul_myl0_13` | other | FP8 | 0.0364 |
| `__myl_ReshCastMulReshAddReshCastMeanSubMulMean_myl0_16` | other | FP16 | 0.0361 |
| `_gemm_mha_v2_myl0_10` | other | FP8 | 0.036 |
| `/layers/layers.0/blocks/blocks.1/mlp/fc1/MatMul_myl0_27` | other | FP8 | 0.0356 |
| `__myl_DivCastErfCastAddMulMulMulCast_myl0_28` | other | FP16 | 0.0355 |

</details>

<details><summary>fp8-head 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `/head/fc/Gemm_myl0_173` | other | FP8 | 0.0672 |
| `_gemm_mha_v2_myl0_24` | other | FP16 | 0.0669 |
| `__mye187547+__mye187535+__mye187523_myl0_5` | other | FP8 | 0.0435 |
| `__mye187744+__mye187732+__mye187720_myl0_18` | other | FP8 | 0.0414 |
| `_gemm_mha_v2_myl0_53` | other | FP16 | 0.0401 |
| `__myl_ReshCastMulReshAddReshCastMeanSubMulMean_myl0_16` | other | FP16 | 0.0387 |
| `__myl_DivCastErfCastAddMulMulMulCast_myl0_14` | other | FP16 | 0.0355 |
| `__myl_DivCastErfCastAddMulMulMulCast_myl0_28` | other | FP16 | 0.0354 |
| `/layers/layers.0/blocks/blocks.1/mlp/fc1/MatMul_myl0_27` | other | FP8 | 0.0354 |
| `/layers/layers.0/blocks/blocks.0/mlp/fc1/MatMul_myl0_13` | other | FP8 | 0.0354 |

</details>

## 4. 瓶颈判读

- **fp8** (b8): GPU纯计算 1.20× → 含传输 1.20×
- **fp8-head** (b8): GPU纯计算 1.25× → 含传输 1.25×
