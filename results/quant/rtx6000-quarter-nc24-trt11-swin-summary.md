# Swin-T 量化实验: rtx6000-quarter-nc24-trt11-swin

环境: label=rtx6000-quarter-nc24-trt11-swin · NVIDIA RTX Pro 6000 Blackwell DC-1-24Q, 595.58.03, [Insufficient Permissions] · TensorRT v110300 · TensorRT version: 11.3.0

## 1. 精度: 与 FP16 引擎的输出一致性 (COCO val2017 4500 张, 与校准集不重叠)

| 变体 | top-1 一致率 | top-5 重合率 | logits 余弦 (均值 / 最小) |
|---|---|---|---|
| fp16 | — | — | — |
| fp8 | 87.91% | 90.65% | 0.98909 / 0.88178 |
| fp8-nomha | 90.84% | 92.62% | 0.99308 / 0.91210 |
| fp8-torch | 89.27% | 91.63% | 0.98978 / 0.87615 |

## 2. 三层吞吐与加速比 (img/s; 括号内为相对 FP16)

| 变体 | GPU纯计算 b1 | GPU纯计算 b8 | GPU纯计算 b32 | 含传输 b1 | 含传输 b8 | 含传输 b32 | Triton QPS@p99≤50ms |
|---|---|---|---|---|---|---|---|
| fp16 | 922 (1.00×) | 3668 (1.00×) | 3330 (1.00×) | 934 (1.00×) | 3664 (1.00×) | 3309 (1.00×) | — |
| fp8 | 1211 (1.31×) | 4428 (1.21×) | 4233 (1.27×) | 1199 (1.28×) | 4412 (1.20×) | 4205 (1.27×) | — |
| fp8-nomha | 1191 (1.29×) | 4357 (1.19×) | 4070 (1.22×) | 1190 (1.27×) | 4344 (1.19×) | 4041 (1.22×) | — |
| fp8-torch | 1080 (1.17×) | 3158 (0.86×) | 2476 (0.74×) | 1070 (1.15×) | 3157 (0.86×) | 2469 (0.75×) | — |

## 3. 逐层归因 (trtexec --dumpProfile, batch 见 PROFILE_BATCH, 默认 8)

| 变体 | 层数 | GPU 总耗时 ms | 低精度卷积占比 (个数) | 卷积按精度计数 | reformat 时间占比 | 未融合 Q/DQ 时间占比 |
|---|---|---|---|---|---|---|
| fp16 | 170 | 2.56 | 0% | {'FP16': 1} | 0.0% | 0.0% |
| fp8 | 174 | 2.17 | 0% | {'FP16': 1} | 0.0% | 0.0% |
| fp8-nomha | 181 | 2.31 | 0% | {'FP16': 1} | 0.0% | 0.0% |
| fp8-torch | 218 | 3.11 | 0% | {'FP16': 1} | 0.0% | 0.0% |

<details><summary>fp16 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `/head/fc/Gemm_myl0_168` | other | FP16 | 0.1266 |
| `/layers/layers.2/blocks/blocks.5/mlp/fc2/MatMul_myl0_137` | other | FP32 | 0.0674 |
| `__myl_Fc_myl0_14` | other | FP16 | 0.0671 |
| `_gemm_mha_v2_myl0_24` | other | FP16 | 0.0648 |
| `__myl_Fc_myl0_27` | other | FP16 | 0.0642 |
| `__myl_Fc_myl0_55` | other | FP16 | 0.0489 |
| `__myl_Fc_myl0_42` | other | FP16 | 0.0481 |
| `__myl_Fc_myl0_151` | other | FP16 | 0.0443 |
| `__mye180880+__mye180868+__mye180856_myl0_5` | other | FP16 | 0.0427 |
| `__myl_Fc_myl0_164` | other | FP16 | 0.0426 |

</details>

<details><summary>fp8 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `/head/fc/Gemm_myl0_172` | other | FP16 | 0.1409 |
| `_gemm_mha_v2_myl0_24` | other | FP16 | 0.0679 |
| `__mye187503+__mye187491+__mye187479_myl0_5` | other | FP8 | 0.044 |
| `__mye187700+__mye187688+__mye187676_myl0_18` | other | FP8 | 0.0398 |
| `_gemm_mha_v2_myl0_53` | other | FP16 | 0.0395 |
| `__myl_ReshCastMulReshAddReshCastMeanSubMulMean_myl0_16` | other | FP16 | 0.038 |
| `/layers/layers.0/blocks/blocks.1/mlp/fc1/MatMul_myl0_27` | other | FP8 | 0.0359 |
| `_gemm_mha_v2_myl0_10` | other | FP8 | 0.0349 |
| `__myl_DivCastErfCastAddMulMulMulCast_myl0_28` | other | FP16 | 0.0348 |
| `__myl_DivCastErfCastAddMulMulMulCast_myl0_14` | other | FP16 | 0.0347 |

</details>

<details><summary>fp8-nomha 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `/head/fc/Gemm_myl0_179` | other | FP16 | 0.1409 |
| `_gemm_mha_v2_myl0_25` | other | FP16 | 0.0665 |
| `__mye186637+__mye186625+__mye186613_myl0_5` | other | FP8 | 0.042 |
| `_gemm_mha_v2_myl0_11` | other | FP16 | 0.0402 |
| `_gemm_mha_v2_myl0_55` | other | FP16 | 0.04 |
| `__mye186834+__mye186822+__mye186810_myl0_19` | other | FP8 | 0.0395 |
| `__myl_DivCastErfCastAddMulMulMulCast_myl0_29` | other | FP16 | 0.0364 |
| `__myl_DivCastErfCastAddMulMulMulCast_myl0_15` | other | FP16 | 0.0356 |
| `/layers/layers.0/blocks/blocks.1/mlp/fc1/MatMul_myl0_28` | other | FP8 | 0.0356 |
| `/layers/layers.0/blocks/blocks.0/mlp/fc1/MatMul_myl0_14` | other | FP8 | 0.0354 |

</details>

<details><summary>fp8-torch 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `/layers/layers.0/blocks/blocks.1/attn/MatMul_1_myl0_30` | other | FP16 | 0.0783 |
| `/layers/layers.0/blocks/blocks.1/attn/MatMul_myl0_28` | other | FP16 | 0.0777 |
| `/layers/layers.0/blocks/blocks.0/attn/MatMul_myl0_11` | other | FP16 | 0.0759 |
| `/layers/layers.0/blocks/blocks.0/attn/MatMul_1_myl0_13` | other | FP16 | 0.0756 |
| `__myl_AddReshAddReshMaxrSubExpSumCastDivCastMulCastMulCastReshCastMul_myl0_29` | other | FP16 | 0.0669 |
| `/head/fc/Gemm_myl0_216` | other | FP8 | 0.0631 |
| `__myl_AddMaxrSubExpSumCastDivCastMulCastMulCastReshCastMul_myl0_12` | other | FP16 | 0.0618 |
| `/layers/layers.1/blocks/blocks.1/attn/MatMul_myl0_64` | other | FP16 | 0.0402 |
| `/layers/layers.1/blocks/blocks.0/attn/MatMul_myl0_47` | other | FP16 | 0.0399 |
| `__mye185580+__mye185568+__mye185556_myl0_22` | other | FP8 | 0.0372 |

</details>

## 4. 瓶颈判读

- **fp8** (b8): GPU纯计算 1.21× → 含传输 1.20×
- **fp8-nomha** (b8): GPU纯计算 1.19× → 含传输 1.19×
- **fp8-torch** (b8): GPU纯计算 0.86× → 含传输 0.86×
