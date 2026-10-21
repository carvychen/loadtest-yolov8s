# swin_t_tanh 量化实验: rtx6000-quarter-nc24-trt11-swin-tanh

环境: label=rtx6000-quarter-nc24-trt11-swin-tanh · NVIDIA RTX Pro 6000 Blackwell DC-1-24Q, 595.58.03, [Insufficient Permissions] · TensorRT v110300 · TensorRT version: 11.3.0

## 1. 精度: 与 FP16 引擎的输出一致性 (COCO val2017 4500 张, 与校准集不重叠)

| 变体 | top-1 一致率 | top-5 重合率 | logits 余弦 (均值 / 最小) |
|---|---|---|---|
| fp16 | — | — | — |
| fp8 | 88.00% | 90.48% | 0.98914 / 0.86484 |
| fp8-head | 88.00% | 90.39% | 0.98859 / 0.86467 |

## 2. 三层吞吐与加速比 (img/s; 括号内为相对 FP16)

| 变体 | GPU纯计算 b1 | GPU纯计算 b8 | GPU纯计算 b32 | 含传输 b1 | 含传输 b8 | 含传输 b32 | Triton QPS@p99≤50ms |
|---|---|---|---|---|---|---|---|
| fp16 | 931 (1.00×) | 3422 (1.00×) | 2892 (1.00×) | 940 (1.00×) | 3416 (1.00×) | 2875 (1.00×) | — |
| fp8 | 1247 (1.34×) | 4843 (1.42×) | 5118 (1.77×) | 1233 (1.31×) | 4836 (1.42×) | 5074 (1.76×) | — |
| fp8-head | 1345 (1.44×) | 5049 (1.48×) | 5175 (1.79×) | 1331 (1.42×) | 5042 (1.48×) | 5130 (1.78×) | — |

## 3. 逐层归因 (trtexec --dumpProfile, batch 见 PROFILE_BATCH, 默认 8)

| 变体 | 层数 | GPU 总耗时 ms | 低精度卷积占比 (个数) | 卷积按精度计数 | reformat 时间占比 | 未融合 Q/DQ 时间占比 |
|---|---|---|---|---|---|---|
| fp16 | 180 | 2.70 | 0% | {'FP16': 1} | 0.0% | 0.0% |
| fp8 | 162 | 1.96 | 0% | {'FP16': 1} | 0.0% | 0.0% |
| fp8-head | 163 | 1.91 | 0% | {'FP16': 1} | 0.0% | 0.0% |

<details><summary>fp16 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `/head/fc/Gemm_myl0_178` | other | FP16 | 0.1348 |
| `_gemm_mha_v2_myl0_25` | other | FP16 | 0.065 |
| `/layers/layers.2/blocks/blocks.5/mlp/fc2/MatMul_myl0_145` | other | FP32 | 0.0599 |
| `__myl_MulCastCastMulMulCastAddMulReshTanhAddMul_myl0_15` | other | FP16 | 0.056 |
| `__myl_MulCastCastMulMulCastAddMulReshTanhAddMul_myl0_29` | other | FP16 | 0.055 |
| `/layers/layers.0/blocks/blocks.0/mlp/fc2/MatMul_myl0_16` | other | FP16 | 0.0523 |
| `/layers/layers.0/blocks/blocks.1/mlp/fc2/MatMul_myl0_30` | other | FP16 | 0.0505 |
| `/layers/layers.0/blocks/blocks.1/mlp/fc1/MatMul_myl0_28` | other | FP16 | 0.0489 |
| `/layers/layers.0/blocks/blocks.0/mlp/fc1/MatMul_myl0_14` | other | FP16 | 0.0485 |
| `__mye181231+__mye181219+__mye181207_myl0_5` | other | FP16 | 0.0432 |

</details>

<details><summary>fp8 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `/head/fc/Gemm_myl0_160` | other | FP16 | 0.143 |
| `_gemm_mha_v2_myl0_23` | other | FP16 | 0.0681 |
| `/layers/layers.0/blocks/blocks.0/mlp/fc1/MatMul_myl0_13` | other | FP8 | 0.0466 |
| `/layers/layers.0/blocks/blocks.1/mlp/fc1/MatMul_myl0_26` | other | FP8 | 0.0465 |
| `__mye187767+__mye187755+__mye187743_myl0_5` | other | FP8 | 0.0446 |
| `__mye187964+__mye187952+__mye187940_myl0_17` | other | FP8 | 0.0403 |
| `_gemm_mha_v2_myl0_50` | other | FP16 | 0.0393 |
| `_gemm_mha_v2_myl0_10` | other | FP8 | 0.0357 |
| `/layers/layers.1/blocks/blocks.1/mlp/fc1/MatMul_myl0_53` | other | FP8 | 0.0319 |
| `/layers/layers.1/blocks/blocks.0/mlp/fc1/MatMul_myl0_40` | other | FP8 | 0.0295 |

</details>

<details><summary>fp8-head 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `/head/fc/Gemm_myl0_161` | other | FP8 | 0.0692 |
| `_gemm_mha_v2_myl0_23` | other | FP16 | 0.0679 |
| `/layers/layers.0/blocks/blocks.0/mlp/fc1/MatMul_myl0_13` | other | FP8 | 0.0462 |
| `/layers/layers.0/blocks/blocks.1/mlp/fc1/MatMul_myl0_26` | other | FP8 | 0.0459 |
| `__mye187811+__mye187799+__mye187787_myl0_5` | other | FP8 | 0.044 |
| `__mye188008+__mye187996+__mye187984_myl0_17` | other | FP8 | 0.04 |
| `_gemm_mha_v2_myl0_50` | other | FP16 | 0.0393 |
| `_gemm_mha_v2_myl0_10` | other | FP8 | 0.0357 |
| `/layers/layers.1/blocks/blocks.1/mlp/fc1/MatMul_myl0_53` | other | FP8 | 0.0324 |
| `/layers/layers.1/blocks/blocks.0/mlp/fc1/MatMul_myl0_40` | other | FP8 | 0.0303 |

</details>

## 4. 瓶颈判读

- **fp8** (b8): GPU纯计算 1.42× → 含传输 1.42×
- **fp8-head** (b8): GPU纯计算 1.48× → 含传输 1.48×
