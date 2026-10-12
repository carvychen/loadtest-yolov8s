# YOLOv8s 量化实验: rtx6000-quarter-nc24

环境: label=rtx6000-quarter-nc24 · NVIDIA RTX Pro 6000 Blackwell DC-1-24Q, 595.58.03, [Insufficient Permissions] · TensorRT v101302 · TensorRT version: 10.13.2 · TensorRT version: 10.13.2 · TensorRT version: 10.13.2

## 1. 精度 (COCO val2017 评测集 4500 张, 与校准集不重叠)

| 变体 | mAP50-95 | ΔmAP50-95 vs FP16 | mAP50 |
|---|---|---|---|
| fp16 | 0.4515 | +0.00 pt | 0.6187 |
| fp8 | 0.4468 | -0.47 pt | 0.6159 |
| int8 | 0.4417 | -0.98 pt | 0.6136 |

## 2. 三层吞吐与加速比 (img/s; 括号内为相对 FP16)

| 变体 | GPU纯计算 b1 | GPU纯计算 b8 | GPU纯计算 b32 | 含传输 b1 | 含传输 b8 | 含传输 b32 | Triton QPS@p99≤50ms |
|---|---|---|---|---|---|---|---|
| fp16 | 858 (1.00×) | 1275 (1.00×) | 1208 (1.00×) | 858 (1.00×) | 1249 (1.00×) | 1178 (1.00×) | 309.9 (1.00×) |
| fp8 | 1071 (1.25×) | 1757 (1.38×) | 1498 (1.24×) | 1071 (1.25×) | 1707 (1.37×) | 1456 (1.24×) | 311.5 (1.01×) |
| int8 | 1373 (1.60×) | 2389 (1.87×) | 2144 (1.77×) | 1380 (1.61×) | 2357 (1.89×) | 2082 (1.77×) | 355.5 (1.15×) |

## 3. 逐层归因 (trtexec --dumpProfile, batch 见 PROFILE_BATCH, 默认 8)

| 变体 | 层数 | GPU 总耗时 ms | 低精度卷积占比 (个数) | 卷积按精度计数 | reformat 时间占比 | 未融合 Q/DQ 时间占比 |
|---|---|---|---|---|---|---|
| fp16 | 146 | 6.68 | 0% | {'FP16': 61} | 16.8% | 0.0% |
| fp8 | 193 | 6.17 | 97% | {'FP16': 2, 'FP8': 59} | 0.6% | 0.0% |
| int8 | 153 | 3.79 | 100% | {'INT8': 64} | 25.2% | 1.0% |

<details><summary>fp16 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `PWN(PWN(/model.0/act/Sigmoid), PWN(/model.0/act/Mul))` | other | FP16 | 0.2878 |
| `/model.22/cv3.0/cv3.0.0/conv/Conv || /model.22/cv2.0/cv2.0.0/conv/Conv` | conv | FP16 | 0.2728 |
| `/model.1/conv/Conv + PWN(PWN(/model.1/act/Sigmoid), PWN(/model.1/act/Mul))` | conv | FP16 | 0.2506 |
| `/model.0/conv/Conv` | conv | FP16 | 0.2183 |
| `/model.22/cv3.0/cv3.0.1/conv/Conv + PWN(PWN(/model.22/cv3.0/cv3.0.1/act/Sigmoid), PWN(/mod` | conv | FP16 | 0.2006 |
| `Reformatting CopyNode for Input Tensor 0 to /model.0/conv/Conv` | reformat | FP32 | 0.1982 |
| `/model.2/cv2/conv/Conv + PWN(PWN(/model.2/cv2/act/Sigmoid), PWN(/model.2/cv2/act/Mul))` | conv | FP16 | 0.1767 |
| `/model.15/cv1/conv/Conv + PWN(PWN(/model.15/cv1/act/Sigmoid), PWN(/model.15/cv1/act/Mul))` | conv | FP16 | 0.1716 |
| `/model.3/conv/Conv + PWN(PWN(/model.3/act/Sigmoid), PWN(/model.3/act/Mul))` | conv | FP16 | 0.1484 |
| `/model.2/cv1/conv/Conv + PWN(PWN(/model.2/cv1/act/Sigmoid), PWN(/model.2/cv1/act/Mul))` | conv | FP16 | 0.1404 |

</details>

<details><summary>fp8 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `__myl_SiluIotaMulFlooMaxMinCastReshReplReshReshReplReshConcMulCastGathMulCastConc_myl8_21` | other | FP16 | 0.3285 |
| `__myl_SiluMulCast_myl0_3` | other | FP16 | 0.2142 |
| `/model_0/conv/Conv_myl0_2` | conv | FP16 | 0.1998 |
| `/model_22/cv3_0/cv3_0_0/conv/Conv+/model_22/cv2_0/cv2_0_0/conv/Conv_myl8_38` | conv | FP8 | 0.1827 |
| `__myl_ReplReplReplConcReshIotaCastCastIotaCastCastIotaCastCastAddReshReplReshReshReplReshC` | other | FP16 | 0.1782 |
| `__myl_Cast_myl0_1` | other | FP32 | 0.1671 |
| `/model_18/m_0/cv1/conv/Conv_myl8_64` | conv | FP8 | 0.1569 |
| `/model_1/conv/Conv_myl0_4` | conv | FP8 | 0.1485 |
| `__myl_SiluIotaMulFlooMaxMinCastReshReplReshReshReplReshConcMulCastGathMulCastConc_myl8_7` | other | FP16 | 0.1467 |
| `__myl_Slic_myl8_66` | other | FP8 | 0.1387 |

</details>

<details><summary>int8 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `images_QuantizeLinear` | reformat | FP32 | 0.1562 |
| `model.0.conv.weight + model.0.conv.weight_QuantizeLinear + /model.0/conv/Conv + PWN(/model` | conv | INT8 | 0.1319 |
| `model.1.conv.weight + model.1.conv.weight_QuantizeLinear + /model.1/conv/Conv + PWN(/model` | conv | INT8 | 0.1129 |
| `model.22.cv3.0.0.conv.weight + model.22.cv3.0.0.conv.weight_QuantizeLinear + /model.22/cv3` | conv | INT8 | 0.1004 |
| `model.22.cv3.0.1.conv.weight + model.22.cv3.0.1.conv.weight_QuantizeLinear + /model.22/cv3` | conv | INT8 | 0.0987 |
| `__myl_ReplReplReplConcReshIotaCastCastIotaCastCastIotaCastCastAddReshReplReshReshReplReshC` | other | FP16 | 0.0969 |
| `model.2.cv2.conv.weight + model.2.cv2.conv.weight_QuantizeLinear + /model.2/cv2/conv/Conv ` | conv | INT8 | 0.0702 |
| `model.3.conv.weight + model.3.conv.weight_QuantizeLinear + /model.3/conv/Conv + PWN(/model` | conv | INT8 | 0.0648 |
| `model.2.m.0.cv1.conv.weight + model.2.m.0.cv1.conv.weight_QuantizeLinear + /model.2/m.0/cv` | conv | INT8 | 0.0616 |
| `model.2.m.0.cv2.conv.weight + model.2.m.0.cv2.conv.weight_QuantizeLinear + /model.2/m.0/cv` | conv | INT8 | 0.0615 |

</details>

## 4. 瓶颈判读

- **fp8** (b8): GPU纯计算 1.38× → 含传输 1.37× → Triton 端到端 1.01×
  - GPU 内核层面加速已不足 1.5×: 瓶颈在量化覆盖 / kernel。低精度卷积占比 97%, reformat+未融合 Q/DQ 占 0.6% GPU 时间。
  - 端到端收益明显小于内核收益: 量化省下的 GPU 时间被 GPU 外开销 (FP32 I/O 搬运 / gRPC / 批处理排队) 稀释。
- **int8** (b8): GPU纯计算 1.87× → 含传输 1.89× → Triton 端到端 1.15×
  - 端到端收益明显小于内核收益: 量化省下的 GPU 时间被 GPU 外开销 (FP32 I/O 搬运 / gRPC / 批处理排队) 稀释。
