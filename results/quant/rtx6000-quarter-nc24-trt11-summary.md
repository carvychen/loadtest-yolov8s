# YOLOv8s 量化实验: rtx6000-quarter-nc24-trt11

环境: label=rtx6000-quarter-nc24-trt11 · NVIDIA RTX Pro 6000 Blackwell DC-1-24Q, 595.58.03, [Insufficient Permissions] · TensorRT v110300 · TensorRT version: 11.3.0 · TensorRT version: 11.3.0 · TensorRT version: 11.3.0 · TensorRT version: 11.3.0 · TensorRT version: 11.3.0 · TensorRT version: 11.3.0

## 1. 精度 (COCO val2017 评测集 4500 张, 与校准集不重叠)

| 变体 | mAP50-95 | ΔmAP50-95 vs FP16 | mAP50 |
|---|---|---|---|
| fp16 | 0.4513 | +0.00 pt | 0.6187 |
| fp8 | 0.4472 | -0.41 pt | 0.6157 |
| fp8-nohead | 0.4491 | -0.22 pt | 0.6168 |
| fp8-conv | 0.4474 | -0.39 pt | 0.6163 |
| fp8-opt5 | 0.4472 | -0.41 pt | 0.6157 |
| int8 | 0.4425 | -0.88 pt | 0.6142 |

## 2. 三层吞吐与加速比 (img/s; 括号内为相对 FP16)

| 变体 | GPU纯计算 b1 | GPU纯计算 b8 | GPU纯计算 b32 | 含传输 b1 | 含传输 b8 | 含传输 b32 | Triton QPS@p99≤50ms |
|---|---|---|---|---|---|---|---|
| fp16 | 877 (1.00×) | 1293 (1.00×) | 1128 (1.00×) | 890 (1.00×) | 1264 (1.00×) | 1100 (1.00×) | — |
| fp8 | 1092 (1.25×) | 2002 (1.55×) | 1690 (1.50×) | 1091 (1.23×) | 1944 (1.54×) | 1638 (1.49×) | — |
| fp8-nohead | 1041 (1.19×) | 1815 (1.40×) | 1516 (1.34×) | 1040 (1.17×) | 1767 (1.40×) | 1473 (1.34×) | — |
| fp8-conv | 1116 (1.27×) | 2017 (1.56×) | 1689 (1.50×) | 1120 (1.26×) | 1958 (1.55×) | 1637 (1.49×) | — |
| fp8-opt5 | 1115 (1.27×) | 2009 (1.55×) | 1689 (1.50×) | 1114 (1.25×) | 1951 (1.54×) | 1636 (1.49×) | — |
| int8 | 860 (0.98×) | 1332 (1.03×) | 1130 (1.00×) | 859 (0.97×) | 1302 (1.03×) | 1111 (1.01×) | — |

## 3. 逐层归因 (trtexec --dumpProfile, batch 见 PROFILE_BATCH, 默认 8)

| 变体 | 层数 | GPU 总耗时 ms | 低精度卷积占比 (个数) | 卷积按精度计数 | reformat 时间占比 | 未融合 Q/DQ 时间占比 |
|---|---|---|---|---|---|---|
| fp16 | 141 | 6.48 | 0% | {'FP16': 61} | 0.0% | 0.0% |
| fp8 | 136 | 4.35 | 97% | {'FP16': 2, 'FP8': 59} | 0.0% | 0.0% |
| fp8-nohead | 136 | 4.72 | 72% | {'FP16': 17, 'FP8': 44} | 0.0% | 0.0% |
| fp8-conv | 136 | 4.32 | 97% | {'FP16': 2, 'FP8': 59} | 0.0% | 0.0% |
| fp8-opt5 | 136 | 4.34 | 97% | {'FP16': 2, 'FP8': 59} | 0.0% | 0.0% |
| int8 | 155 | 6.40 | 100% | {'INT8': 61} | 0.0% | 0.0% |

<details><summary>fp16 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `__myl_Silu_myl0_3` | other | FP16 | 0.2795 |
| `/model.22/cv3.0/cv3.0.0/conv/Conv+/model.22/cv2.0/cv2.0.0/conv/Conv_myl0_119` | conv | FP16 | 0.2781 |
| `/model.1/conv/Conv_myl0_4` | conv | FP16 | 0.2364 |
| `__myl_SiluSlic_myl0_7` | other | FP16 | 0.2188 |
| `/model.0/conv/Conv_myl0_2` | conv | FP16 | 0.184 |
| `/model.22/cv3.0/cv3.0.1/conv/Conv_myl0_136` | conv | FP16 | 0.1695 |
| `/model.2/cv2/conv/Conv_myl0_13` | conv | FP16 | 0.1656 |
| `__myl_Cast_myl0_1` | other | FP32 | 0.1626 |
| `__myl_ConcSigmCastSlicSlicReshAddSubSubAddMulConcCastMulConc_myl0_140` | other | FP16 | 0.1593 |
| `/model.15/cv1/conv/Conv_myl0_78` | conv | FP16 | 0.1433 |

</details>

<details><summary>fp8 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `__myl_SiluMulCast_myl0_3` | other | FP16 | 0.2124 |
| `/model.0/conv/Conv_myl0_2` | conv | FP16 | 0.1837 |
| `/model.22/cv3.0/cv3.0.0/conv/Conv+/model.22/cv2.0/cv2.0.0/conv/Conv_myl0_97` | conv | FP8 | 0.1633 |
| `/model.1/conv/Conv_myl0_4` | conv | FP8 | 0.1632 |
| `__myl_ConcSigmSlicSlicReshAddSubSubAddMulConcMulConcCast_myl0_135` | other | FP16 | 0.1613 |
| `__myl_Cast_myl0_1` | other | FP32 | 0.1606 |
| `__myl_CastMulSiluMulCastCastMulAddMulCastConc_myl0_11` | other | FP8 | 0.1081 |
| `/model.2/cv2/conv/Conv_myl0_12` | conv | FP8 | 0.104 |
| `/model.2/m.0/cv1/conv/Conv_myl0_8` | conv | FP8 | 0.1008 |
| `/model.2/m.0/cv2/conv/Conv_myl0_10` | conv | FP8 | 0.0969 |

</details>

<details><summary>fp8-nohead 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `/model.22/cv3.0/cv3.0.0/conv/Conv+/model.22/cv2.0/cv2.0.0/conv/Conv_myl0_97` | conv | FP16 | 0.2791 |
| `__myl_SiluMulCast_myl0_3` | other | FP16 | 0.2126 |
| `/model.0/conv/Conv_myl0_2` | conv | FP16 | 0.1829 |
| `/model.22/cv3.0/cv3.0.1/conv/Conv_myl0_127` | conv | FP16 | 0.175 |
| `__myl_ConcSigmSlicSlicReshAddSubSubAddMulConcMulConcCast_myl0_135` | other | FP16 | 0.1634 |
| `__myl_Cast_myl0_1` | other | FP32 | 0.1602 |
| `/model.1/conv/Conv_myl0_4` | conv | FP8 | 0.1593 |
| `/model.22/cv3.1/cv3.1.0/conv/Conv+/model.22/cv2.1/cv2.1.0/conv/Conv_myl0_92` | conv | FP16 | 0.138 |
| `__myl_CastMulSiluMulCastCastMulAddMulCastConc_myl0_11` | other | FP8 | 0.1081 |
| `/model.2/cv2/conv/Conv_myl0_12` | conv | FP8 | 0.1021 |

</details>

<details><summary>fp8-conv 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `__myl_SiluMulCast_myl0_3` | other | FP16 | 0.2128 |
| `/model.0/conv/Conv_myl0_2` | conv | FP16 | 0.1857 |
| `__myl_ConcSigmSlicSlicReshAddSubSubAddMulConcMulConcCast_myl0_135` | other | FP16 | 0.1634 |
| `/model.1/conv/Conv_myl0_4` | conv | FP8 | 0.1612 |
| `__myl_Cast_myl0_1` | other | FP32 | 0.1591 |
| `/model.22/cv3.0/cv3.0.0/conv/Conv+/model.22/cv2.0/cv2.0.0/conv/Conv_myl0_97` | conv | FP8 | 0.1544 |
| `__myl_CastMulSiluAddMulCastConc_myl0_11` | other | FP8 | 0.1061 |
| `/model.2/m.0/cv1/conv/Conv_myl0_8` | conv | FP8 | 0.101 |
| `/model.2/cv2/conv/Conv_myl0_12` | conv | FP8 | 0.101 |
| `/model.2/m.0/cv2/conv/Conv_myl0_10` | conv | FP8 | 0.0978 |

</details>

<details><summary>fp8-opt5 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `__myl_SiluMulCast_myl0_3` | other | FP16 | 0.2105 |
| `/model.0/conv/Conv_myl0_2` | conv | FP16 | 0.1829 |
| `/model.1/conv/Conv_myl0_4` | conv | FP8 | 0.1633 |
| `__myl_ConcSigmSlicSlicReshAddSubSubAddMulConcMulConcCast_myl0_135` | other | FP16 | 0.1613 |
| `__myl_Cast_myl0_1` | other | FP32 | 0.1589 |
| `/model.22/cv3.0/cv3.0.0/conv/Conv+/model.22/cv2.0/cv2.0.0/conv/Conv_myl0_97` | conv | FP8 | 0.155 |
| `__myl_CastMulSiluMulCastCastMulAddMulCastConc_myl0_11` | other | FP8 | 0.1103 |
| `/model.2/cv2/conv/Conv_myl0_12` | conv | FP8 | 0.1039 |
| `/model.2/m.0/cv1/conv/Conv_myl0_8` | conv | FP8 | 0.1019 |
| `/model.2/m.0/cv2/conv/Conv_myl0_10` | conv | FP8 | 0.0968 |

</details>

<details><summary>int8 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `__myl_CastMulMinMaxRounReplConcReshMoveCast_myl0_1` | other | FP32 | 0.4782 |
| `/model.0/conv/Conv_myl0_2` | conv | INT8 | 0.4358 |
| `__myl_CastMulSiluMulMinMaxRounCastCastMulReshAddMulMinMaxRounCastConcReshMove_myl0_11` | other | INT8 | 0.275 |
| `__myl_SiluMulMinMaxRounCastReshSlicReshMoveSlic_myl0_7` | other | FP16 | 0.2381 |
| `__myl_SiluMulMinMaxRounCast_myl0_3` | other | FP16 | 0.2115 |
| `__myl_SiluMulMinMaxRounCastReshSlic_myl0_78` | other | FP16 | 0.1656 |
| `/model.22/cv3.0/cv3.0.0/conv/Conv+/model.22/cv2.0/cv2.0.0/conv/Conv_myl0_128` | conv | INT8 | 0.146 |
| `/model.1/conv/Conv_myl0_4` | conv | INT8 | 0.1347 |
| `__myl_ReshCastMulSiluMulMinMaxRounCastCastMulAddMulMinMaxRounCastReshConcReshMove_myl0_25` | other | INT8 | 0.1288 |
| `__myl_SiluMulMinMaxRounCastReshSlicReshMoveSlic_myl0_17` | other | FP16 | 0.1164 |

</details>

## 4. 瓶颈判读

- **fp8** (b8): GPU纯计算 1.55× → 含传输 1.54×
- **fp8-nohead** (b8): GPU纯计算 1.40× → 含传输 1.40×
  - GPU 内核层面加速已不足 1.5×: 瓶颈在量化覆盖 / kernel。低精度卷积占比 72%, reformat+未融合 Q/DQ 占 0.0% GPU 时间。
- **fp8-conv** (b8): GPU纯计算 1.56× → 含传输 1.55×
- **fp8-opt5** (b8): GPU纯计算 1.55× → 含传输 1.54×
- **int8** (b8): GPU纯计算 1.03× → 含传输 1.03×
  - GPU 内核层面加速已不足 1.5×: 瓶颈在量化覆盖 / kernel。低精度卷积占比 100%, reformat+未融合 Q/DQ 占 0.0% GPU 时间。
