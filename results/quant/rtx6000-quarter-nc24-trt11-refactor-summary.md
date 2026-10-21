# YOLOv8s 量化实验: rtx6000-quarter-nc24-trt11-refactor

环境: label=rtx6000-quarter-nc24-trt11-refactor · NVIDIA RTX Pro 6000 Blackwell DC-1-24Q, 595.58.03, [Insufficient Permissions] · TensorRT v110300 · TensorRT version: 11.3.0

## 1. 精度 (COCO val2017 评测集 4500 张, 与校准集不重叠)

| 变体 | mAP50-95 | ΔmAP50-95 vs FP16 | mAP50 |
|---|---|---|---|
| fp16 | 0.4516 | +0.00 pt | 0.6189 |
| fp8 | 0.4472 | -0.44 pt | 0.6157 |

## 2. 三层吞吐与加速比 (img/s; 括号内为相对 FP16)

| 变体 | GPU纯计算 b1 | GPU纯计算 b8 | GPU纯计算 b32 | 含传输 b1 | 含传输 b8 | 含传输 b32 | Triton QPS@p99≤50ms |
|---|---|---|---|---|---|---|---|
| fp16 | 843 (1.00×) | 1286 (1.00×) | 1124 (1.00×) | 855 (1.00×) | 1256 (1.00×) | 1096 (1.00×) | — |
| fp8 | 1063 (1.26×) | 1981 (1.54×) | 1683 (1.50×) | 1061 (1.24×) | 1926 (1.53×) | 1631 (1.49×) | — |

## 3. 逐层归因 (trtexec --dumpProfile, batch 见 PROFILE_BATCH, 默认 8)

| 变体 | 层数 | GPU 总耗时 ms | 低精度卷积占比 (个数) | 卷积按精度计数 | reformat 时间占比 | 未融合 Q/DQ 时间占比 |
|---|---|---|---|---|---|---|
| fp16 | 141 | 6.50 | 0% | {'FP16': 61} | 0.0% | 0.0% |
| fp8 | 136 | 4.33 | 97% | {'FP16': 2, 'FP8': 59} | 0.0% | 0.0% |

<details><summary>fp16 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `__myl_Silu_myl0_3` | other | FP16 | 0.2862 |
| `/model.22/cv3.0/cv3.0.0/conv/Conv+/model.22/cv2.0/cv2.0.0/conv/Conv_myl0_119` | conv | FP16 | 0.278 |
| `/model.1/conv/Conv_myl0_4` | conv | FP16 | 0.2351 |
| `__myl_SiluSlic_myl0_7` | other | FP16 | 0.2088 |
| `/model.0/conv/Conv_myl0_2` | conv | FP16 | 0.1819 |
| `/model.22/cv3.0/cv3.0.1/conv/Conv_myl0_136` | conv | FP16 | 0.1706 |
| `/model.2/cv2/conv/Conv_myl0_13` | conv | FP16 | 0.1676 |
| `/model.15/cv1/conv/Conv_myl0_78` | conv | FP16 | 0.1629 |
| `__myl_ConcSigmCastSlicSlicReshAddSubSubAddMulConcCastMulConc_myl0_140` | other | FP16 | 0.1614 |
| `__myl_Cast_myl0_1` | other | FP32 | 0.1605 |

</details>

<details><summary>fp8 最耗时 10 层</summary>

| 层 | 类型 | 精度 | ms |
|---|---|---|---|
| `__myl_SiluMulCast_myl0_3` | other | FP16 | 0.2105 |
| `/model.0/conv/Conv_myl0_2` | conv | FP16 | 0.1812 |
| `__myl_ConcSigmSlicSlicReshAddSubSubAddMulConcMulConcCast_myl0_135` | other | FP16 | 0.1634 |
| `/model.22/cv3.0/cv3.0.0/conv/Conv+/model.22/cv2.0/cv2.0.0/conv/Conv_myl0_97` | conv | FP8 | 0.1628 |
| `/model.1/conv/Conv_myl0_4` | conv | FP8 | 0.1624 |
| `__myl_Cast_myl0_1` | other | FP32 | 0.1598 |
| `__myl_CastMulSiluMulCastCastMulAddMulCastConc_myl0_11` | other | FP8 | 0.1101 |
| `/model.2/cv2/conv/Conv_myl0_12` | conv | FP8 | 0.1031 |
| `/model.2/m.0/cv1/conv/Conv_myl0_8` | conv | FP8 | 0.1012 |
| `/model.2/m.0/cv2/conv/Conv_myl0_10` | conv | FP8 | 0.0973 |

</details>

## 4. 瓶颈判读

- **fp8** (b8): GPU纯计算 1.54× → 含传输 1.53×
