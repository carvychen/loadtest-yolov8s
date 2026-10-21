# golden 样本

由重构前的脚本 (commit 34c495a 的 run_quant.sh / bench_trtexec.sh / make_q_repo.py) 在本地录制,
用 tests/fake_bin 里的假 trtexec / nvidia-smi 和假 docker / curl 代替 GPU 与容器。重构后的代码必须产出相同结果。

| 目录 | 内容 |
|---|---|
| bench-trt10/, bench-trt11/ | 变体 `fp16 fp8 int8 fp8-nohead fp8-conv fp8-opt5`、BATCHES=`1 8`, FAKE_TRT_VER=101302 / 110300 下的 trtexec 调用参数、CSV、env.txt |
| legacy-run_quant/ | 同一组变体下 run_quant.sh 发出的全部 docker 命令 (`<PWD>` = 工作目录) 和生成的 config.pbtxt |

旧脚本需要 bash ≥ 4.4 (空数组 + `set -u`), macOS 自带的 bash 3.2 跑不了。

录制于目录重组之前 (脚本与生成物都在根目录)。比对时由 `tests/legacy.py` 把路径换成现在的布局 (`quant/`、`serving/`、`artifacts/`)，其余逐字不变。
