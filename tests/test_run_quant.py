"""run_quant.sh 的容器编排: 在假 docker 上跑, 与重构前录下的 docker 命令比对。
只有第 [1] 步多了变体校验、第 [3]/[4] 步换成 prepare_variants.py / bench_trtexec.py, 其余命令逐字不变。"""
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).parent.parent
GOLDEN = Path(__file__).parent / "golden/legacy-run_quant/docker_calls.txt"
VARIANTS = "fp16 fp8 int8 fp8-nohead fp8-conv fp8-opt5"
QRUN = "run --rm -v <PWD>:/work -w /work"


def expected_calls():
    replaced = ("python -c ", "modelopt.onnx", "make_q_repo.py", "bench_trtexec.sh")
    out = []
    for line in GOLDEN.read_text().splitlines():
        if line.startswith("image inspect"):
            out += [line, f"{QRUN} yolo-quant:modelopt-0.47.0 python variants.py check {VARIANTS}"]
        elif "bench_trtexec.sh" in line:
            out += [f"{QRUN} -e FP8_CALIB -e INT8_CALIB yolo-quant:modelopt-0.47.0 python prepare_variants.py {VARIANTS}",
                    "run --gpus all --rm -v <PWD>:/work -w /work -e BATCHES -e PROFILE_BATCH -e DURATION "
                    f"nvcr.io/nvidia/tritonserver:25.08-py3 python3 bench_trtexec.py lbl {VARIANTS}"]
        elif not any(r in line for r in replaced):
            out.append(line)
    return out


def test_docker_orchestration_matches_legacy(tmp_path):
    shutil.copy(REPO / "run_quant.sh", tmp_path)
    (tmp_path / "yolov8s.onnx").touch()
    calls = tmp_path / "docker_calls.txt"
    env = dict(os.environ, PATH=f"{REPO / 'tests/fake_host_bin'}{os.pathsep}{os.environ['PATH']}",
               FAKE_DOCKER_CALLS=str(calls), VARIANTS=VARIANTS, PWD=str(tmp_path))
    r = subprocess.run(["bash", "run_quant.sh", "lbl"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls.read_text().splitlines() == expected_calls()
