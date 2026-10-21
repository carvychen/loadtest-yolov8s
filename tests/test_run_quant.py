"""run_quant.sh 的容器编排: 在假 docker 上跑, 与重构前录下的 docker 命令比对。
只有第 [1] 步多了变体校验、第 [3]/[4] 步换成 prepare_variants.py / bench_trtexec.py, 其余命令逐字不变。"""
import os
import shutil
import subprocess
from pathlib import Path

from legacy import to_new_layout

REPO = Path(__file__).parent.parent
GOLDEN = Path(__file__).parent / "golden/legacy-run_quant/docker_calls.txt"
VARIANTS = "fp16 fp8 int8 fp8-nohead fp8-conv fp8-opt5"
QRUN = "run --rm -v <PWD>:/work -w /work"


def expected_calls():
    replaced = ("python -c ", "modelopt.onnx", "make_q_repo.py", "bench_trtexec.sh")
    out = []
    for line in to_new_layout(GOLDEN.read_text()).splitlines():
        if line.startswith("image inspect"):
            out += [line, f"{QRUN} yolo-quant:modelopt-0.47.0 python quant/variants.py check {VARIANTS}"]
        elif "bench_trtexec.sh" in line:
            out += [f"{QRUN} -e FP8_CALIB -e INT8_CALIB yolo-quant:modelopt-0.47.0 python quant/prepare_variants.py {VARIANTS}",
                    "run --gpus all --rm -v <PWD>:/work -w /work -e BATCHES -e PROFILE_BATCH -e DURATION "
                    f"nvcr.io/nvidia/tritonserver:25.08-py3 python3 quant/bench_trtexec.py lbl {VARIANTS}"]
        elif not any(r in line for r in replaced):
            out.append(line)
    return out


def test_docker_orchestration_matches_legacy(tmp_path):
    shutil.copy(REPO / "run_quant.sh", tmp_path)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/yolov8s.onnx").touch()
    calls = tmp_path / "docker_calls.txt"
    env = dict(os.environ, PATH=f"{REPO / 'tests/fake_host_bin'}{os.pathsep}{os.environ['PATH']}",
               FAKE_DOCKER_CALLS=str(calls), VARIANTS=VARIANTS, PWD=str(tmp_path))
    r = subprocess.run(["bash", "run_quant.sh", "lbl"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls.read_text().splitlines() == expected_calls()


def test_push_results_commits_only_this_labels_results(tmp_path):
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    subprocess.run(["git", "clone", "-q", str(remote), str(work)], check=True, capture_output=True)
    git = lambda *a: subprocess.run(["git", "-C", str(work), *a], check=True, capture_output=True, text=True).stdout
    git("config", "user.name", "t"); git("config", "user.email", "t@t")
    shutil.copy(REPO / "run_quant.sh", work)
    shutil.copy(REPO / ".gitignore", work)
    git("add", "-A"); git("commit", "-qm", "init"); git("push", "-q", "origin", "HEAD")
    (work / "artifacts").mkdir()
    (work / "artifacts/yolov8s.onnx").touch()
    out = work / "results/quant"
    out.mkdir(parents=True)
    for f in ("lbl-trtexec.csv", "lbl-fp8-map.json", "lbl-fp8-b8-compute.log", "other-trtexec.csv"):
        (out / f).write_text("x")
    (work / "quant_run.log").write_text("console")
    env = dict(os.environ, PATH=f"{REPO / 'tests/fake_host_bin'}{os.pathsep}{os.environ['PATH']}",
               FAKE_DOCKER_CALLS=str(tmp_path / "calls.txt"), VARIANTS="fp8", PWD=str(work), PUSH_RESULTS="1")
    r = subprocess.run(["bash", "run_quant.sh", "lbl"], cwd=work, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    pushed = subprocess.run(["git", "-C", str(remote), "show", "--name-only", "--format=%s", "HEAD"],
                            capture_output=True, text=True).stdout.split()
    # 只提交本 label 的 CSV / JSON; .log (被 .gitignore 忽略) 与别的 label、根目录终端日志都不带
    assert pushed == ["结果:", "lbl", "results/quant/lbl-fp8-map.json", "results/quant/lbl-summary.md",
                      "results/quant/lbl-trtexec.csv"]
