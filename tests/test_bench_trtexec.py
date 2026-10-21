"""bench_trtexec.py 端到端: 在假 trtexec 上跑, 与重构前 bench_trtexec.sh 录下的 golden 逐字比对。"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from legacy import to_new_layout

REPO = Path(__file__).parent.parent
GOLDEN = Path(__file__).parent / "golden"
VARIANTS = ["fp16", "fp8", "int8", "fp8-nohead", "fp8-conv", "fp8-opt5"]
ONNX = ["yolov8s.onnx", "yolov8s.fp16.onnx", "yolov8s.fp8.onnx", "yolov8s.int8.onnx",
        "yolov8s.fp8-nohead.onnx", "yolov8s.fp8-conv.onnx"]


def run_bench(tmp_path, trt_ver, variants=VARIANTS, missing_onnx=()):
    for f in ONNX:
        if f not in missing_onnx:
            (tmp_path / "artifacts").mkdir(exist_ok=True)
            (tmp_path / "artifacts" / f).touch()
    calls = tmp_path / "calls.txt"
    env = dict(os.environ, PATH=f"{REPO / 'tests/fake_bin'}{os.pathsep}{os.environ['PATH']}",
               FAKE_TRT_VER=str(trt_ver), FAKE_TRTEXEC_CALLS=str(calls), BATCHES="1 8")
    r = subprocess.run([sys.executable, str(REPO / "quant/bench_trtexec.py"), "lbl", *variants],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    return r, calls


@pytest.mark.parametrize("trt_ver, major", [(101302, 10), (110300, 11)])
def test_matches_legacy_golden(tmp_path, trt_ver, major):
    r, calls = run_bench(tmp_path, trt_ver)
    assert r.returncode == 0, r.stdout + r.stderr
    g = GOLDEN / f"bench-trt{major}"
    out = tmp_path / "results/quant"
    # 旧脚本调了两次 --help (取版本 + 写 env.txt), 新脚本只调一次; 其余调用逐字一致
    no_help = lambda text: [l for l in text.splitlines() if l != "--help"]
    assert no_help(calls.read_text()) == no_help(to_new_layout((g / "trtexec_calls.txt").read_text()))
    assert (out / "lbl-trtexec.csv").read_text() == (g / "trtexec.csv").read_text()
    # env.txt: 有意修改 —— TensorRT 版本行只写一次 (旧脚本每个变体重复一行)
    legacy_env = (g / "env.txt").read_text().splitlines()
    first_ver = next(i for i, l in enumerate(legacy_env) if l.startswith("TensorRT version"))
    assert (out / "lbl-env.txt").read_text().splitlines() == legacy_env[:first_ver + 1]
    for v in VARIANTS:
        assert (out / f"lbl-{v}-layers.json").exists() and (out / f"lbl-{v}-profile.json").exists()


def test_typo_in_variant_fails_before_any_build(tmp_path):
    r, calls = run_bench(tmp_path, 110300, ["fp8", "fp8-nohed"])
    assert r.returncode != 0 and "nohed" in r.stdout + r.stderr
    assert "--saveEngine" not in (calls.read_text() if calls.exists() else "")


def test_empty_env_falls_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("DURATION", "")
    r, calls = run_bench(tmp_path, 110300, ["fp8"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--duration=15" in calls.read_text() and "--duration=\n" not in calls.read_text()


def test_missing_onnx_is_reported(tmp_path):
    r, _ = run_bench(tmp_path, 110300, ["int8"], missing_onnx=["yolov8s.int8.onnx"])
    assert r.returncode != 0 and "artifacts/yolov8s.int8.onnx" in r.stdout + r.stderr
