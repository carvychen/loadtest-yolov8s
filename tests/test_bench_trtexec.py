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


def test_swin_uses_its_own_onnx_and_input_shape(tmp_path):
    (tmp_path / "artifacts").mkdir()
    for f in ("swin_t.onnx", "swin_t.fp16.onnx", "swin_t.fp8.onnx"):
        (tmp_path / "artifacts" / f).touch()
    calls = tmp_path / "calls.txt"
    env = dict(os.environ, PATH=f"{REPO / 'tests/fake_bin'}{os.pathsep}{os.environ['PATH']}",
               FAKE_TRT_VER="110300", FAKE_TRTEXEC_CALLS=str(calls), BATCHES="8")
    r = subprocess.run([sys.executable, str(REPO / "quant/bench_trtexec.py"), "--model", "swin_t", "lbl", "fp16", "fp8"],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    text = calls.read_text()
    assert "--onnx=artifacts/swin_t.fp16.onnx --saveEngine=artifacts/model_repository_q/swin_t_fp16/1/model.plan" in text
    assert "--maxShapes=input:32x3x224x224" in text and "--shapes=input:8x3x224x224" in text
    assert (tmp_path / "results/quant/lbl-trtexec.csv").read_text().count("\n") == 1 + 2 * 2


def test_build_failure_skips_that_variant_and_keeps_going(tmp_path, monkeypatch):
    # 一个变体建不起来 (如 TRT 10.13 上的 fp8-torch) 不应中止整轮: 记下来, 其余照测;
    # 旧引擎先删掉, 免得 Triton 加载到别的 TRT 版本留下的 plan
    stale = tmp_path / "artifacts/model_repository_q/yolov8s_fp8/1/model.plan"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    monkeypatch.setenv("FAKE_TRTEXEC_FAIL_BUILD", "yolov8s.fp8.onnx")
    r, _ = run_bench(tmp_path, 110300, ["fp16", "fp8", "int8"])
    assert r.returncode == 0, r.stdout + r.stderr
    rows = (tmp_path / "results/quant/lbl-trtexec.csv").read_text().splitlines()[1:]
    assert {row.split(",")[0] for row in rows} == {"fp16", "int8"}
    assert not stale.exists()
    assert (tmp_path / "results/quant/lbl-build-failed.txt").read_text() == "fp8\n"
    assert "构建失败" in r.stdout
