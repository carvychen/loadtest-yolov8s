"""prepare_variants.py: 与重构前 run_quant.sh 第 [3] 步 + make_q_repo.py 录下的 golden 比对 (不跑 ModelOpt, 不读真 ONNX)。"""
import shlex
import shutil
import sys
from pathlib import Path

import pytest

import prepare_variants
from legacy import to_new_layout
from variants import Variant

REPO = Path(__file__).parent.parent
GOLDEN = Path(__file__).parent / "golden/legacy-run_quant"
VARIANTS = ["fp16", "fp8", "int8", "fp8-nohead", "fp8-conv", "fp8-opt5"]


def legacy_modelopt_calls():
    marker = "yolo-quant:modelopt-0.47.0 "
    return [to_new_layout(l.split(marker, 1)[1]) for l in (GOLDEN / "docker_calls.txt").read_text().splitlines()
            if "modelopt.onnx" in l]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(REPO / "model_repository", "model_repository")
    Path("datasets").mkdir()
    Path("datasets/autocast_calib.npz").touch()   # 已存在则不重新生成 (生成要 numpy + 校准集)
    Path("artifacts").mkdir()
    Path("artifacts/yolov8s.onnx").touch()
    return tmp_path


def test_modelopt_commands_match_legacy(workdir):
    calls = []

    def fake_runner(cmd, log):
        assert cmd[0] == sys.executable
        calls.append(shlex.join(["python", *cmd[1:]]))
        Path(next(a.split("=", 1)[1] for a in cmd if a.startswith("--output_path="))).touch()
        return True

    prepare_variants.quantize([Variant.parse(n) for n in VARIANTS], runner=fake_runner)
    assert calls == legacy_modelopt_calls()


def test_quantization_without_output_is_an_error(workdir):
    Path("artifacts/yolov8s.fp16.onnx").touch()
    with pytest.raises(SystemExit, match="quant-fp8.log"):
        prepare_variants.quantize([Variant.parse("fp8")], runner=lambda cmd, log: True)


def test_modelopt_failure_stops_even_if_output_exists(workdir):
    Path("artifacts/yolov8s.fp16.onnx").touch()

    def crashing_runner(cmd, log):   # ModelOpt 中途崩溃但留下了半成品 ONNX
        Path("artifacts/yolov8s.fp8.onnx").touch()
        return False

    with pytest.raises(SystemExit, match="quant-fp8.log"):
        prepare_variants.quantize([Variant.parse("fp8")], runner=crashing_runner)


def test_triton_configs_match_legacy(workdir):
    prepare_variants.write_configs([Variant.parse(n) for n in VARIANTS],
                                   io_types=lambda onnx: ("TYPE_FP32", "TYPE_FP32"))
    for n in VARIANTS:
        assert (Path(f"artifacts/model_repository_q/yolov8s_{n}/config.pbtxt").read_text()
                == (GOLDEN / f"configs/yolov8s_{n}.pbtxt").read_text())
