import subprocess
import sys
from pathlib import Path

import pytest

from variants import Variant

REPO = Path(__file__).parent.parent

CALIB = "datasets/calib_500.npy"


def modelopt_cmd(name, **kw):
    return " ".join(Variant.parse(name).modelopt_args(CALIB, **kw))


def test_plain_fp8_quantizes_with_max_calibration():
    v = Variant.parse("fp8")
    assert v.onnx_path(10) == v.onnx_path(11) == "artifacts/yolov8s.fp8.onnx"
    assert modelopt_cmd("fp8") == (
        "--onnx_path=artifacts/yolov8s.onnx --quantize_mode=fp8 --calibration_data_path=datasets/calib_500.npy "
        "--calibration_method=max --calibration_eps cpu --output_path=artifacts/yolov8s.fp8.onnx")
    assert v.triton_name == "yolov8s_fp8"


def test_int8_uses_entropy_and_method_can_be_overridden():
    assert "--calibration_method=entropy" in modelopt_cmd("int8")
    assert "--calibration_method=minmax" in modelopt_cmd("int8", calibration_method="minmax")


def test_quant_tags_change_onnx_and_modelopt_args():
    assert modelopt_cmd("fp8-nohead").endswith(
        "--output_path=artifacts/yolov8s.fp8-nohead.onnx --nodes_to_exclude /model\\.22/.*")
    assert modelopt_cmd("fp8-conv").endswith(
        "--output_path=artifacts/yolov8s.fp8-conv.onnx --op_types_to_quantize Conv")


@pytest.mark.parametrize("name, trt_major, onnx, flags", [
    # TRT 10: fp16 = 原 FP32 ONNX + --fp16 (弱类型); 量化变体 = Q/DQ ONNX + --stronglyTyped
    ("fp16", 10, "artifacts/yolov8s.onnx", ["--fp16"]),
    ("fp8", 10, "artifacts/yolov8s.fp8.onnx", ["--stronglyTyped"]),
    ("fp8-opt5", 10, "artifacts/yolov8s.fp8.onnx", ["--stronglyTyped", "--builderOptimizationLevel=5"]),
    # TRT 11: 弱类型已移除, 强类型为默认; fp16 改用 AutoCast FP16 ONNX
    ("fp16", 11, "artifacts/yolov8s.fp16.onnx", []),
    ("int8", 11, "artifacts/yolov8s.int8.onnx", []),
    ("fp8-opt5", 11, "artifacts/yolov8s.fp8.onnx", ["--builderOptimizationLevel=5"]),
])
def test_engine_build_depends_on_trt_major(name, trt_major, onnx, flags):
    v = Variant.parse(name)
    assert v.onnx_path(trt_major) == onnx
    assert v.build_flags(trt_major) == flags


def test_fp16_is_not_quantized():
    assert Variant.parse("fp16").modelopt_args(CALIB) is None
    with pytest.raises(ValueError, match="TRT"):   # fp16 的 ONNX 必须指明 TRT 主版本
        Variant.parse("fp16").onnx_path()


def test_build_tag_shares_onnx_but_keeps_its_own_triton_model():
    v = Variant.parse("fp8-opt5")
    assert v.modelopt_args(CALIB) == Variant.parse("fp8").modelopt_args(CALIB)
    assert v.model_dir == "artifacts/model_repository_q/yolov8s_fp8-opt5"


@pytest.mark.parametrize("name, hint", [
    ("fp8-nohed", "nohed"),       # 拼错的标签: 以前会被当成普通 fp8 悄悄跑完
    ("bf16", "bf16"),
    ("fp16-nohead", "fp16"),      # fp16 不量化, 量化标签没有意义
    ("fp8-opt5-opt5", "opt5"),
])
def test_invalid_names_are_rejected(name, hint):
    with pytest.raises(ValueError, match=hint):
        Variant.parse(name)


def test_tag_order_does_not_matter_for_onnx():
    assert Variant.parse("fp8-conv-nohead").onnx_path(11) == "artifacts/yolov8s.fp8-nohead-conv.onnx"
    assert Variant.parse("fp8-conv-nohead").triton_name == "yolov8s_fp8-conv-nohead"


def test_fp16_with_build_tag_is_allowed():
    assert Variant.parse("fp16-opt5").build_flags(11) == ["--builderOptimizationLevel=5"]


def run_check(*names):
    return subprocess.run([sys.executable, "quant/variants.py", "check", *names], cwd=REPO, capture_output=True, text=True)


def test_check_command_fails_fast_on_a_typo():
    r = run_check("fp16", "fp8-nohed")
    assert r.returncode != 0 and "nohed" in r.stderr
    assert run_check("fp16", "fp8", "int8", "fp8-nohead-opt5").returncode == 0


def test_quant_name_identifies_the_quantized_onnx():
    # 变体名去掉构建标签、量化标签按固定顺序: 同一 quant_name 共用一份 ONNX 与量化日志
    assert Variant.parse("fp8-conv-nohead-opt5").quant_name == "fp8-nohead-conv"
    assert Variant.parse("fp16-opt5").quant_name is None
