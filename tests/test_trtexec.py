from pathlib import Path

import pytest

import trtexec

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name):
    return (FIXTURES / name).read_text()


@pytest.mark.parametrize("log, batch, expected", [
    ("trt10-b8-compute.log", 8, {"infer_per_s": "219.674", "img_per_s": "1757.4",
                                 "gpu_compute_mean_ms": "4.55026", "gpu_compute_p99_ms": "4.56714"}),
    ("trt11-b8-transfer.log", 8, {"infer_per_s": "242.935", "img_per_s": "1943.5",
                                  "gpu_compute_mean_ms": "4.11341", "gpu_compute_p99_ms": "4.12891"}),
])
def test_run_log_metrics(log, batch, expected):
    # Throughput 的 qps 是每秒推理次数, 每次 batch 张; 指标说明段里的 "GPU Compute Time:" 不能误取
    assert trtexec.parse_metrics(fixture(log), batch) == expected


def test_run_log_without_throughput_is_an_error():
    truncated = fixture("trt10-b8-compute.log").split("=== Performance summary ===")[0]
    with pytest.raises(ValueError, match="Throughput"):
        trtexec.parse_metrics(truncated, 8)


@pytest.mark.parametrize("log, version", [("trt10-b8-compute.log", 101302), ("trt11-b8-compute.log", 110300)])
def test_version_from_banner(log, version):
    # trtexec --help 与每份日志首行都有 "[TensorRT v101302]"
    assert trtexec.parse_version(fixture(log)) == version


def test_version_defaults_to_trt10_when_banner_missing():
    assert trtexec.parse_version("no banner") == 100000


@pytest.mark.parametrize("major, mode, flags", [
    # TRT 10 需显式开 CUDA Graph / spin-wait; compute 关掉 H2D/D2H
    (10, "compute", ["--useCudaGraph", "--useSpinWait", "--noDataTransfers"]),
    (10, "transfer", ["--useCudaGraph", "--useSpinWait"]),
    # TRT 11 默认已开 CUDA Graph / spin-wait / 关数据传输; 含传输要用 --includeDataTransfers
    (11, "compute", []),
    (11, "transfer", ["--includeDataTransfers"]),
])
def test_measure_flags(major, mode, flags):
    assert trtexec.measure_flags(major, mode) == flags


def test_measure_flags_match_what_the_logs_report():
    for major in (10, 11):
        for mode, transfers in (("compute", "Disabled"), ("transfer", "Enabled")):
            assert f"Data transfers: {transfers}" in fixture(f"trt{major}-b8-{mode}.log")
            assert "CUDA Graph: Enabled" in fixture(f"trt{major}-b8-{mode}.log")


@pytest.mark.parametrize("major, flags", [(10, ["--noDataTransfers", "--separateProfileRun"]), (11, [])])
def test_profile_flags(major, flags):
    assert trtexec.profile_flags(major) == flags
