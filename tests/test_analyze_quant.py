"""analyze_quant.py 的汇总输出 (需要 pandas; 本地没装时跳过)。"""
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
REPO = Path(__file__).parent.parent
HEADER = "variant,batch,mode,infer_per_s,img_per_s,gpu_compute_mean_ms,gpu_compute_p99_ms\n"


def test_summary_without_fp16_baseline_does_not_crash(tmp_path):
    # 只跑 VARIANTS=fp8 时没有 FP16 基线: 吞吐照常列出, 加速比判读跳过并说明原因
    out = tmp_path / "results/quant"
    out.mkdir(parents=True)
    (out / "lbl-trtexec.csv").write_text(HEADER + "fp8,8,compute,248.3,1986.7,4.03,4.04\n"
                                                  "fp8,8,transfer,241.5,1931.7,4.14,4.15\n")
    r = subprocess.run([sys.executable, str(REPO / "quant/analyze_quant.py"), "--label", "lbl"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "1987 (—)" in r.stdout and "没有 fp16 基线" in r.stdout


def test_swin_summary_uses_agreement_and_skips_cnn_judgement(tmp_path):
    out = tmp_path / "results/quant"
    out.mkdir(parents=True)
    (out / "lbl-trtexec.csv").write_text(HEADER + "fp16,8,compute,100,800,10,10\nfp16,8,transfer,90,720,11,11\n"
                                                  "fp8,8,compute,115,920,8.7,8.8\nfp8,8,transfer,104,832,9.6,9.7\n")
    (out / "lbl-fp8-agree.json").write_text('{"top1_agree": 0.9812, "top5_overlap": 0.9644, "cos_mean": 0.99871, "cos_min": 0.981}')
    r = subprocess.run([sys.executable, str(REPO / "quant/analyze_quant.py"), "--model", "swin_t", "--label", "lbl"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("# Swin-T 量化实验: lbl")
    assert "| fp8 | 98.12% | 96.44% | 0.99871 / 0.98100 |" in r.stdout
    assert "瓶颈在量化覆盖" not in r.stdout and "GPU纯计算 1.15×" in r.stdout
