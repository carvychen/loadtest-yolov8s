"""eval_agreement.py: FP8 与 FP16 引擎输出一致性的统计口径 (纯 numpy)。"""
import numpy as np
import pytest

from eval_agreement import Agreement


def test_identical_logits_agree_fully():
    x = np.random.default_rng(0).normal(size=(4, 50)).astype(np.float32)
    a = Agreement()
    a.add(x, x)
    assert a.result() == {"n_images": 4, "top1_agree": 1.0, "top5_overlap": 1.0, "cos_mean": 1.0, "cos_min": 1.0}


def test_counts_top1_flips_and_partial_top5_overlap():
    ref = np.array([[9, 8, 7, 6, 5, 0, 0], [9, 8, 7, 6, 5, 0, 0]], np.float32)
    got = np.array([[9, 8, 7, 6, 5, 0, 0],          # 完全一致
                    [8, 9, 7, 6, 0, 0, 5.5]], np.float32)  # top-1 换了; top-5 少了第 5 名, 多了最后一类
    a = Agreement()
    a.add(ref[:1], got[:1])          # 分批累加与一次算等价
    a.add(ref[1:], got[1:])
    r = a.result()
    assert r["n_images"] == 2 and r["top1_agree"] == 0.5
    assert r["top5_overlap"] == pytest.approx((5 / 5 + 4 / 5) / 2)
    # 第二张: 点积 229, |ref|²=255, |got|²=260.25 -> cos = 229/√(255·260.25) ≈ 0.88893
    assert r["cos_min"] == pytest.approx(0.88893, abs=1e-4)
    assert r["cos_mean"] == pytest.approx((1 + 0.88893) / 2, abs=1e-4)
