"""配对二分类检验的测试（McNemar + 通过率差值 CI）。

核心验收：
1. **McNemar 只用不一致对** —— 两边都通过的样本不影响 p 值；
2. **n=1 不能判显著** —— 实测踩过：只看 bootstrap CI 会让单样本显示"显著"；
3. 显著性要求 **CI 不含 0 且 p < 0.05** 两个条件；
4. 全部一致时 p = 1（没有证据说明两者不同）。
"""

from __future__ import annotations

import pytest

from copilot.stats import (
    compare_proportions,
    mcnemar_test,
    paired_proportion_diff_ci,
)

# ─────────────────── McNemar ───────────────────


def test_mcnemar_counts_discordant_pairs_only() -> None:
    """两边都通过的样本对"哪个更好"没有发言权 —— 不该影响 p。"""
    a = [True, True, True, False]
    b = [True, True, True, True]
    # 只有 1 对不一致：A 失败、B 通过
    b_count, c_count, _p = mcnemar_test(a, b)
    assert b_count == 0
    assert c_count == 1


def test_mcnemar_all_agree_gives_p_one() -> None:
    assert mcnemar_test([True, False], [True, False]) == (0, 0, 1.0)


def test_mcnemar_symmetric_flips_not_significant() -> None:
    """翻转数相当时不显著 —— 说明两种配置一样好。"""
    a = [True] * 5 + [False] * 5
    b = [False] * 5 + [True] * 5
    b_count, c_count, p = mcnemar_test(a, b)
    assert b_count == c_count == 5
    assert p > 0.5


def test_mcnemar_all_one_way_is_significant() -> None:
    a = [False] * 10
    b = [True] * 10
    b_count, c_count, p = mcnemar_test(a, b)
    assert (b_count, c_count) == (0, 10)
    assert p < 0.01


def test_mcnemar_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="长度"):
        mcnemar_test([True], [True, False])


def test_mcnemar_single_flip_cannot_be_significant() -> None:
    """★ 1 个翻转永远推不出结论 —— 精确二项的 p 至少 1.0。"""
    _b, _c, p = mcnemar_test([False], [True])
    assert p == 1.0


# ─────────────────── 差值 CI ───────────────────


def test_proportion_diff_ci_reports_point_estimate() -> None:
    diff, lo, hi = paired_proportion_diff_ci([False] * 10, [True] * 10)
    assert diff == pytest.approx(1.0)
    assert lo <= diff <= hi


def test_proportion_diff_ci_is_deterministic() -> None:
    """固定种子 —— 同一份数据必须跑出同样的区间。"""
    a = [True, False] * 10
    b = [True, True] * 10
    assert paired_proportion_diff_ci(a, b) == paired_proportion_diff_ci(a, b)


def test_proportion_diff_ci_zero_when_identical() -> None:
    a = [True, False, True, False]
    diff, lo, hi = paired_proportion_diff_ci(a, list(a))
    assert diff == 0.0
    assert lo <= 0 <= hi


# ─────────────────── 综合判据 ───────────────────


def test_significant_requires_both_ci_and_p() -> None:
    """★ 只有"CI 不含 0 且 p < 0.05"才算显著。"""
    r = compare_proportions([False] * 10, [True] * 10)
    assert r["ci_excludes_zero"] is True
    assert r["mcnemar_p"] < 0.05
    assert r["significant"] is True


def test_single_sample_is_never_significant() -> None:
    """★ 实测踩过的坑：n=1 时 Δ=+100%，CI 退化成 [100%, 100%]，
    只看 CI 会显示"显著"。但一个样本什么都推不出来。"""
    r = compare_proportions([False], [True])
    assert r["diff"] == 1.0
    assert r["mcnemar_p"] == 1.0
    assert r["significant"] is False
    assert r["ci_excludes_zero"] is True   # CI 确实不含 0 —— 所以必须再看 p


def test_direction_without_significance_is_flagged_separately() -> None:
    """方向明确但统计不显著时，两个字段要能分别表达。"""
    # 4 个翻转、无反向翻转 → p = 2*(0.5^4) = 0.125 > 0.05
    r = compare_proportions([False] * 4 + [True] * 4, [True] * 8)
    assert r["ci_excludes_zero"] is True
    assert r["mcnemar_p"] > 0.05
    assert r["significant"] is False


def test_worse_direction_detected() -> None:
    r = compare_proportions([True] * 10, [False] * 10)
    assert r["diff"] == -1.0
    assert r["significant"] is True
    assert r["diff"] < 0


def test_compare_returns_all_report_fields() -> None:
    r = compare_proportions([True, False], [True, True], label_a="v1", label_b="v2")
    for key in (
        "label_a", "label_b", "n", "pass_rate_a", "pass_rate_b",
        "diff", "diff_ci_low", "diff_ci_high",
        "b_only_a", "b_only_b", "mcnemar_p",
        "ci_excludes_zero", "significant",
    ):
        assert key in r, key
    assert r["label_a"] == "v1"
    assert r["n"] == 2
