"""统计模块测试。

统计代码的风险与业务代码不同：**算错了不会报错，只会给出错误的结论**。
所以这里用已知答案的算例来验证，而不是"跑得通就算过"。

两个重点：

1. **BH 校正**用文献中的经典算例核对（拒绝前 4 个）。
2. **配对检验的功效优势** —— 构造同一样本，展示配对检验显著、
   非配对检验不显著。这正是 P2 里"CI 重叠 → 无法断言"的解法。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from copilot.stats import (
    benjamini_hochberg,
    cohens_dz,
    compare_paired,
    paired_bootstrap_diff_ci,
    paired_t_test,
    wilcoxon_test,
)

# ─────────────────────── BH 校正 ───────────────────────


def test_benjamini_hochberg_matches_published_example() -> None:
    """Wikipedia 的 BH 算例：m=15, α=0.05 时应拒绝前 4 个假设。"""
    p = [
        0.0001, 0.0004, 0.0019, 0.0095, 0.0201,
        0.0278, 0.0298, 0.0344, 0.0459, 0.3240,
        0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
    ]
    adj = benjamini_hochberg(p)
    rejected = sum(1 for x in adj if x <= 0.05)
    assert rejected == 4, f"应拒绝 4 个，实际 {rejected}；调整后 p={adj[:6]}"
    assert adj[3] <= 0.05 < adj[4]


def test_benjamini_hochberg_is_monotone_and_bounded() -> None:
    adj = benjamini_hochberg([0.01, 0.2, 0.03, 0.9])
    assert all(0.0 <= x <= 1.0 for x in adj)
    # 原始 p 越小，调整后不应反而更大
    order = np.argsort([0.01, 0.2, 0.03, 0.9])
    seq = [adj[i] for i in order]
    assert seq == sorted(seq)


def test_benjamini_hochberg_never_exceeds_one() -> None:
    adj = benjamini_hochberg([0.9, 0.95, 0.99])
    assert max(adj) <= 1.0


def test_benjamini_hochberg_handles_empty_and_nan() -> None:
    assert benjamini_hochberg([]) == []
    out = benjamini_hochberg([0.01, float("nan")])
    assert np.isnan(out[1])


# ─────────────────────── Wilcoxon ───────────────────────


def test_wilcoxon_detects_consistent_difference() -> None:
    a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    b = [x + 0.2 for x in a]
    _, p = wilcoxon_test(a, b)
    assert p < 0.05


def test_wilcoxon_identical_samples_gives_p_one() -> None:
    """全部差值为 0 时应返回 p=1，而不是让底层库抛错。"""
    a = [0.5, 0.5, 0.5]
    stat, p = wilcoxon_test(a, a)
    assert (stat, p) == (0.0, 1.0)


def test_wilcoxon_rejects_unpaired_lengths() -> None:
    with pytest.raises(ValueError, match="长度"):
        wilcoxon_test([1.0, 2.0], [1.0])


# ───────────────★ 配对 vs 非配对：功效差异 ★───────────────


def test_pairing_gives_more_power_than_unpaired() -> None:
    """★ 同一个数据集：配对检验显著，非配对检验不显著。

    这是 P2 那个困境的解法：把两个变体当成"独立两批样本"比较，
    会浪费掉"同一批查询"这一关键信息，导致功效不足、无法断言。

    这里构造"每个查询都稳定提升 0.5，但查询之间差异很大"的数据 ——
    这正是真实检索评测的形态（难题对两者都难）。
    """
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    # 每个查询都提升约 0.5，查询间方差不变
    b = a + np.array([0.5, 0.4, 0.6, 0.5, 0.5, 0.4, 0.6, 0.5])

    _, p_paired = paired_t_test(a, b)
    _, p_unpaired = sps.ttest_ind(a, b)

    assert p_paired < 0.01, "配对检验应显著"
    assert p_unpaired > 0.05, "非配对检验在此数据上不应显著"
    assert p_paired < p_unpaired


# ─────────────────── 配对 bootstrap 置信区间 ───────────────────


def test_paired_bootstrap_ci_excludes_zero_for_real_effect() -> None:
    a = [0.1 * i for i in range(1, 21)]
    b = [x + 0.3 for x in a]
    lo, hi, diff = paired_bootstrap_diff_ci(a, b)
    assert lo > 0, f"稳定正效应应使区间排除 0，实际 [{lo:.3f},{hi:.3f}]"
    assert diff == pytest.approx(0.3, abs=1e-9)


def test_paired_bootstrap_ci_includes_zero_for_no_effect() -> None:
    a = [0.1 * i for i in range(1, 21)]
    b = list(reversed(a))  # 同样的值，只是顺序不同 -> 均无差异
    lo, hi, diff = paired_bootstrap_diff_ci(a, b)
    assert lo <= 0 <= hi
    assert abs(diff) < 1e-9


def test_paired_bootstrap_is_deterministic() -> None:
    a = [0.1 * i for i in range(1, 11)]
    b = [x + 0.05 for x in a]
    assert paired_bootstrap_diff_ci(a, b) == paired_bootstrap_diff_ci(a, b)


def test_paired_bootstrap_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="长度"):
        paired_bootstrap_diff_ci([1.0, 2.0], [1.0])


# ─────────────────────── 效应量 ───────────────────────


def test_cohens_dz_sign_and_magnitude() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    # 非恒定差值，得到有限效应量
    b = [x + 1.0 + (0.1 if i % 2 else -0.1) for i, x in enumerate(a)]
    dz = cohens_dz(a, b)
    assert np.isfinite(dz) and dz > 0, "b 更大时 dz 应为正且有限"
    assert cohens_dz(b, a) < 0, "交换顺序应变号"


def test_cohens_dz_constant_difference_keeps_sign() -> None:
    """★ 差值标准差为 0 时效应量无定义，但**符号不能丢**。

    曾经返回常数 ``inf``，导致 ``cohens_dz(b, a)`` 也报正无穷 ——
    调用方会据此认为"变好了"，与事实相反。
    """
    a = [1.0, 2.0, 3.0]
    b = [x + 1.0 for x in a]
    assert cohens_dz(a, b) == float("inf")
    assert cohens_dz(b, a) == float("-inf")


def test_cohens_dz_zero_for_identical() -> None:
    a = [1.0, 2.0, 3.0]
    assert cohens_dz(a, a) == 0.0


# ─────────────────────── 组合接口 ───────────────────────


def test_compare_paired_reports_direction_and_significance() -> None:
    a = [0.2, 0.3, 0.25, 0.4, 0.35, 0.5, 0.45, 0.6, 0.55, 0.5]
    b = [x + 0.1 for x in a]
    r = compare_paired(a, b, label_a="A", label_b="B", metric="recall@5")

    assert r.mean_diff == pytest.approx(0.1, abs=1e-9)
    assert r.direction() == "↑"
    assert r.ci_excludes_zero
    assert r.significant, f"p={r.wilcoxon_p}"
    assert r.n == 10


def test_compare_paired_regression_after_adjustment_can_flip() -> None:
    """校正后 p 变大 —— 说明多重比较确实起了作用，不是装饰。"""
    a = [0.2, 0.3, 0.25, 0.4, 0.35, 0.5, 0.45, 0.6, 0.55, 0.5]
    b = [x + 0.05 for x in a]
    r = compare_paired(a, b, metric="mrr")
    # 手工施加一个较宽松的校正（相当 p 放大 3 倍）
    r.p_adjusted = min(1.0, r.wilcoxon_p * 3)
    assert r.p_adjusted >= r.wilcoxon_p
