"""配对显著性检验与效应量。

为什么必须做配对检验
--------------------
P2 的消融报告里出现了一个典型困境：某个变体的点估计看着变化很大
（Recall@1 从 0.800 掉到 0.675），但两组的 **95% 置信区间仍然重叠**，
于是**无法断言**差异存在。

问题出在用错了方法。把两个变体当成"两批独立样本、比较均值"，
会浪费掉最重要的信息：**它们跑的是同一批查询**。
同一查询的难度在两个变体之间是相关的 —— 难题对两者都难。
配对检验正是利用这种相关性来消掉查询难度带来的方差，
因此在同样样本量下**功效高得多**。

三个必须一起报的东西
--------------------
1. **p 值** —— 差异是否可能纯属偶然；
2. **差值置信区间** —— 差异可能有多大（p 值不告诉你量级）；
3. **效应量**（Cohen's dz）—— 差异相对于变异的规模。

只报 p 值是常见错误：样本足够大时，毫无实际意义的微小差异也会"显著"。

多重比较必须校正
----------------
本轮要做 5 组变体 × 3 个指标 = 15 次检验。若每次都按 α=0.05 判定，
**即使全部变体毫无差别，也期望出现约 0.75 个假阳性**。
因此默认用 Benjamini–Hochberg 控制错误发现率（FDR），
并同时给出未校正与校正后的 p 值，便于对照。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats as sps

__all__ = [
    "PairedResult",
    "paired_bootstrap_diff_ci",
    "wilcoxon_test",
    "paired_t_test",
    "cohens_dz",
    "benjamini_hochberg",
    "compare_paired",
    "mcnemar_test",
    "paired_proportion_diff_ci",
]

#: 固定种子：否则同一份数据每次跑出不同区间，可复现性即告丧失
DEFAULT_SEED = 20260101


@dataclass
class PairedResult:
    """一次配对比较的完整结果。"""

    label_a: str
    label_b: str
    metric: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float          # b - a，正数表示 b 更好
    diff_ci_low: float
    diff_ci_high: float
    wilcoxon_stat: float
    wilcoxon_p: float
    t_stat: float
    t_p: float
    cohens_dz: float
    #: 多重比较校正后的 p（由调用方统一回填）
    p_adjusted: float | None = None

    @property
    def ci_excludes_zero(self) -> bool:
        """差值置信区间是否排除 0。这是比"p<0.05"更直观的判据。"""
        return self.diff_ci_low > 0 or self.diff_ci_high < 0

    @property
    def significant(self) -> bool:
        """以校正后 p 值为准（未校正时退回未校正值）。"""
        p = self.p_adjusted if self.p_adjusted is not None else self.wilcoxon_p
        return p < 0.05

    def direction(self) -> str:
        return "↑" if self.mean_diff > 0 else ("↓" if self.mean_diff < 0 else "=")


# ─────────────────────── 基础统计量 ───────────────────────


def paired_bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """配对 bootstrap：对**差值的均值**做置信区间，返回 ``(lo, hi, mean_diff)``。

    关键是按**查询**重采样（而不是各自独立重采样）：
    同一次重采样里两个变体取同一个查询下标，配对关系才被保留。
    独立重采样会把配对结构打散，区间因而偏宽。
    """
    av = np.asarray(list(a), dtype=float)
    bv = np.asarray(list(b), dtype=float)
    if av.shape != bv.shape:
        raise ValueError(f"配对样本长度必须一致：{av.shape} vs {bv.shape}")
    if av.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    d = bv - av
    mean_diff = float(d.mean())
    if av.size == 1:
        return (mean_diff, mean_diff, mean_diff)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, av.size, size=(n_boot, av.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi), mean_diff)


def wilcoxon_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Wilcoxon 符号秩检验（配对、非参数）。

    默认选它而非 t 检验，因为检索指标常有**天花板效应**
    （很多查询两个变体都是 1.0，差值恒为 0）与偏态分布，
    正态性假设不可靠。非参数检验对此更稳健，代价是功效略低。

    全部差值为 0 时返回 ``(0.0, 1.0)`` —— 两变体完全一致，
    不存在差异，而不是让 scipy 抛错。
    """
    av = np.asarray(list(a), dtype=float)
    bv = np.asarray(list(b), dtype=float)
    if av.size != bv.size:
        raise ValueError("配对样本长度必须一致")
    if av.size == 0:
        return (float("nan"), float("nan"))

    d = bv - av
    if np.allclose(d, 0.0):
        return (0.0, 1.0)

    res = sps.wilcoxon(av, bv, zero_method="wilcox", alternative="two-sided")
    return (float(res.statistic), float(res.pvalue))


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """配对 t 检验（参数方法）。与 Wilcoxon 一起报，便于判断结论是否依赖分布假设。"""
    av = np.asarray(list(a), dtype=float)
    bv = np.asarray(list(b), dtype=float)
    if av.size != bv.size:
        raise ValueError("配对样本长度必须一致")
    if av.size < 2:
        return (float("nan"), float("nan"))

    d = bv - av
    if np.allclose(d, 0.0):
        return (0.0, 1.0)
    if np.allclose(d.std(ddof=1), 0.0):
        return (float("nan"), float("nan"))

    res = sps.ttest_rel(bv, av)
    return (float(res.statistic), float(res.pvalue))


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    """配对效应量 Cohen's dz = mean(差值) / std(差值)。

    经验参考：|dz| ≈ 0.2 小、0.5 中、0.8 大。
    比 p 值更能说明"差异值不值得关心"。

    边界：差值标准差为 0 时效应量在数学上无定义（分母为 0）。
    此处返回**带符号的无穷**而非常数 ``inf`` —— 符号承载着
    "变好还是变坏"这个关键信息，丢掉符号会让调用方得出相反结论。
    """
    av = np.asarray(list(a), dtype=float)
    bv = np.asarray(list(b), dtype=float)
    if av.size < 2:
        return float("nan")
    d = bv - av
    sd = d.std(ddof=1)
    mean = float(d.mean())
    if np.isclose(sd, 0.0):
        if np.isclose(mean, 0.0):
            return 0.0
        return float("inf") if mean > 0 else float("-inf")
    return mean / sd


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini–Hochberg 校正，返回调整后的 p 值（控制 FDR）。

    做法：把 p 升序排列后按 ``p_i * n / i`` 放缩，
    再从最大端向前取累计最小值，保证调整后 p 单调不回退。
    """
    p = np.asarray(list(p_values), dtype=float)
    n = p.size
    if n == 0:
        return []
    if np.any(np.isnan(p)):
        # 含 NaN 时不做校正，避免把整批结果污染成 NaN
        return list(p)

    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return [float(x) for x in out]


# ─────────────────────── 组合接口 ───────────────────────


def compare_paired(
    a: Sequence[float],
    b: Sequence[float],
    *,
    label_a: str = "A",
    label_b: str = "B",
    metric: str = "",
    n_boot: int = 5000,
    seed: int = DEFAULT_SEED,
) -> PairedResult:
    """跑完一组配对比较的全部统计量。"""
    av = np.asarray(list(a), dtype=float)
    bv = np.asarray(list(b), dtype=float)
    if av.size != bv.size:
        raise ValueError(f"配对样本长度必须一致：{av.size} vs {bv.size}")

    lo, hi, mean_diff = paired_bootstrap_diff_ci(av, bv, n_boot=n_boot, seed=seed)
    w_stat, w_p = wilcoxon_test(av, bv)
    t_stat, t_p = paired_t_test(av, bv)

    return PairedResult(
        label_a=label_a,
        label_b=label_b,
        metric=metric,
        n=int(av.size),
        mean_a=float(av.mean()) if av.size else float("nan"),
        mean_b=float(bv.mean()) if bv.size else float("nan"),
        mean_diff=mean_diff,
        diff_ci_low=lo,
        diff_ci_high=hi,
        wilcoxon_stat=w_stat,
        wilcoxon_p=w_p,
        t_stat=t_stat,
        t_p=t_p,
        cohens_dz=cohens_dz(av, bv),
    )


# ─────────────────── 配对二分类（通过率）───────────────────


def mcnemar_test(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """McNemar 配对检验：两组配置在**同一批样本**上的通过率是否有差异。

    参数是两条等长的布尔序列（``True`` = 通过）。返回 ``(b_count, c_count, p)``：

    ======  ====================================
    ``b``   A 通过、B 不通过
    ``c``   A 不通过、B 通过
    ``p``   精确二项检验的双侧 p 值
    ======  ====================================

    ★ 为什么不能用 Wilcoxon / t 检验
    --------------------------------
    通过率是 0/1 数据，一次调用只贡献 0 或 1。**真正携带信息的是不一致对**
    （``b`` 与 ``c``）—— 两边都通过的样本，对"哪个配置更好"没有发言权。

    把 0/1 直接喂给连续型检验，会把这少数翻转的信息**摊薄到全部样本上**，
    功效低得多。这是"检验要和数据类型匹配"的一个具体例子。

    ★ 为什么用精确二项而不是卡方近似
    --------------------------------
    ``b + c`` 常常小于 25，此时卡方近似不可靠。scipy 文档也建议这么做。
    """
    if len(a) != len(b):
        raise ValueError(f"两条序列长度必须相同：{len(a)} != {len(b)}")

    pairs = list(zip(a, b, strict=True))
    b_count = sum(1 for x, y in pairs if x and not y)
    c_count = sum(1 for x, y in pairs if (not x) and y)
    n_disc = b_count + c_count

    if n_disc == 0:
        # 完全一致：没有任何证据说明两者不同 → p = 1
        return (0, 0, 1.0)

    result = sps.binomtest(min(b_count, c_count), n_disc, 0.5, alternative="two-sided")
    return (b_count, c_count, float(result.pvalue))


def paired_proportion_diff_ci(
    a: Sequence[bool],
    b: Sequence[bool],
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """配对 bootstrap：``通过率(B) − 通过率(A)`` 的差值与其置信区间。

    返回 ``(diff, lo, hi)``。

    **为什么用配对 bootstrap 而不是两个独立比例的区间**：
    两组跑的是**同一批片段**，片段难度相关（难改的片段在两个配置下都难）。
    独立区间会把片段难度当成噪声算进去，区间被撑宽，
    于是容易得出"区间重叠，无法断言"——这正是 P2 踩过的坑。
    """
    if len(a) != len(b):
        raise ValueError(f"两条序列长度必须相同：{len(a)} != {len(b)}")
    if not a:
        return (float("nan"), float("nan"), float("nan"))

    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    diff = float(arr_b.mean() - arr_a.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr_a.size, size=(n_boot, arr_a.size))
    diffs = arr_b[idx].mean(axis=1) - arr_a[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return (diff, float(lo), float(hi))


def compare_proportions(
    a: Sequence[bool],
    b: Sequence[bool],
    *,
    label_a: str = "A",
    label_b: str = "B",
    n_boot: int = 5000,
    seed: int = DEFAULT_SEED,
) -> dict:
    """把"通过率比较"该报的三个量一次算齐：差值、差值 CI、McNemar p。

    与 :func:`compare_paired` 对应 —— 那个用于连续指标，这个用于通过率。
    返回字典便于直接写进报告。
    """
    diff, lo, hi = paired_proportion_diff_ci(a, b, n_boot=n_boot, seed=seed)
    b_count, c_count, p = mcnemar_test(a, b)
    n = len(a)
    ci_excludes_zero = bool(lo > 0 or hi < 0)
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n": n,
        "pass_rate_a": (sum(a) / n) if n else float("nan"),
        "pass_rate_b": (sum(b) / n) if n else float("nan"),
        "diff": diff,
        "diff_ci_low": lo,
        "diff_ci_high": hi,
        "b_only_a": b_count,
        "b_only_b": c_count,
        "mcnemar_p": p,
        # ★ 显著性要求**两个条件同时满足**：
        #   ① 差值 CI 不含 0（量级不是噪声）
        #   ② McNemar p < 0.05（翻转方向不是五五开）
        #
        # 只看 CI 是不够的 —— 样本极少时 bootstrap 区间会退化成单点，
        # 于是"1 个样本 vs 1 个样本"也会显示 CI 不含 0，看起来"显著"。
        # 实测踩过：n=1 时 A→B 得到 Δ=+100% [+100%,+100%]，
        # 但 McNemar p=1.0000 —— 一个样本根本推不出任何结论。
        "ci_excludes_zero": ci_excludes_zero,
        "significant": bool(ci_excludes_zero and p < 0.05),
    }
