"""C2 数据分析：配对检验 + 差值 CI + 效应量 + 多重比较校正。

## 为什么这三样必须一起报

| 量 | 回答 | 缺了会怎样 |
|---|---|---|
| p 值（校正后） | 差异是不是偶然 | 无法判断 |
| 差值 CI | 量级多大 | "显著"但可能微不足道 |
| Cohen's dz | 值不值得关心 | 样本够大时微小差异也会显著 |

## 为什么必须配对

同一批样本跑两个配置，样本难度是相关的。
把两组当独立样本比较会**浪费掉这个信息**，
区间被撑宽 → 容易得出"无法断言"。这是本项目在检索消融上踩过的坑。

## 为什么必须做多重比较校正

一次分析里做 k 次检验，即使全部无差别，也期望出现约 `k × α` 个假阳性。
本项目实测过：某个指标未校正时 p=0.0455（判显著），
BH 校正后 p=0.171（不显著）—— 不校正就会多声称一项改进。
"""

from __future__ import annotations

from typing import Sequence

from copilot.stats import (
    benjamini_hochberg,
    cohens_dz,
    paired_bootstrap_diff_ci,
    wilcoxon_test,
)

from copilot.research.models import Comparison, Observation, StatsResult

__all__ = ["analyze_paired", "analyze_results", "ALPHA"]


#: 显著性水平
ALPHA = 0.05


def analyze_paired(
    observations: Sequence[Observation],
    *,
    metric: str,
    label_a: str = "baseline",
    label_b: str = "method",
    n_boot: int = 5000,
) -> Comparison:
    """对一组配对观测做完整比较。"""
    if len(observations) < 2:
        raise ValueError(
            f"配对观测至少需要 2 条才能估计方差，收到 {len(observations)} 条"
        )

    a = [o.a for o in observations]
    b = [o.b for o in observations]

    diff, lo, hi = paired_bootstrap_diff_ci(a, b, n_boot=n_boot)
    _stat, p = wilcoxon_test(a, b)
    dz = cohens_dz(a, b)

    return Comparison(
        metric=metric,
        label_a=label_a,
        label_b=label_b,
        n=len(observations),
        mean_a=sum(a) / len(a),
        mean_b=sum(b) / len(b),
        diff=diff,
        diff_ci_low=lo,
        diff_ci_high=hi,
        p_raw=p,
        effect_size=dz,
        # 单次比较先用未校正判定；analyze_results 会统一重判
        significant=p < ALPHA,
    )


def analyze_results(
    data: dict[str, Sequence[Observation]],
    *,
    label_a: str = "baseline",
    label_b: str = "method",
    n_boot: int = 5000,
) -> StatsResult:
    """对多个指标一起分析，并做 BH 校正。

    ``data``：``{指标名: [Observation, ...]}``。

    **校正后重判显著性**：把 ``p_raw`` 过一遍 BH 得到 ``p_adjusted``，
    再以它为准。这一步不能省 —— 否则报告里的"显著"是未校正的。
    """
    metrics = [m for m, obs in data.items() if len(obs) >= 2]
    if not metrics:
        return StatsResult(
            correction="",
            n_tests=0,
            conclusion="没有可分析的指标（每个指标至少需要 2 条配对观测）。",
            limitations=["没有数据就没有结论 —— 这里不编造任何数字。"],
        )

    comparisons = [
        analyze_paired(
            data[m], metric=m, label_a=label_a, label_b=label_b, n_boot=n_boot
        )
        for m in metrics
    ]

    raw_ps = [c.p_raw for c in comparisons]
    adjusted = benjamini_hochberg(raw_ps)
    for c, p_adj in zip(comparisons, adjusted, strict=True):
        c.p_adjusted = p_adj
        c.significant = p_adj < ALPHA

    result = StatsResult(
        comparisons=comparisons,
        correction="Benjamini–Hochberg (FDR)",
        n_tests=len(comparisons),
        conclusion=_conclude(comparisons, label_a, label_b),
        limitations=_limitations(comparisons),
    )
    return result


def _conclude(comparisons: Sequence[Comparison], label_a: str, label_b: str) -> str:
    """生成**带范围限定**的结论。

    绝不输出无限定的普遍结论 —— 样本量小时那样写就是过度声称。
    """
    sig = [c for c in comparisons if c.significant]
    n = len(comparisons)
    sample = min(c.n for c in comparisons)

    if not sig:
        return (
            f"在 {n} 项指标、每项 {sample} 条配对样本上，"
            f"{label_b} 与 {label_a} 没有显著差异（BH 校正后 p ≥ {ALPHA}）。"
            f"这**不证明**两者等价 —— 只能说本样本量下没检出差异。"
        )

    gains = [c for c in sig if c.diff > 0]
    losses = [c for c in sig if c.diff < 0]
    parts: list[str] = []
    if gains:
        names = "、".join(c.metric for c in gains)
        parts.append(f"{names} 上 {label_b} 显著优于 {label_a}")
    if losses:
        names = "、".join(c.metric for c in losses)
        parts.append(f"{names} 上 {label_b} 显著劣于 {label_a}")
    body = "；".join(parts)

    return (
        f"在 {n} 项指标、每项 {sample} 条配对样本上，BH 校正后有 {len(sig)} 项显著：{body}。"
        f"**该结论仅限本实验的样本与设置**，不能直接外推到其他数据集。"
    )


def _limitations(comparisons: Sequence[Comparison]) -> list[str]:
    n = min((c.n for c in comparisons), default=0)
    out = [
        f"样本量 {n} 条配对观测 —— 只能检出较大效应，小效应可能被漏掉",
        "只做了配对检验，未做功效分析（power analysis）",
        "结论限定在本实验的数据与设置内",
    ]
    if any(abs(c.effect_size) < 0.2 and c.significant for c in comparisons):
        out.append(
            "存在统计显著但效应量很小（|dz| < 0.2）的指标 —— 实际意义有限，需谨慎表述"
        )
    return out
