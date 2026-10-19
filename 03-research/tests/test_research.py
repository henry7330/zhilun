"""C 阶段（研究）测试。

核心验收：
1. 设计缺失项**一律是阻断项**（骨架也不例外 —— 骨架不是完成品）；
2. 没有有效对照组时，分析结论**被判为不成立**；
3. 多指标必须做 **BH 校正**，且以校正后 p 值判显著；
4. 结论**必须带范围限定**，不允许无限定的普遍结论。
"""

from __future__ import annotations

import pytest

from copilot.research import analyze_results, run_research
from copilot.research.analysis import analyze_paired
from copilot.research.models import (
    PLACEHOLDER_BASELINE,
    Metric,
    Observation,
    ResearchDesign,
)


def _obs(n: int = 20, *, delta: float = 0.05) -> list[Observation]:
    return [Observation(f"u{i:02d}", 0.30 + 0.001 * i, 0.30 + 0.001 * i + delta) for i in range(n)]


# ─────────────────── 设计校验 ───────────────────


def test_placeholder_baseline_is_a_problem() -> None:
    design = ResearchDesign(question="q", baselines=[PLACEHOLDER_BASELINE])
    assert any("对照" in p for p in design.validate())


def test_design_without_baselines_is_flagged() -> None:
    design = ResearchDesign(question="q", method="m")
    assert any("对照" in p for p in design.validate())


def test_metric_requires_rationale() -> None:
    """选错指标是常见的方法论硬伤 —— 必须说明为什么选它。"""
    design = ResearchDesign(question="q", method="m", baselines=["b"],
                            metrics=[Metric(name="准确率", rationale="")])
    assert any("为什么选它" in p for p in design.validate())


def test_complete_design_validates_clean() -> None:
    design = ResearchDesign(
        question="q",
        method="两阶段",
        alternatives=["端到端"],
        baselines=["BM25"],
        metrics=[Metric("nDCG@10", rationale="排序质量的标准指标")],
        sample_size={"n": 30},
        risks=["长尾覆盖不足"],
    )
    assert design.validate() == []


# ─────────────────── 分析 ───────────────────


def test_analyze_paired_reports_three_quantities() -> None:
    c = analyze_paired(_obs(), metric="ndcg")
    assert c.n == 20
    assert c.diff_ci_low <= c.diff <= c.diff_ci_high
    assert c.p_raw >= 0
    assert c.effect_size != 0


def test_analyze_paired_rejects_tiny_sample() -> None:
    with pytest.raises(ValueError, match="至少需要 2"):
        analyze_paired([Observation("u1", 1.0, 2.0)], metric="x")


def test_multiple_metrics_get_bh_adjustment() -> None:
    result = analyze_results({"a": _obs(), "b": _obs(), "c": _obs()})
    assert result.n_tests == 3
    assert "Benjamini" in result.correction
    assert all(c.p_adjusted is not None for c in result.comparisons)
    # 校正后的 p 不会比原始更小
    assert all(c.p_adjusted >= c.p_raw for c in result.comparisons)


def test_significance_uses_adjusted_p() -> None:
    result = analyze_results({"a": _obs()})
    c = result.comparisons[0]
    assert c.significant == (c.p_adjusted < 0.05)


def test_conclusion_is_scoped() -> None:
    """★ 结论必须带范围限定，不允许无限定的普遍结论。"""
    result = analyze_results({"a": _obs()})
    assert "样本" in result.conclusion
    assert "不能直接外推" in result.conclusion or "仅限" in result.conclusion


def test_no_data_means_no_conclusion() -> None:
    result = analyze_results({})
    assert result.comparisons == []
    assert "不编造" in result.limitations[0]
    assert result.limitations


def test_limitations_mention_small_sample() -> None:
    result = analyze_results({"a": _obs(3)})
    assert any("样本量" in x for x in result.limitations)


# ─────────────────── 阶段入口 ───────────────────


def test_skeleton_design_is_blocking() -> None:
    """★ 骨架设计（7 个空字段）必须是阻断项，不能判为通过。

    实测踩过：早期把骨架的缺失报成 WARNING，结果骨架被判 ✅ 通过 ——
    那等于告诉用户"你的研究设计合格了"。
    """
    report = run_research("某个问题")
    assert not report.can_submit
    assert report.by_node()["C1"].count() > 0


def test_missing_baseline_blocks_analysis() -> None:
    """★ 没有对照组的分析结论不成立 —— 必须在报告里拦住。"""
    report = run_research("某个问题", data={"ndcg": _obs()})
    assert any("对照" in f.message for f in report.by_node()["C2"].findings)


def test_analysis_runs_and_reports() -> None:
    report = run_research("某个问题", data={"ndcg": _obs(30)})
    c2 = report.by_node()["C2"]
    assert c2.ok
    assert c2.stats["analyzed"] == 1


def test_stage_runs_independently() -> None:
    report = run_research("某个问题")
    assert report.stage == "C"
    assert {r.node for r in report.results} == {"C1", "C2"}
