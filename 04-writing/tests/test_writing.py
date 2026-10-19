"""D 阶段（写作）测试。

核心验收：
1. 大纲骨架来自**模板**（确定性），要点由**规则**分配；
2. 篇幅必须分配（不分配必然导致结构失衡）；
3. D3 只给**具体修复建议**，不自动改文件；
4. **按章编号**（图 2-1）不参与顺序重排 —— 实测过的误报。
"""

from __future__ import annotations

from copilot.quality.format import parse_markdown
from copilot.writing import analyze_figures, build_outline, run_writing
from copilot.writing.outline import assign_points

DESIGN = {
    "method": "两阶段：粗筛 + 精排",
    "alternatives": ["单阶段端到端"],
    "baselines": ["BM25", "矩阵分解"],
    "metrics": [{"name": "nDCG@10"}],
    "sample_size": {"n": 30},
    "risks": ["长尾覆盖不足"],
    "variables": {"independent": ["检索策略"], "dependent": ["nDCG"], "controlled": ["语料"]},
}


# ─────────────────── D1 大纲 ───────────────────


def test_outline_sections_come_from_template() -> None:
    outline = build_outline(title="t", required_sections=["摘要", "方法", "结论"])
    assert [s.title for s in outline.sections] == ["摘要", "方法", "结论"]


def test_outline_distributes_word_budget() -> None:
    """★ 不分配篇幅必然导致结构失衡（绪论 5000 字、方法 300 字）。"""
    outline = build_outline(title="t", required_sections=["摘要", "研究方法", "结论"])
    assert outline.total_words > 0
    words = {s.title: s.target_words for s in outline.sections}
    assert words["研究方法"] > words["摘要"], "方法章应比摘要拿更多篇幅"


def test_points_are_assigned_by_rule() -> None:
    points = assign_points("研究方法", DESIGN)
    assert any("两阶段" in p for p in points)
    assert any("检索策略" in p for p in points)


def test_experiment_chapter_gets_baselines_and_metrics() -> None:
    points = assign_points("实验与分析", DESIGN)
    assert any("BM25" in p for p in points)
    assert any("nDCG@10" in p for p in points)


def test_outline_without_design_is_skeleton() -> None:
    outline = build_outline(title="t", required_sections=["方法"])
    assert outline.skeleton is True
    assert any("没有研究设计" in n for n in outline.notes)


# ─────────────────── D3 图表 ───────────────────


def test_caption_renumbering_is_detected() -> None:
    md = "# 论文\n\n## 方法\n\n**图 2** 框架\n\n**图 3** 流程\n"
    report = analyze_figures(parse_markdown(md))
    fixes = [f for f in report.fixes if "题注编号" in f.what]
    assert len(fixes) == 2
    assert fixes[0].new == "图 1"


def test_reference_to_missing_figure_is_reported_without_guess() -> None:
    """引用了不存在的图：**只报，不给替换值** —— 猜一个数会改坏正文。"""
    md = "# 论文\n\n## 方法\n\n如 图 5 所示。\n\n**图 1** 框架\n"
    report = analyze_figures(parse_markdown(md))
    fixes = [f for f in report.fixes if "引用" in f.what]
    assert fixes
    assert fixes[0].new == "（需人工确认）"


def test_chapter_based_numbering_is_not_renumbered() -> None:
    """★ 实测踩过的误报：`图 2-1` 被当成 `图 2` 参与顺序重排。

    按章编号是刻意的写法，不能重排，也不能据此判断引用是否断裂。
    """
    md = "# 论文\n\n## 方法\n\n如 图 2-1 所示。\n\n**图 2-1** 框架\n\n**图 3-1** 流程\n"
    report = analyze_figures(parse_markdown(md))
    assert report.fixes == []
    assert report.unreferenced == []


def test_unreferenced_caption_is_reported() -> None:
    md = "# 论文\n\n## 方法\n\n**图 1** 框架\n"
    report = analyze_figures(parse_markdown(md))
    assert len(report.unreferenced) == 1


# ─────────────────── 阶段入口 ───────────────────


def test_d2_without_generator_only_says_pending() -> None:
    report = run_writing(title="t", required_sections=["摘要", "结论"])
    d2 = report.by_node()["D2"]
    assert d2.ok
    assert d2.stats["written"] == 0
    assert any("待写" in f.message for f in d2.findings)


def test_d3_skipped_without_draft() -> None:
    report = run_writing(title="t", required_sections=["方法"])
    assert any("跳过" in n for n in report.by_node()["D3"].notes)


def test_stage_runs_independently_with_design() -> None:
    report = run_writing(
        title="基于图神经网络的推荐方法研究",
        design=DESIGN,
        required_sections=["摘要", "研究方法", "实验与分析", "结论"],
    )
    assert report.stage == "D"
    d1 = report.by_node()["D1"]
    assert d1.ok
    assert d1.stats["skeleton"] is False
