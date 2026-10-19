"""统一契约测试 —— **6 个 Agent（阶段）+ 节点**的架构。

★ 术语纠正（本项目曾经写错）
--------------------------
**一个阶段 = 一个 Agent**，共 6 个；Agent 内部的工作内容叫**节点**。

早期把 20 个节点都叫 "Agent" 是概念错误：Agent 应当是"一个能独立对外
提供服务的单位"——那正是阶段。节点只是它内部的工作步骤。

核心验收：
1. `STAGES` 恰好 6 个，每个阶段的节点清单完整；
2. `NodeResult` / `StageReport` 只依赖 `Finding`，**不认识任何具体节点**；
3. **节点失败按阻断处理** —— 跑挂的检查不等于"这项没问题"。
"""

from __future__ import annotations

import json

import pytest

from copilot.contracts import (
    STAGES,
    Finding,
    NodeResult,
    NodeSpec,
    QualityReport,
    StageReport,
    build_report,
    describe_stages,
    finding_from_issue,
    implemented_stages,
    stage_specs,
)
from copilot.quality.format.report import Issue, Severity

# ─────────────────── Finding ───────────────────


def test_finding_keeps_node_specific_payload() -> None:
    """逃生舱：各节点特有的数据原样保留，合并层不解释也不丢。"""
    f = Finding(
        node="E3", code="LOC-FOUND", severity=Severity.INFO, message="已定位",
        location="第 12 行", payload={"confidence": 0.87},
    )
    assert f.payload["confidence"] == 0.87
    assert f.to_dict()["node"] == "E3"


def test_finding_agent_property_is_backward_compatible() -> None:
    """早期字段名是 ``agent``，保留只读属性避免旧代码全炸。"""
    f = Finding(node="E1", code="SEC001", severity=Severity.ERROR, message="x")
    assert f.agent == "E1"


def test_finding_from_issue_carries_line_and_fix() -> None:
    issue = Issue(code="SEC001", severity=Severity.ERROR, rule="structure",
                  line=3, message="缺少结论章", fix="补写结论")
    f = finding_from_issue("E1", issue)
    assert f.node == "E1"
    assert "第" in f.location and "4" in f.location
    assert f.fix == "补写结论"


# ─────────────────── NodeResult ───────────────────


def test_node_result_counts_by_severity() -> None:
    r = NodeResult(node="E1", findings=[
        Finding("E1", "A", Severity.ERROR, "e"),
        Finding("E1", "B", Severity.WARNING, "w"),
    ])
    assert r.count() == 2
    assert r.count(Severity.ERROR) == 1


def test_failed_node_keeps_error() -> None:
    r = NodeResult(node="E1", ok=False, error="ParseError: 坏输入")
    assert "失败" in r.summary_line()
    assert r.to_dict()["error"] == "ParseError: 坏输入"


# ─────────────────── ★ 架构：6 个 Agent ───────────────────


def test_exactly_six_agents() -> None:
    """★ 一个阶段 = 一个 Agent。"""
    assert len(STAGES) == 6
    assert sorted(STAGES) == ["A", "B", "C", "D", "E", "F"]


def test_each_agent_has_nodes() -> None:
    for key, stage in STAGES.items():
        assert stage.nodes, f"{key} 阶段没有节点"
        assert stage.nodes, f"{key} 阶段节点清单为空"


def test_node_codes_are_prefixed_by_stage() -> None:
    """节点编号必须以所属阶段字母开头 —— 否则报告里看不出它属于谁。"""
    for key, stage in STAGES.items():
        for node in stage.nodes:
            assert node.code.startswith(key), f"{node.code} 不属于阶段 {key}"


def test_all_seventeen_nodes_are_present() -> None:
    """六阶段链路：A3 + B3 + C2 + D4 + E3 + F2 = 17 个节点。

    原来是 20 个（质检 6 个 + 交付 3 个）。简化后：
    质检拆成「格式规整」3 个 + 「降重降AI」2 个，交付取消，
    写作补了 D4 文献引用 —— 20 - 6 - 3 + 3 + 2 + 1 = 17。
    """
    total = sum(len(s.nodes) for s in STAGES.values())
    assert total == 17


def test_node_count_per_stage() -> None:
    assert [len(STAGES[k].nodes) for k in "ABCDEF"] == [3, 3, 2, 4, 3, 2]


def test_stage_names_match_the_six_link_chain() -> None:
    """阶段名必须就是用户要的那条链路，不能含糊。"""
    assert [STAGES[k].name for k in "ABCDEF"] == [
        "选题分析", "文献检索", "研究路线", "论文写作", "格式规整", "降重降AI",
    ]


def test_every_agent_has_an_entry_point() -> None:
    for key, stage in STAGES.items():
        assert stage.entry, f"{key} 阶段缺少对外入口"
        assert ".run_" in stage.entry, stage.entry


def test_all_stages_are_implemented() -> None:
    """A–D 补齐后，6 个阶段应当全部可用。"""
    assert implemented_stages() == ["A", "B", "C", "D", "E", "F"]


def test_requires_model_is_derived_from_nodes() -> None:
    """任一节点需要模型，该 Agent 就需要"可选的模型接入"。"""
    assert STAGES["C"].requires_model is True     # C1 需要
    assert STAGES["A"].requires_model is True     # A2/A3 需要
    assert STAGES["E"].requires_model is True     # E1 格式规整需要
    assert all(isinstance(s.requires_model, bool) for s in STAGES.values())


def test_stage_node_lookup() -> None:
    assert STAGES["E"].node("E1").name == "格式规整"
    with pytest.raises(KeyError):
        STAGES["E"].node("C1")


def test_format_stage_has_three_nodes() -> None:
    """E 阶段 = 规整 → 一致性 → 导出。"""
    assert [n.name for n in STAGES["E"].nodes] == ["格式规整", "文本一致性", "导出"]


def test_revise_stage_has_two_nodes() -> None:
    """F 阶段 = 定位 → 合并改写。"""
    assert [n.name for n in STAGES["F"].nodes] == ["报告解析定位", "合并改写"]


def test_writing_stage_ends_with_citation_check() -> None:
    """D4 必须在最后 —— 先有正文，才谈得上核对引用来源。"""
    names = [n.name for n in STAGES["D"].nodes]
    assert names[-1] == "文献引用"


def test_describe_stages_renders_six_rows() -> None:
    md = describe_stages()
    assert md.startswith("| 阶段 |")
    assert md.count("\n") == 7   # 表头 + 分隔 + 6 行


def test_stage_spec_sorted_and_serializable() -> None:
    keys = [s.key for s in stage_specs()]
    assert keys == sorted(keys)
    d = STAGES["E"].to_dict()
    assert d["key"] == "E"
    assert len(d["nodes"]) == 3
    json.dumps(d, ensure_ascii=False)   # 必须可序列化


def test_node_spec_defaults() -> None:
    n = NodeSpec(code="X1", name="测试")
    assert n.requires_model is False
    assert n.implemented is False


# ─────────────────── StageReport ───────────────────


def _result(node: str, *severities: Severity) -> NodeResult:
    return NodeResult(node=node, findings=[
        Finding(node, f"C{i}", s, f"m{i}") for i, s in enumerate(severities)
    ])


def test_report_merges_results_from_many_nodes() -> None:
    report = build_report("x.md", [
        _result("E1", Severity.ERROR, Severity.WARNING),
        _result("E6", Severity.INFO),
    ], stage="E")
    assert report.counts()["nodes"] == 2
    assert report.counts()["error"] == 1
    assert report.stage == "E"


def test_report_blocks_on_error() -> None:
    report = build_report("x.md", [_result("E1", Severity.ERROR)], stage="E")
    assert not report.can_submit
    assert "未通过" in report.verdict()


def test_report_allows_when_only_warnings() -> None:
    report = build_report("x.md", [_result("E1", Severity.WARNING)], stage="E")
    assert report.can_submit


def test_report_blocks_when_a_node_failed() -> None:
    """★ 跑挂的检查不等于"这项没问题" —— 必须阻断。"""
    report = build_report("x.md", [
        _result("E1"),
        NodeResult(node="E2", ok=False, error="网络超时"),
    ], stage="E")
    assert not report.can_submit
    assert "E2" in report.verdict()


def test_report_markdown_has_verdict_before_details() -> None:
    md = build_report("x.md", [_result("E1", Severity.ERROR)], stage="E").to_markdown()
    assert md.index("结论") < md.index("阻断项")


def test_report_dict_is_json_serializable() -> None:
    report = build_report("x.md", [_result("E1", Severity.ERROR)], stage="E")
    payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    assert payload["can_submit"] is False
    assert payload["stage"] == "E"


def test_empty_report_is_technically_submittable() -> None:
    assert StageReport(label="x.md").can_submit


def test_quality_report_alias_still_works() -> None:
    """早期名字保留，旧代码不用全改。"""
    assert QualityReport is StageReport
