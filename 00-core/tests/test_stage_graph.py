"""阶段图（LangGraph 版 run_stage）测试。

## 核心验收

1. **每个阶段内部都是一张状态图**，节点数是 2–4，边是线性的
2. **`only` 在建图前裁剪**：没选中的节点根本不出现在图里
3. **状态只有 `data` + `results`**：新增节点不需要改类型声明
4. **`results` 用 reducer 累加**：否则后跑的节点会冲掉前面的结果
5. **失败隔离**：节点抛异常不炸整张图，但报告判为不可通过
6. **可暂停**：传 checkpointer 就能带 thread_id 跑
"""

from __future__ import annotations

import pytest

from copilot.contracts import Finding, NodeResult
from copilot.quality.format.report import Severity
from copilot.stage import (
    NodeOutcome,
    StageNode,
    StageState,
    build_stage_graph,
    ok_outcome,
    run_stage,
)


def _node(code: str, *, patch: dict | None = None, error: str | None = None,
          when=None) -> StageNode:
    def _run(state: dict) -> NodeOutcome:
        if error:
            raise RuntimeError(error)
        return ok_outcome(
            findings=[Finding(code, f"{code}-X", Severity.INFO, f"{code} 跑了")],
            stats={"saw": sorted(state.keys())},
            notes=[f"{code} 完成"],
            patch=patch or {},
        )

    return StageNode(code=code, name=f"节点{code}", run=_run, when=when)


# ─────────────────── 图结构 ───────────────────


def test_stage_is_a_graph_with_linear_edges() -> None:
    """★ 阶段内部现在是 StateGraph：START → n1 → n2 → END。"""
    nodes = (_node("n1"), _node("n2"), _node("n3"))
    mermaid = build_stage_graph(nodes).compile().get_graph().draw_mermaid()
    assert "n1 --> n2" in mermaid
    assert "n2 --> n3" in mermaid
    assert "__start__ --> n1" in mermaid
    assert "n3 --> __end__" in mermaid


def test_state_has_only_two_keys() -> None:
    """★ 业务字段收进 `data`，所以新增节点不用改状态声明。"""
    assert set(StageState.__annotations__) == {"data", "results"}


def test_unknown_node_in_only_is_an_error() -> None:
    with pytest.raises(KeyError, match="n9"):
        run_stage(stage="X", label="l", nodes=(_node("n1"),), state={}, only=["n9"])


def test_only_removes_nodes_before_building_the_graph() -> None:
    """★ 没选中的节点根本不进图 —— 不是"进了图再跳过"。"""
    nodes = (_node("n1"), _node("n2"), _node("n3"))
    mermaid = build_stage_graph([n for n in nodes if n.code == "n2"]).compile()\
        .get_graph().draw_mermaid()
    assert "n1" not in mermaid and "n3" not in mermaid
    assert "__start__ --> n2" in mermaid


def test_only_empty_runs_nothing() -> None:
    report = run_stage(stage="X", label="l", nodes=(_node("n1"),), state={}, only=[])
    assert report.results == []


# ─────────────────── 状态传递 ───────────────────


def test_patch_flows_to_the_next_node() -> None:
    seen: dict[str, list[str]] = {}

    def first(state: dict) -> NodeOutcome:
        return ok_outcome(patch={"outline": "大纲对象"})

    def second(state: dict) -> NodeOutcome:
        seen["keys"] = sorted(state.keys())
        seen["outline"] = state.get("outline")
        return ok_outcome()

    run_stage(
        stage="X", label="l",
        nodes=(StageNode("n1", "一", first), StageNode("n2", "二", second)),
        state={"title": "t"},
    )
    assert seen["outline"] == "大纲对象"
    assert "title" in seen["keys"], "初始状态也要传给节点"


def test_patch_is_visible_in_stats_of_later_nodes() -> None:
    report = run_stage(
        stage="X", label="l",
        nodes=(_node("n1", patch={"a": 1}), _node("n2", patch={"b": 2})),
        state={"seed": 0},
    )
    n2 = report.by_node()["n2"]
    assert "a" in n2.stats["saw"], "前一节点的产物应当出现在后一节点的输入里"


def test_results_accumulate_not_overwrite() -> None:
    """★ `results` 用 operator.add —— 否则后跑的节点会冲掉前面的。"""
    report = run_stage(
        stage="X", label="l",
        nodes=(_node("n1"), _node("n2"), _node("n3")),
        state={},
    )
    assert [r.node for r in report.results] == ["n1", "n2", "n3"]


# ─────────────────── 失败隔离 ───────────────────


def test_node_failure_does_not_break_the_graph() -> None:
    """★ 一个节点挂了，后面的节点照样跑。"""
    report = run_stage(
        stage="X", label="l",
        nodes=(_node("n1"), _node("n2", error="炸了"), _node("n3")),
        state={},
    )
    assert [r.node for r in report.results] == ["n1", "n2", "n3"]
    assert report.by_node()["n2"].ok is False


def test_failed_node_blocks_submission() -> None:
    """★ 跑挂的检查 ≠ 这项没问题。"""
    report = run_stage(
        stage="X", label="l",
        nodes=(_node("n1"), _node("n2", error="炸了")),
        state={},
    )
    assert report.can_submit is False
    assert "n2" in report.verdict()


def test_conditional_skip_leaves_a_trace() -> None:
    report = run_stage(
        stage="X", label="l",
        nodes=(_node("n1", when=lambda s: False), _node("n2")),
        state={},
    )
    n1 = report.by_node()["n1"]
    assert n1.ok is True
    assert any("跳过" in n for n in n1.notes), "when 跳过要留痕（区别于 only 排除）"


# ─────────────────── 回调与暂停能力 ───────────────────


def test_on_node_callback_fires_in_order() -> None:
    seen: list[str] = []
    run_stage(
        stage="X", label="l",
        nodes=(_node("n1"), _node("n2")),
        state={},
        on_node=lambda code, _r: seen.append(code),
    )
    assert seen == ["n1", "n2"]


def test_checkpointer_supports_resume_capability() -> None:
    """★ 传 checkpointer 就能带 thread_id 跑 —— 暂停恢复的接口是通的。"""
    from langgraph.checkpoint.memory import InMemorySaver

    report = run_stage(
        stage="X", label="l",
        nodes=(_node("n1"), _node("n2")),
        state={},
        checkpointer=InMemorySaver(),
        thread_id="t-stage",
    )
    assert [r.node for r in report.results] == ["n1", "n2"]


def test_report_marks_the_stage() -> None:
    report = run_stage(stage="E", label="l", nodes=(_node("n1"),), state={})
    assert report.stage == "E"


# ─────────────────── 真实阶段也是图 ───────────────────


def test_real_stage_graph_has_its_nodes() -> None:
    """E 阶段的图应当正好是 E1 → E2 → E3。"""
    from copilot.format_stage import build_format_nodes

    nodes = build_format_nodes(draft="# 方法\n\n正文。\n", requirement="正文：宋体小四")
    mermaid = build_stage_graph(nodes).compile().get_graph().draw_mermaid()
    assert "E1 --> E2" in mermaid
    assert "E2 --> E3" in mermaid


def test_node_result_roundtrips_through_graph_state() -> None:
    """图里存的是 `NodeResult.to_dict()`，还原后必须一致。"""
    result = NodeResult(
        node="n1",
        findings=[Finding("n1", "C", Severity.WARNING, "m", location="l")],
        stats={"a": 1},
        notes=["n"],
    )
    back = NodeResult.from_dict(result.to_dict())
    assert back.to_dict() == result.to_dict()
