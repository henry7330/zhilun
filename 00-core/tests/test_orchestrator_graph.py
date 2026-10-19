"""主 Agent 的 LangGraph 版测试。

## 核心验收

1. **两种编排等价**：同一请求 → 同样的计划、同样的结论（否则用户不知道该信哪个）
2. **图结构正确**：plan / approve / pick / run / summarize，且 `pick → run → pick` 是循环
3. **`interrupt` 真的能暂停**：确认前不执行任何阶段
4. **拒绝执行要如实说**，不能伪装成"通过了"
5. **状态可序列化**：`StageReport` 往返一致，否则 checkpoint 读不回来
6. 包级导入是**延迟**的：`import copilot.orchestrator` 不拉起 langgraph
"""

from __future__ import annotations

import pytest

from copilot.contracts import Finding, NodeResult, StageReport
from copilot.orchestrator import run_request
from copilot.orchestrator.graph import (
    OrchestratorRun,
    _result_from_dict,
    _result_to_dict,
    build_orchestrator_graph,
    run_request_graph,
)
from copilot.quality.format.report import Severity

DRAFT = """# 摘要

本文研究长尾推荐问题。

# 方法

本文设计了两阶段策略。

# 结论

方法有效。
"""

CTX = {
    "title": "图神经网络推荐",
    "draft": DRAFT,
    "requirement": "正文：宋体小四",
    "report": "本文设计了两阶段策略。",
}


def _two_stage(**over) -> dict:
    return {**CTX, **over}


# ─────────────────── 图结构 ───────────────────


def test_graph_has_expected_nodes() -> None:
    app = build_orchestrator_graph(context=CTX)
    mermaid = app.get_graph().draw_mermaid()
    for node in ("plan", "approve", "pick", "run", "summarize"):
        assert node in mermaid


def test_run_loops_back_to_pick() -> None:
    """★ `pick → run → pick` 是循环 —— 手写 for 在图里的等价形式。"""
    mermaid = build_orchestrator_graph(context=CTX).get_graph().draw_mermaid()
    assert "run --> pick" in mermaid


def test_approval_requires_checkpointer() -> None:
    """★ interrupt 暂停后状态必须落盘，没有 checkpointer 就没法恢复。"""
    with pytest.raises(ValueError, match="checkpointer"):
        build_orchestrator_graph(context=CTX, require_approval=True)


# ─────────────────── 顺序执行 ───────────────────


def test_runs_all_stages_without_approval() -> None:
    run = run_request_graph(
        "把格式整理一下，再看看有没有重复的段落", context=CTX
    )
    report = run.to_report()
    assert report.plan.stages == ["E", "F"]
    assert len(report.ran_steps) == 2


def test_no_match_finishes_without_running() -> None:
    run = run_request_graph("随便说点什么", context=CTX)
    report = run.to_report()
    assert report.plan.steps == []
    assert report.steps == []
    assert "没有任何阶段被执行" in report.summary_line()


def test_missing_input_is_skipped() -> None:
    run = run_request_graph("帮我降重", context={"draft": DRAFT})
    report = run.to_report()
    assert report.steps[0].ran is False
    assert "外部查重" in report.steps[0].skipped_reason


def test_skipped_only_is_not_a_pass() -> None:
    run = run_request_graph("帮我降重", context={"draft": DRAFT})
    assert run.to_report().can_submit is False


# ─────────────────── 两条入口等价 ───────────────────


@pytest.mark.parametrize(
    "request_text",
    [
        "帮我看看格式",
        "帮我降重",
        "引用来源靠谱吗",
        "把格式整理一下，再看看有没有重复的段落",
        "随便说点什么",
    ],
)
def test_graph_matches_manual_version(request_text: str) -> None:
    """★ 两种编排必须给出一致的计划与结论 —— 否则用户不知道该信哪个。"""
    manual = run_request(request_text, context=CTX)
    graphed = run_request_graph(request_text, context=CTX).to_report()

    assert manual.plan.describe() == graphed.plan.describe()
    assert [s.step.nodes for s in manual.steps] == [s.step.nodes for s in graphed.steps]
    assert manual.summary_line() == graphed.summary_line()


def test_graph_and_manual_agree_on_node_scoping() -> None:
    """节点级路由（only）在图上也要生效。"""
    request_text = "字体和字号对不对"
    manual = run_request(request_text, context=CTX)
    graphed = run_request_graph(request_text, context=CTX).to_report()
    assert manual.plan.steps[0].nodes == ("E1",)
    assert graphed.plan.steps[0].nodes == ("E1",)
    # 只跑了 E1
    assert [r.node for r in graphed.steps[0].report.results] == ["E1"]


# ─────────────────── interrupt 计划确认 ───────────────────


def test_interrupt_pauses_before_running_anything() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    run = run_request_graph(
        "把格式整理一下，再看看有没有重复的段落",
        context=CTX, require_approval=True,
        checkpointer=InMemorySaver(), thread_id="t-approve",
    )
    assert run.needs_approval is True
    assert run.state.get("results") == []          # ★ 确认前一步都没跑


def test_pending_payload_exposes_the_plan() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    run = run_request_graph(
        "把格式整理一下，再看看有没有重复的段落",
        context=CTX, require_approval=True,
        checkpointer=InMemorySaver(), thread_id="t-pending",
    )
    payload = run.pending
    assert payload is not None
    assert payload["stages"] == ["E", "F"]
    assert len(payload["steps"]) == 2


def test_rejecting_the_plan_runs_nothing() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    run = run_request_graph(
        "帮我看看格式", context=CTX, require_approval=True,
        checkpointer=InMemorySaver(), thread_id="t-reject",
    )
    after = run.resume(False)
    report = after.to_report()
    assert report.ran_steps == []
    assert report.can_submit is False


def test_approving_the_plan_runs_it() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    run = run_request_graph(
        "帮我看看格式", context=CTX, require_approval=True,
        checkpointer=InMemorySaver(), thread_id="t-approve-yes",
    )
    after = run.resume(True)
    report = after.to_report()
    assert len(report.ran_steps) == 1
    assert report.ran_steps[0].report.stage == "E"


def test_resume_returns_a_new_run_object() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    run = run_request_graph(
        "帮我看看格式", context=CTX, require_approval=True,
        checkpointer=InMemorySaver(), thread_id="t-new-obj",
    )
    after = run.resume(True)
    assert isinstance(after, OrchestratorRun)
    assert after is not run
    assert after.needs_approval is False


# ─────────────────── 状态可序列化 ───────────────────


def test_step_result_roundtrips_through_dict() -> None:
    """★ checkpoint 读回来靠这一对函数；往返不一致，恢复就会出错。"""
    from copilot.orchestrator.models import PlanStep, StepResult

    report = StageReport(
        stage="E",
        label="t",
        results=[
            NodeResult(
                node="E1",
                findings=[
                    Finding("E1", "X", Severity.WARNING, "m", location="l", fix="f")
                ],
                stats={"a": 1},
                notes=["n"],
            )
        ],
    )
    original = StepResult(
        step=PlanStep(stage="E", nodes=("E1",), model="m", reason="r"),
        report=report,
    )
    back = _result_from_dict(_result_to_dict(original))
    assert back.step == original.step
    assert back.ran is True
    assert back.report.to_dict() == report.to_dict()


def test_skipped_result_roundtrips() -> None:
    from copilot.orchestrator.models import PlanStep, StepResult

    original = StepResult(
        step=PlanStep(stage="F"), skipped_reason="缺少输入：外部报告"
    )
    back = _result_from_dict(_result_to_dict(original))
    assert back.ran is False
    assert back.skipped_reason == "缺少输入：外部报告"


def test_stage_report_recomputes_can_submit() -> None:
    """`can_submit` 不缓存 —— 读回后重算，避免缓存值与 findings 不一致。"""
    report = StageReport(
        stage="E",
        label="t",
        results=[NodeResult("E1", findings=[Finding("E1", "X", Severity.ERROR, "m")])],
    )
    assert report.can_submit is False
    back = StageReport.from_dict(report.to_dict())
    assert back.can_submit is False


# ─────────────────── 延迟导入 ───────────────────


def test_manual_version_does_not_import_langgraph() -> None:
    """★ `import copilot.orchestrator` 不应当拉起 langgraph（延迟到调用时）。"""
    import copilot.orchestrator.execute as ex

    src = open(ex.__file__, encoding="utf-8").read()
    assert "from langgraph" not in src
    assert "import langgraph" not in src
