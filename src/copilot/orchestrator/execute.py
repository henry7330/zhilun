"""主 Agent 的**执行器**：按 Plan 调阶段、按节点收窄。

## 两个共用函数

本模块把执行拆成两个纯函数，供**两种编排**共用：

| 函数 | 做什么 | 谁在用 |
|---|---|---|
| `build_plan` | 请求 → `Plan`（含模型决策） | 图的 `plan` 节点 |
| `execute_step` | 跑一个 `PlanStep` → `StepResult` | 图的 `run` 节点 |

**为什么抽成函数而不是各写一遍**：两种编排只是"谁在驱动循环"不同，
每一步做什么必须完全一样 —— 复制一份迟早会不一致，而且极难发现。

## 缺输入为什么跳过而不是硬跑

`run_revise(draft, report="")` 会抛 `ValueError` —— 因为**没有报告就没有这个阶段的意义**。

如果直接让异常冒出来，用户看到的是一个栈，而不是
"你要降重，但没给我查重报告"。所以调用前先检查，把它变成**一句人话**。

**跳过 ≠ 通过。** `OrchestrationReport.can_submit` 只在
"有阶段真的跑过、且都通过"时为真。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

from copilot.contracts import StageReport
from copilot.orchestrator.models import (
    REQUIRED_INPUTS,
    OrchestrationReport,
    Plan,
    PlanStep,
    StepResult,
)
from copilot.orchestrator.plan import Planner, RulePlanner
from copilot.orchestrator.router import ModelRouter

__all__ = [
    "run_request",
    "build_plan",
    "execute_step",
    "attach_models",
    "INPUT_LABEL",
    "STAGE_RUNNERS",
]

#: 缺输入时给用户看的说法（不要把字段名直接抛给用户）
INPUT_LABEL: dict[str, str] = {
    "title": "研究题目 / 标题",
    "draft": "论文草稿",
    "question": "研究问题",
    "report": "外部查重 / AI 检测报告",
    "requirement": "学校格式要求",
}


# ─────────────────── 每个阶段的适配器 ───────────────────
#
# 六个阶段的签名各不相同（有的位置参数、有的全关键字），
# 执行器统一用 (ctx, only, on_node) 调用它们 —— 差异收敛在这一层。
#
# 为什么用延迟 import：主 Agent 住在 00-core（共享层），
# 不能在建包时就 import 六个阶段，否则共享层反过来依赖了阶段。


def _run_a(ctx: Mapping[str, Any], only: Sequence[str] | None, on_node) -> StageReport:
    from copilot.topic import run_topic

    return run_topic(
        title=str(ctx["title"]),
        texts=ctx.get("texts") or (),
        years=ctx.get("years"),
        conditions=ctx.get("conditions"),
        judge=ctx.get("judge"),
        generator=ctx.get("generator"),
        top_k=int(ctx.get("top_k") or 20),
        only=only,
        on_node=on_node,
    )


def _run_b(ctx: Mapping[str, Any], only: Sequence[str] | None, on_node) -> StageReport:
    from copilot.literature import run_literature

    return run_literature(
        keywords=ctx.get("keywords") or (),
        corpus=ctx.get("corpus") or (),
        entries=ctx.get("entries") or (),
        library=ctx.get("library"),
        resolver=ctx.get("resolver"),
        k=int(ctx.get("k") or 20),
        label=str(ctx.get("label") or "<inline>"),
        only=only,
        on_node=on_node,
    )


def _run_c(ctx: Mapping[str, Any], only: Sequence[str] | None, on_node) -> StageReport:
    from copilot.research import run_research

    return run_research(
        str(ctx["question"]),
        data=ctx.get("data"),
        planner=ctx.get("design_planner"),
        only=only,
        on_node=on_node,
    )


def _run_d(ctx: Mapping[str, Any], only: Sequence[str] | None, on_node) -> StageReport:
    from copilot.writing import run_writing

    return run_writing(
        title=str(ctx["title"]),
        draft=str(ctx.get("draft") or ""),
        design=ctx.get("design"),
        generator=ctx.get("generator"),
        required_sections=ctx.get("required_sections"),
        entries=ctx.get("entries") or (),
        label=str(ctx.get("label") or "<inline>"),
        only=only,
        on_node=on_node,
    )


def _run_e(ctx: Mapping[str, Any], only: Sequence[str] | None, on_node) -> StageReport:
    from copilot.format_stage import run_format

    return run_format(
        draft=str(ctx["draft"]),
        requirement=str(ctx.get("requirement") or ""),
        formatter=ctx.get("formatter"),
        template=ctx.get("template"),
        anonymity_required=bool(ctx.get("anonymity_required")),
        label=str(ctx.get("label") or "<inline>"),
        only=only,
        on_node=on_node,
    )


def _run_f(ctx: Mapping[str, Any], only: Sequence[str] | None, on_node) -> StageReport:
    from copilot.revise import run_revise

    return run_revise(
        draft=str(ctx["draft"]),
        report=str(ctx["report"]),
        report_type=str(ctx.get("report_type") or "duplicate"),
        rewriter=ctx.get("rewriter"),
        max_attempts=int(ctx.get("max_attempts") or 3),
        label=str(ctx.get("label") or "<inline>"),
        only=only,
        on_node=on_node,
    )


STAGE_RUNNERS: dict[str, Callable[..., StageReport]] = {
    "A": _run_a,
    "B": _run_b,
    "C": _run_c,
    "D": _run_d,
    "E": _run_e,
    "F": _run_f,
}


# ─────────────────── 两个共用函数 ───────────────────


def attach_models(plan: Plan, router: ModelRouter) -> Plan:
    """给每个 step 补上模型决策（第三层路由）。"""
    steps: list[PlanStep] = []
    notes = list(plan.notes)
    for step in plan.steps:
        decision = router.resolve(step.stage, step.nodes)
        steps.append(replace(step, model=decision.model))
        if decision.note:
            notes.append(f"{step.stage}: {decision.note}")
    return Plan(
        request=plan.request, steps=steps, notes=notes, missing=list(plan.missing)
    )


def build_plan(
    request: str,
    context: Mapping[str, Any],
    *,
    planner: Planner | None = None,
    router: ModelRouter | None = None,
) -> Plan:
    """请求 → `Plan`（含三层路由的结果）。"""
    plan = (planner or RulePlanner())(request, dict(context))
    return attach_models(plan, router or ModelRouter())


def execute_step(step: PlanStep, context: Mapping[str, Any]) -> StepResult:
    """跑一个 step。**缺输入 / 未知阶段 / 执行失败都变成可读的跳过原因。**"""
    missing = [
        k for k in REQUIRED_INPUTS.get(step.stage, ())
        if not str(context.get(k) or "").strip()
    ]
    if missing:
        return StepResult(
            step=step,
            skipped_reason="缺少输入：" + "、".join(INPUT_LABEL.get(m, m) for m in missing),
        )

    runner = STAGE_RUNNERS.get(step.stage)
    if runner is None:
        return StepResult(step=step, skipped_reason=f"未知阶段 {step.stage}")

    try:
        report = runner(context, step.nodes, None)
    except Exception as exc:  # noqa: BLE001 - 单个阶段失败不该带崩整次请求
        return StepResult(
            step=step, skipped_reason=f"执行失败：{type(exc).__name__}: {exc}"
        )
    return StepResult(step=step, report=report)


# ─────────────────── 对外入口（走图） ───────────────────


def run_request(
    request: str,
    *,
    context: Mapping[str, Any] | None = None,
    planner: Planner | None = None,
    router: ModelRouter | None = None,
    plan_only: bool = False,
    on_step: Callable[[PlanStep, Any], None] | None = None,
    require_approval: bool = False,
    checkpointer: Any = None,
    thread_id: str | None = None,
) -> OrchestrationReport:
    """**主 Agent 的唯一入口**：一句请求 → 规划 → 执行 → 汇总报告。

    执行走 LangGraph 状态图（`orchestrator/graph.py`）——
    本函数只负责两件事：``plan_only`` 的短路，以及把图的结果转成报告。

    ``plan_only=True`` 只出计划不执行（**不需要图**，因为没有任何节点要跑）。
    ``require_approval=True`` 需要同时给 ``checkpointer`` —— 那会走 ``interrupt``。
    """
    ctx: dict[str, Any] = dict(context or {})

    # 只要计划：不用建图，直接返回
    if plan_only:
        plan = build_plan(request, ctx, planner=planner, router=router)
        return OrchestrationReport(request=request, plan=plan, steps=[])

    from copilot.orchestrator.graph import run_request_graph

    run = run_request_graph(
        request,
        context=ctx,
        planner=planner,
        router=router,
        require_approval=require_approval,
        checkpointer=checkpointer,
        thread_id=thread_id,
        on_step=on_step,
    )
    return run.to_report()
