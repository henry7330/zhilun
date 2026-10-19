"""主 Agent 的 **LangGraph 版本**。

## 为什么要有这一版

主 Agent 的执行走这张图。图能提供三样手写循环拿不到的东西：

| 能力 | 手写 `for` 循环 | 本图 |
|---|---|---|
| **执行前让用户确认计划** | 只能调两次（先 `plan_only` 看，再执行） | `interrupt()` —— **同一个请求内暂停** |
| **跑一半崩了能续** | 不能，D 跑完崩了，E/F 要重来 | `checkpointer` + `thread_id` 从断点续 |
| **编排长什么样** | 只能读代码 | 能画成图（`get_graph().draw_mermaid()`） |

所以图的形状是：

```
START → plan ─┬─(有 step 且需确认)→ approve ─┬─(同意)→ pick
              │                              └─(拒绝)→ summarize
              ├─(有 step 不需确认)→ pick
              └─(没匹配到阶段)→ summarize

pick ─┬─(还有 step)→ run → pick
      └─(跑完了)→ summarize → END
```

**`pick → run → pick` 是一个循环** —— 这正是手写 `for` 循环在图里的等价形式。
它也是 LangGraph 里做"动态工作流"的标准写法：**用条件边表达 while**。

## 粒度：图里只有 6 个阶段节点吗

不是 6 个 —— 是**一个通用 `run` 节点**，它按 `step.stage` 分派到对应的 `run_*`。

**为什么不做成 6 个节点**：主 Agent 不需要"边"来决定走哪个阶段（那是 `plan` 决定的），
它只需要"跑下一个 step"。做成 6 个节点会让图更花，但没有任何额外能力。

**节点级路由怎么体现**：`step.nodes` 传给 `run_*` 的 `only` 参数 ——
所以"只跑 E1"仍然是成立的，只是它发生在 `run` 节点**内部**，不是图的一条边。

## 状态里能放什么

**只能放可序列化的东西**（dict / list / str / int / bool）。

- `plan` 存 `Plan.to_dict()`
- `results` 存每个 step 的结果 dict（含 `StageReport.to_dict()`）

**`context` 不能进状态** —— 里面有 `rewriter` / `formatter` / `generator` 这类对象。
它们通过闭包传进节点（图是每次请求现建的，所以闭包有效）。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypedDict

from copilot.contracts import StageReport
from copilot.orchestrator.execute import build_plan, execute_step
from copilot.orchestrator.models import (
    OrchestrationReport,
    Plan,
    PlanStep,
    StepResult,
)
from copilot.orchestrator.plan import Planner
from copilot.orchestrator.router import ModelRouter

__all__ = [
    "OrchestratorState",
    "OrchestratorRun",
    "build_orchestrator_graph",
    "run_request_graph",
]


class OrchestratorState(TypedDict, total=False):
    """图的状态。**只放可序列化的东西。**"""

    request: str
    plan: dict[str, Any]
    cursor: int
    results: list[dict[str, Any]]
    approved: bool
    summary: str


# ─────────────────── 序列化辅助 ───────────────────


def _step_to_dict(step: PlanStep) -> dict[str, Any]:
    return {
        "stage": step.stage,
        "nodes": list(step.nodes) if step.nodes else None,
        "model": step.model,
        "reason": step.reason,
    }


def _step_from_dict(data: Mapping[str, Any]) -> PlanStep:
    nodes = data.get("nodes")
    return PlanStep(
        stage=str(data.get("stage") or ""),
        nodes=tuple(nodes) if nodes else None,
        model=data.get("model"),
        reason=str(data.get("reason") or ""),
    )


def _plan_to_dict(plan: Plan) -> dict[str, Any]:
    return {
        "request": plan.request,
        "steps": [_step_to_dict(s) for s in plan.steps],
        "notes": list(plan.notes),
        "missing": list(plan.missing),
    }


def _plan_from_dict(data: Mapping[str, Any]) -> Plan:
    return Plan(
        request=str(data.get("request") or ""),
        steps=[_step_from_dict(s) for s in (data.get("steps") or [])],
        notes=list(data.get("notes") or []),
        missing=list(data.get("missing") or []),
    )


def _result_to_dict(result: StepResult) -> dict[str, Any]:
    return {
        "step": _step_to_dict(result.step),
        "ran": result.ran,
        "skipped_reason": result.skipped_reason,
        "report": result.report.to_dict() if result.ran else None,
    }


def _result_from_dict(data: Mapping[str, Any]) -> StepResult:
    raw = data.get("report")
    return StepResult(
        step=_step_from_dict(data.get("step") or {}),
        report=StageReport.from_dict(raw) if raw else None,
        skipped_reason=str(data.get("skipped_reason") or ""),
    )


# ─────────────────── 建图 ───────────────────


def build_orchestrator_graph(
    *,
    context: Mapping[str, Any] | None = None,
    planner: Planner | None = None,
    router: ModelRouter | None = None,
    require_approval: bool = False,
    checkpointer: Any = None,
    on_step: Any = None,
):
    """编译主 Agent 的 StateGraph。

    ``require_approval=True`` 会在执行前用 ``interrupt`` 暂停等人确认 ——
    **必须配 checkpointer**，否则 LangGraph 会直接报错（暂停后状态无处可存）。
    """
    from langgraph.graph import END, START, StateGraph

    if require_approval and checkpointer is None:
        raise ValueError(
            "require_approval=True 需要 checkpointer："
            "interrupt 暂停后状态必须落盘，否则无法恢复。"
        )

    ctx: dict[str, Any] = dict(context or {})

    # ── plan：把请求变成计划 ──
    def plan_node(state: OrchestratorState) -> dict[str, Any]:
        plan = build_plan(state["request"], ctx, planner=planner, router=router)
        return {"plan": _plan_to_dict(plan), "cursor": 0, "results": []}

    # ── approve：计划确认（可选）──
    def approve_node(state: OrchestratorState) -> dict[str, Any]:
        from langgraph.types import interrupt

        plan = state["plan"]
        decision = interrupt(
            {
                "question": "是否执行这个计划？",
                "stages": [s["stage"] for s in plan["steps"]],
                "steps": plan["steps"],
                "missing": plan["missing"],
            }
        )
        # interrupt 恢复后这个节点会**从头重跑**，
        # 所以这里必须无副作用 —— 只把决定写回状态。
        return {"approved": bool(decision)}

    # ── pick：什么都不做，只作为"要不要继续"的判定点 ──
    def pick_node(state: OrchestratorState) -> dict[str, Any]:
        return {}

    # ── run：跑当前 step ──
    def run_node(state: OrchestratorState) -> dict[str, Any]:
        steps = state["plan"]["steps"]
        index = int(state.get("cursor") or 0)
        step = _step_from_dict(steps[index])
        result = execute_step(step, ctx)
        if on_step:
            on_step(step, result)
        return {
            "results": [*(state.get("results") or []), _result_to_dict(result)],
            "cursor": index + 1,
        }

    # ── summarize：收尾 ──
    def summarize_node(state: OrchestratorState) -> dict[str, Any]:
        results = [_result_from_dict(r) for r in (state.get("results") or [])]
        plan = _plan_from_dict(state["plan"])
        report = OrchestrationReport(
            request=state["request"], plan=plan, steps=results
        )
        return {"summary": report.summary_line()}

    # ── 条件边 ──
    def after_plan(state: OrchestratorState) -> str:
        if not state["plan"]["steps"]:
            return "summarize"
        return "approve" if require_approval else "pick"

    def after_approve(state: OrchestratorState) -> str:
        return "pick" if state.get("approved") else "summarize"

    def after_pick(state: OrchestratorState) -> str:
        steps = state["plan"]["steps"]
        return "run" if int(state.get("cursor") or 0) < len(steps) else "summarize"

    builder = StateGraph(OrchestratorState)
    builder.add_node("plan", plan_node)
    builder.add_node("approve", approve_node)
    builder.add_node("pick", pick_node)
    builder.add_node("run", run_node)
    builder.add_node("summarize", summarize_node)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan", after_plan,
        {"approve": "approve", "pick": "pick", "summarize": "summarize"},
    )
    builder.add_conditional_edges(
        "approve", after_approve, {"pick": "pick", "summarize": "summarize"}
    )
    builder.add_conditional_edges(
        "pick", after_pick, {"run": "run", "summarize": "summarize"}
    )
    builder.add_edge("run", "pick")
    builder.add_edge("summarize", END)

    return builder.compile(checkpointer=checkpointer)


# ─────────────────── 一次运行 ───────────────────


@dataclass
class OrchestratorRun:
    """LangGraph 版的一次运行：**要么跑完了，要么停在等确认**。"""

    app: Any
    thread_id: str
    request: str
    state: dict[str, Any]

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}}

    @property
    def needs_approval(self) -> bool:
        return bool(self.state.get("__interrupt__"))

    @property
    def pending(self) -> dict[str, Any] | None:
        """等确认时，把待确认的计划原样交出去。"""
        raw = self.state.get("__interrupt__")
        if not raw:
            return None
        first = raw[0]
        return getattr(first, "value", first)

    def resume(self, approved: bool) -> "OrchestratorRun":
        """确认计划。``approved=False`` 表示放弃执行，只留计划。"""
        from langgraph.types import Command

        state = self.app.invoke(Command(resume=bool(approved)), self.config)
        return OrchestratorRun(
            app=self.app, thread_id=self.thread_id, request=self.request,
            state=dict(state),
        )

    def to_report(self) -> OrchestrationReport:
        """把状态还原成 `OrchestrationReport`。"""
        plan = _plan_from_dict(self.state.get("plan") or {"request": self.request})
        results = [_result_from_dict(r) for r in (self.state.get("results") or [])]
        return OrchestrationReport(request=self.request, plan=plan, steps=results)


def run_request_graph(
    request: str,
    *,
    context: Mapping[str, Any] | None = None,
    planner: Planner | None = None,
    router: ModelRouter | None = None,
    require_approval: bool = False,
    checkpointer: Any = None,
    thread_id: str | None = None,
    on_step: Any = None,
) -> OrchestratorRun:
    """**主 Agent 的入口（LangGraph 版）**。

    ```python
    app = build_orchestrator_graph(checkpointer=InMemorySaver(), require_approval=True)
    run = run_request_graph("帮我降重", context={...}, require_approval=True,
                            checkpointer=app_saver, thread_id="t1")
    if run.needs_approval:
        print(run.pending)          # 先看计划
        run = run.resume(True)      # 确认后继续
    print(run.to_report().summary_line())
    ```
    """
    app = build_orchestrator_graph(
        context=context, planner=planner, router=router,
        require_approval=require_approval, checkpointer=checkpointer,
        on_step=on_step,
    )
    tid = thread_id or uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": tid}}
    state = app.invoke({"request": request}, config)
    return OrchestratorRun(app=app, thread_id=tid, request=request, state=dict(state))
