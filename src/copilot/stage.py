"""阶段运行器：把**一个 Agent 内部的节点**编成一张 LangGraph 状态图。

## 它在编排层里的位置

```
主 Agent（orchestrator/）      决定调哪几个阶段、每个阶段跑哪几个节点
        │
        ▼
本模块 run_stage()             把该阶段的节点编成 StateGraph 并跑完
        │
        ▼
各节点的具体实现
```

**六个阶段全部经由本模块执行** —— 也就是说，系统里每一个子 Agent
内部都是一张 LangGraph 状态图。主 Agent 调它时传 ``only=[...]``，
就能只跑阶段里的某几个节点（图的节点在编译前就被裁掉）。

## 图长什么样

每个阶段的节点**线性串行**，所以图就是一条链：

```
START → e1_normalize → e2_consistency → e3_export → END
```

看起来简单，但用图的三个理由都是实的：

| 理由 | 说明 |
|---|---|
| **统一范式** | 全系统只有 LangGraph 一种编排方式，没有"这里用图那里用 for" |
| **可观测** | `app.get_graph().draw_mermaid()` 能直接把阶段内部画出来 |
| **可暂停** | 传 `checkpointer` 就能中断恢复 —— 现在默认不传，但接口是通的 |

## ★ 状态设计：为什么只有一个 `data` 键

LangGraph 的图状态**只保留 `TypedDict` 里声明过的键**。
少声明一个，那个产物就**静默消失**（不报错，运行时表现为「下游拿到 None」）。

六个阶段往 ``patch`` 里写的键有三十多个且会持续增加，
逐个声明在同一个 TypedDict 里非常容易漏 —— 一漏就是静默数据丢失。

所以本模块把业务字段收进**一个** ``data`` 字典：

```python
class StageState(TypedDict, total=False):
    data: dict[str, Any]                                  # 业务字段全在这里
    results: Annotated[list[dict], operator.add]          # 每个节点的结果（累加）
```

节点拿到的 ``working`` 就是 ``data`` 的内容，写回的 ``patch`` 合并回 ``data``。
**新增节点不需要改状态声明** —— 从根上消除了那类静默失败。

``results`` 用 ``operator.add`` 做 reducer：每个节点只返回自己那一条，
累加而不是覆盖。没有这个 reducer，后跑的节点会把前面的结果冲掉。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Sequence, TypedDict
import operator

from copilot.contracts import Finding, NodeResult, StageReport, build_report

__all__ = [
    "NodeOutcome",
    "StageNode",
    "StageState",
    "run_stage",
    "build_stage_graph",
    "ok_outcome",
]


@dataclass
class NodeOutcome:
    """一个节点交回来的东西。

    节点函数返回它，运行器负责把它转成 :class:`NodeResult`
    并把 ``patch`` 合并进共享状态。
    """

    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: 写回共享状态的键（供后续节点使用），例如 ``{"outline": ...}``
    patch: dict[str, Any] = field(default_factory=dict)


def ok_outcome(
    *,
    findings: Sequence[Finding] = (),
    stats: dict[str, Any] | None = None,
    notes: Sequence[str] = (),
    patch: dict[str, Any] | None = None,
) -> NodeOutcome:
    """构造成功结果的便捷函数（让节点函数写起来短一点）。"""
    return NodeOutcome(
        findings=list(findings),
        stats=dict(stats or {}),
        notes=list(notes),
        patch=dict(patch or {}),
    )


@dataclass(frozen=True)
class StageNode:
    """阶段内部的一个工作内容。"""

    code: str
    name: str
    #: 节点实现：接共享状态，返回 :class:`NodeOutcome`
    run: Callable[[dict[str, Any]], NodeOutcome]
    #: 可选：返回 ``False`` 时跳过本节点（例如"没有外部报告就不跑解析"）
    when: Callable[[dict[str, Any]], bool] | None = None
    requires_model: bool = False
    description: str = ""

    def should_run(self, state: dict[str, Any]) -> bool:
        return True if self.when is None else bool(self.when(state))


# ─────────────────────────── 图状态 ───────────────────────────


class StageState(TypedDict, total=False):
    """阶段图的共享状态。

    **只声明三个键** —— 业务字段全部收进 ``data``，理由见模块文档。
    """

    data: dict[str, Any]
    """业务字段：节点读它，``patch`` 合并回它。"""

    results: Annotated[list[dict[str, Any]], operator.add]
    """每个节点交回的结果（reducer 累加）。"""


# ─────────────────────────── 建图 ───────────────────────────


def _wrap_node(
    node: StageNode,
    on_node: Callable[[str, NodeResult], None] | None,
) -> Callable[[StageState], dict[str, Any]]:
    """把一个 :class:`StageNode` 包成图节点。

    **失败隔离在这里**：节点抛异常不往外抛，而是转成 ``ok=False`` 的结果 ——
    一个节点挂了不该让整张图失败，但会让报告判为不可通过。
    """

    def _run(state: StageState) -> dict[str, Any]:
        working = dict(state.get("data") or {})

        # ① 条件跳过
        if not node.should_run(working):
            result = NodeResult(
                node=node.code, ok=True, notes=[f"按条件跳过（{node.name}）"]
            )
            if on_node:
                on_node(node.code, result)
            return {"results": [result.to_dict()]}

        # ② 执行（异常 → 失败结果，不中断图）
        try:
            outcome = node.run(working)
            if not isinstance(outcome, NodeOutcome):
                raise TypeError(
                    f"节点 {node.code} 必须返回 NodeOutcome，"
                    f"收到 {type(outcome).__name__}"
                )
        except Exception as exc:  # noqa: BLE001 - 单节点失败不带崩整个阶段
            result = NodeResult(
                node=node.code, ok=False, error=f"{type(exc).__name__}: {exc}"
            )
            if on_node:
                on_node(node.code, result)
            return {"results": [result.to_dict()]}

        # ③ 产物合并回 data
        merged = {**working, **outcome.patch}
        result = NodeResult(
            node=node.code,
            ok=True,
            findings=outcome.findings,
            stats=outcome.stats,
            notes=outcome.notes,
        )
        if on_node:
            on_node(node.code, result)
        return {"data": merged, "results": [result.to_dict()]}

    return _run


def build_stage_graph(
    nodes: Sequence[StageNode],
    *,
    on_node: Callable[[str, NodeResult], None] | None = None,
):
    """把节点编成一张状态图（线性链）。

    单独暴露出来，是为了让文档/演示能直接 ``draw_mermaid()`` 看结构，
    也为将来某个阶段要加分支时留好入口。
    """
    from langgraph.graph import END, START, StateGraph

    if not nodes:
        raise ValueError("至少要有一个节点才能建图")

    builder = StateGraph(StageState)
    for node in nodes:
        builder.add_node(node.code, _wrap_node(node, on_node))

    builder.add_edge(START, nodes[0].code)
    for prev, nxt in zip(nodes, nodes[1:], strict=False):
        builder.add_edge(prev.code, nxt.code)
    builder.add_edge(nodes[-1].code, END)
    return builder


def run_stage(
    *,
    stage: str,
    label: str,
    nodes: Sequence[StageNode],
    state: dict[str, Any],
    on_node: Callable[[str, NodeResult], None] | None = None,
    only: Sequence[str] | None = None,
    checkpointer: Any = None,
    thread_id: str | None = None,
) -> StageReport:
    """**把一个阶段的节点编成 LangGraph 状态图并跑完**，返回合并报告。

    ``state`` 是初始业务字段，节点之间用 ``NodeOutcome.patch`` 传递 ——
    **一个阶段内部共享状态，阶段之间不共享**。

    ``on_node`` 可选回调，用于演示脚本实时打印进度。

    ``only`` 只跑这些节点（按 code 匹配，``None`` = 全部）。
    **这是主 Agent 做细粒度路由的入口**：它在**建图前**裁剪节点，
    所以没选中的节点根本不会进图。

    ★ 被排除 ≠ 跳过：``when`` 跳过的节点会在报告里留一条「按条件跳过」，
    而 ``only`` 排除的节点**完全不出现** —— 它们不是这次任务的一部分。

    ``checkpointer`` / ``thread_id``：传了就支持中断恢复（默认不用）。
    """
    if only is not None:
        wanted = {str(c) for c in only}
        unknown = wanted - {n.code for n in nodes}
        if unknown:
            raise KeyError(f"阶段 {stage} 没有这些节点：{sorted(unknown)}")
        nodes = tuple(n for n in nodes if n.code in wanted)

    if not nodes:
        return build_report(label, [], stage=stage)

    app = build_stage_graph(nodes, on_node=on_node).compile(checkpointer=checkpointer)

    init: StageState = {"data": dict(state), "results": []}
    config: dict[str, Any] | None = None
    if checkpointer is not None:
        tid = thread_id or uuid.uuid4().hex[:12]
        config = {"configurable": {"thread_id": tid}}

    final = app.invoke(init, config) if config else app.invoke(init)

    results = [NodeResult.from_dict(r) for r in (final.get("results") or [])]
    return build_report(label, results, stage=stage)
