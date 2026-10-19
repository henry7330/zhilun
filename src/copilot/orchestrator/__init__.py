"""主 Agent（Master Agent）—— **本系统的入口**。

## 这不是一个固定流程系统

用户输入的是一句**自然语言请求**，不是"第几步"。主 Agent 负责把它拆开：

```
"帮我把这篇论文的格式整理一下，顺便看看有没有重复的段落"
        │
        ├─▶ 拆成两个任务：格式规整 + 降重
        ├─▶ 决定调哪几个子 Agent：E、F
        ├─▶ 决定每个子 Agent 里跑哪几个节点：E1、E3；F1、F2
        └─▶ 决定每个节点用什么模型：E1 用严格型、F2 用便宜稳的
```

**三层路由**（全部落在 `PlanStep` 上）：

| 层 | 决定什么 | 在哪 |
|---|---|---|
| ① | 调哪几个**子 Agent（阶段）** | `plan.STAGE_CUES` |
| ② | 阶段里跑哪几个**节点** | `plan.NODE_CUES` + `NODE_DEPS` |
| ③ | 每个节点用什么**模型** | `router.ModelRouter` |

## 与"六个阶段独立"的关系

六个阶段**依然是独立服务**（可以单独调用），主 Agent 是它们之上的**一层调度**。
这不矛盾：

- 想只做一件事 → 直接调 `run_format(...)`，不经过主 Agent；
- 想描述需求让它自己挑 → 调 `run_request("帮我检查格式")`。

**主 Agent 不改变子 Agent 的行为，只决定调谁、调哪几个节点。**

## 它刻意不做的事

| 不做 | 为什么 |
|---|---|
| 把六个阶段全跑一遍 | 用户没要求的事不该做（成本与风险都由用户承担） |
| 替用户判断"结果好不好" | 它只看结构化报告，不做二次判断 |
| 失败时重试到成功 | 规划与执行失败都要**如实报告**，不是掩盖 |
| 偷偷优化执行顺序 | 计划里写什么就执行什么，否则用户无法核对 |

## 用法

```python
from copilot.orchestrator import run_request

# ① 先看它打算怎么做（不执行、不花钱）
report = run_request("这篇论文的格式不太行，帮我看一下", context={"draft": text}, plan_only=True)
print(report.plan.describe())        # E
print(report.plan.steps[0].nodes)    # ("E1", "E3")

# ② 确认后执行
report = run_request("这篇论文的格式不太行，帮我看一下", context={"draft": text})
print(report.summary_line())

# ③ 一句话触发多个阶段
report = run_request(
    "帮我把格式整理一下，再看看有没有重复的段落",
    context={"draft": text, "report": dup_report},
)
print(report.plan.describe())        # E → F
```

缺少输入时**不会硬跑**，而是跳过并说明：

```
E ✅ 通过（3 个节点 / 错误 0 ...）
F ⏭ 跳过：缺少输入：外部查重 / AI 检测报告
```
"""

from __future__ import annotations


from copilot.orchestrator.execute import (
    INPUT_LABEL,
    STAGE_RUNNERS,
    attach_models,
    build_plan,
    execute_step,
    run_request,
)
from copilot.orchestrator.models import (
    REQUIRED_INPUTS,
    OrchestrationReport,
    Plan,
    PlanStep,
    StepResult,
)
from copilot.orchestrator.plan import (
    NODE_DEPS,
    NODE_CUES,
    SOFT_DEPS,
    STAGE_CUES,
    LlmPlanner,
    Planner,
    RulePlanner,
    close_dependencies,
)
from copilot.orchestrator.router import ModelRouter, RouteDecision


# LangGraph 版（``orchestrator.graph``）**延迟导入**：
# 这样 ``import copilot.orchestrator`` 本身不会拉起 langgraph，
# 只有真正调用 ``run_request`` 时才需要它。
_GRAPH_EXPORTS = {
    "run_request_graph",
    "build_orchestrator_graph",
    "OrchestratorRun",
    "OrchestratorState",
}


def __getattr__(name: str):
    if name in _GRAPH_EXPORTS:
        from copilot.orchestrator import graph as _graph

        return getattr(_graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "run_request",
    "Planner",
    "RulePlanner",
    "LlmPlanner",
    "ModelRouter",
    "RouteDecision",
    "Plan",
    "PlanStep",
    "StepResult",
    "OrchestrationReport",
    "STAGE_CUES",
    "NODE_CUES",
    "NODE_DEPS",
    "REQUIRED_INPUTS",
    "INPUT_LABEL",
    "STAGE_RUNNERS",
    "SOFT_DEPS",
    "close_dependencies",
    "build_plan",
    "execute_step",
    "attach_models",
    # LangGraph 版（延迟导入）
    "run_request_graph",
    "build_orchestrator_graph",
    "OrchestratorRun",
    "OrchestratorState",
]
