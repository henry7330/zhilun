"""主 Agent 的规划产物：**一次请求会被拆成什么**。

## 为什么规划要变成「数据」而不是「代码分支」

如果用 `if "降重" in request: ...` 直接串流程，会有三个问题：

1. **看不见**：用户只能看到最后的结果，不知道系统打算做什么；
2. **不可测**：想验证"这句话该不该触发 F 阶段"，只能跑完整流程；
3. **不可改**：想加一个阶段就得改那个巨大的 if。

所以规划先产出**一个数据对象** `Plan`，再由执行器去跑它。
好处是：**规划可以单测**（给一句话 → 断言 Plan 里有哪些 step），
用户也能在动手前先看一眼 `plan` 长什么样（`run_request(..., plan_only=True)`）。

## 三层路由

```
用户请求
   │
   ├─ ① 选阶段   PlanStep.stage      "这句话该调哪几个 Agent"
   ├─ ② 选节点   PlanStep.nodes      "那个阶段里只跑哪几个节点"
   └─ ③ 选模型   PlanStep.model      "那个节点用什么模型"
```

三层都落在 `PlanStep` 上 —— 一个 step 就是一次「用某个模型、跑某阶段的某几个节点」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "PlanStep",
    "Plan",
    "StepResult",
    "OrchestrationReport",
    "REQUIRED_INPUTS",
]

#: 每个阶段**至少要有什么输入**才能跑。
#:
#: 缺了就跳过该 step 并说明原因 —— **不硬跑**。
#: 硬跑的后果是产出一堆"骨架"或空报告，用户会以为检查过了。
REQUIRED_INPUTS: dict[str, tuple[str, ...]] = {
    "A": ("title",),
    "B": (),
    "C": ("question",),
    "D": ("title",),
    "E": ("draft",),
    "F": ("draft", "report"),
}


@dataclass(frozen=True)
class PlanStep:
    """**一次调用**：用某个模型，跑某个阶段的某几个节点。"""

    stage: str
    """阶段键：A–F。"""

    nodes: tuple[str, ...] | None = None
    """只跑这些节点；``None`` = 该阶段的全部节点。"""

    reason: str = ""
    """为什么要调它 —— 写给用户看的一句话，也是规划的可解释性。"""

    model: str | None = None
    """建议使用的模型；``None`` 表示这一步全是确定性节点，不需要模型。"""

    def describe(self) -> str:
        scope = "全部节点" if self.nodes is None else "/".join(self.nodes)
        model = self.model or "（确定性，不用模型）"
        return f"{self.stage} · {scope} · {model}"


@dataclass
class Plan:
    """主 Agent 对一次请求的完整计划。"""

    request: str
    steps: list[PlanStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    """缺哪些输入（缺了会让对应的 step 无法执行）。"""

    @property
    def stages(self) -> list[str]:
        return [s.stage for s in self.steps]

    def describe(self) -> str:
        if not self.steps:
            return "（没有匹配到任何阶段）"
        return " → ".join(self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "stages": self.stages,
            "steps": [
                {
                    "stage": s.stage,
                    "nodes": list(s.nodes) if s.nodes else None,
                    "model": s.model,
                    "reason": s.reason,
                }
                for s in self.steps
            ],
            "notes": list(self.notes),
            "missing": list(self.missing),
        }


@dataclass
class StepResult:
    """一个 step 的执行结果。"""

    step: PlanStep
    report: Any = None
    """`StageReport`；被跳过时为 ``None``。"""

    skipped_reason: str = ""
    """跳过原因；空字符串表示真的执行了。"""

    @property
    def ran(self) -> bool:
        return self.report is not None

    def summary_line(self) -> str:
        if not self.ran:
            return f"⏭ {self.step.stage} 跳过：{self.skipped_reason}"
        return f"{self.step.stage} {self.report.summary_line()}"


@dataclass
class OrchestrationReport:
    """主 Agent 跑完一次请求的总报告。"""

    request: str
    plan: Plan
    steps: list[StepResult] = field(default_factory=list)

    @property
    def ran_steps(self) -> list[StepResult]:
        return [s for s in self.steps if s.ran]

    @property
    def skipped_steps(self) -> list[StepResult]:
        return [s for s in self.steps if not s.ran]

    @property
    def can_submit(self) -> bool:
        """所有**跑过**的阶段都通过，且没有因缺输入被跳过的必做阶段。"""
        if not self.ran_steps:
            return False
        return all(s.report.can_submit for s in self.ran_steps)

    def summary_line(self) -> str:
        ran = len(self.ran_steps)
        skipped = len(self.skipped_steps)
        head = f"计划 {len(self.plan.steps)} 步：执行 {ran} / 跳过 {skipped}"
        if not ran:
            return f"⚠️ {head}（没有任何阶段被执行）"
        mark = "✅ 通过" if self.can_submit else "❌ 有未通过项"
        return f"{mark}（{head}）"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "plan": self.plan.to_dict(),
            "can_submit": self.can_submit,
            "summary": self.summary_line(),
            "steps": [
                {
                    "stage": s.step.stage,
                    "nodes": list(s.step.nodes) if s.step.nodes else None,
                    "model": s.step.model,
                    "ran": s.ran,
                    "skipped_reason": s.skipped_reason,
                    "report": s.report.to_dict() if s.ran else None,
                }
                for s in self.steps
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            "# 主 Agent 执行报告",
            "",
            f"- 请求：`{self.request}`",
            f"- 计划：{self.plan.describe()}",
            f"- 结论：**{self.summary_line()}**",
            "",
            "## 规划明细",
            "",
            "| 阶段 | 节点 | 模型 | 为什么调它 |",
            "|---|---|---|---|",
        ]
        for s in self.plan.steps:
            scope = "全部" if s.nodes is None else "、".join(s.nodes)
            lines.append(
                f"| {s.stage} | {scope} | {s.model or '—'} | {s.reason or '—'} |"
            )
        if self.plan.missing:
            lines += ["", f"> ⚠️ 缺少输入：{'、'.join(self.plan.missing)}"]
        for s in self.steps:
            lines += ["", f"## {s.step.stage} 阶段", ""]
            if s.ran:
                lines.append(s.report.to_markdown())
            else:
                lines.append(f"**跳过**：{s.skipped_reason}")
        return "\n".join(lines)
