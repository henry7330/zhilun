"""第三层路由：**这个节点该用哪个模型**。

## 为什么需要单独一层

同一个阶段里，节点的模型需求差别很大：

| 节点 | 需要模型吗 | 用什么模型 |
|---|---|---|
| A1 选题分析 | ❌ | — （纯统计） |
| A2 可行性评估 | ✅ | 需要"判断力"，用强一点的 |
| E1 格式规整 | ✅ | 需要严格遵循指令，宁可慢 |
| F2 合并改写 | ✅ | 需要稳定输出，且要便宜（一次要改很多段） |

**用一个模型打全场是浪费**：确定性节点根本不该调用模型，
而"改一段话"和"评估一个研究方向"对模型的要求完全不同。

## 模型从哪来

`ModelRouter` 只是一张**登记表**，不自己造模型：

```python
router = ModelRouter(
    default="gpt-5.6-luna",
    by_stage={"F": "cheap-model"},      # 整个 F 阶段用便宜模型
    by_node={"A2": "strong-model"},     # 但 A2 单独用强模型
)
```

`None` 表示"不接模型"——对应节点会走它的确定性/骨架路径。

## 一个 honest 限制

一个阶段一次调用只能接**一个** `formatter` / `rewriter`，
所以同一阶段内的"混合模型"**无法真的分开用**。
本模块遇到混合时会：**选一个统一的，并在 note 里说明** ——
而不是假装按节点分别路由了。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from copilot.contracts import STAGES

__all__ = ["RouteDecision", "ModelRouter"]


@dataclass(frozen=True)
class RouteDecision:
    """一个 step 的模型决策。"""

    model: str | None
    """要用的模型；``None`` = 这一步不接模型。"""

    needs_model: bool
    """这一步里是否有节点需要模型（哪怕没登记模型）。"""

    note: str = ""

    def describe(self) -> str:
        if not self.needs_model:
            return "确定性（不用模型）"
        if self.model is None:
            return "需要模型但**未登记** —— 会走骨架/确定性路径"
        return self.model


@dataclass
class ModelRouter:
    """按阶段 / 节点登记模型。

    优先级：**节点 > 阶段 > 默认**。
    """

    default: str | None = None
    by_stage: Mapping[str, str] = field(default_factory=dict)
    by_node: Mapping[str, str] = field(default_factory=dict)

    def resolve(self, stage: str, nodes: Sequence[str] | None) -> RouteDecision:
        spec = STAGES.get(stage)
        if spec is None:
            return RouteDecision(model=None, needs_model=False, note=f"未知阶段 {stage}")

        selected = list(nodes) if nodes else [n.code for n in spec.nodes]
        needing = [c for c in selected if _requires_model(stage, c)]

        if not needing:
            return RouteDecision(
                model=None, needs_model=False,
                note="所选节点全部是确定性的",
            )

        chosen = {
            c: (self.by_node.get(c) or self.by_stage.get(stage) or self.default)
            for c in needing
        }
        distinct = sorted({m for m in chosen.values() if m})

        if not distinct:
            return RouteDecision(
                model=None, needs_model=True,
                note="有节点需要模型，但没登记任何模型 —— 会走骨架路径",
            )
        if len(distinct) == 1:
            return RouteDecision(model=distinct[0], needs_model=True, note="")

        # 混合模型：一次阶段调用只能接一个，如实说明
        used = distinct[0]
        per_node = "、".join(f"{c}={m}" for c, m in sorted(chosen.items()) if m)
        return RouteDecision(
            model=used,
            needs_model=True,
            note=f"混合模型（{per_node}）；一次阶段调用只能接一个，本次统一用 {used}",
        )


def _requires_model(stage: str, code: str) -> bool:
    try:
        return bool(STAGES[stage].node(code).requires_model)
    except KeyError:
        return False
