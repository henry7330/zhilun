"""C1 方法设计：给出研究方案（无模型时给**骨架**）。

## 为什么无模型也要能跑

"没有 API key 就跑不起来"会让整条学习链断在第一步。
所以所有需要模型的节点都实现成「**有模型用模型，没模型给骨架**」：

- 骨架 = 结构完整 + 明确标出待填处；
- 用户拿到骨架后自己填，或配上模型重跑。

**骨架不是"假装完成"**：它会被 ``validate()`` 判为不合格，
阶段报告会给出阻断项 —— 用户一眼能看出"这只是骨架"。

## 为什么 C1 优先产出「比选」而不是「一条路线」

只给一条路线等于没做设计 —— "我选了 M"只是偏好。
比选（至少两条 + 为什么没选另一条）才是设计工作本身。
"""

from __future__ import annotations

from typing import Any, Callable

from copilot.research.models import (
    PLACEHOLDER_BASELINE,
    Metric,
    ResearchDesign,
)

__all__ = ["propose_design", "DEFAULT_METRICS", "DesignPlanner"]

#: 未知领域时的默认指标集。
#:
#: 刻意只给最通用的一对，并**要求填 rationale**（validate 会检查）。
#: 堆一堆"看起来很专业"的指标反而有害 —— 选错指标是最常见的方法论硬伤。
DEFAULT_METRICS: tuple[Metric, ...] = (
    Metric(name="效果", better="higher", rationale=""),
    Metric(name="成本", better="lower", rationale=""),
)

#: 规划器签名：问题 → 设计要点
DesignPlanner = Callable[[str], dict[str, Any]]


def propose_design(
    question: str,
    *,
    planner: DesignPlanner | None = None,
    metrics: tuple[Metric, ...] = DEFAULT_METRICS,
) -> ResearchDesign:
    """给出研究设计。

    ``planner`` 为 ``None`` 时返回**骨架**（``skeleton=True``）：结构齐全、
    内容待填，``validate()`` 会如实报问题。
    """
    if planner is None:
        return ResearchDesign(
            question=question,
            method="",
            alternatives=[],
            baselines=[PLACEHOLDER_BASELINE],
            variables={"independent": [], "dependent": [], "controlled": []},
            metrics=list(metrics),
            sample_size={},
            risks=[],
            skeleton=True,
        )

    raw = planner(question) or {}
    return ResearchDesign(
        question=question,
        method=str(raw.get("method", "")).strip(),
        alternatives=[str(x) for x in raw.get("alternatives") or []],
        baselines=[str(x) for x in raw.get("baselines") or []],
        variables={
            k: [str(x) for x in (raw.get("variables") or {}).get(k) or []]
            for k in ("independent", "dependent", "controlled")
        },
        metrics=list(raw.get("metrics") or metrics),
        sample_size=dict(raw.get("sample_size") or {}),
        risks=[str(x) for x in raw.get("risks") or []],
        skeleton=bool(raw.get("skeleton", False)),
    )
