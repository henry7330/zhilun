"""A 阶段 · Agent「选题与开题」—— 一个可以独立调用的服务。

## 这个 Agent 做什么

把「不知道写什么」变成「一个可执行的研究题目 + 一份开题报告骨架」。

```
A1 选题分析 ──▶ TopicSignals（关键词共现 / 时间趋势 / 空白候选）
A2 可行性评估 ──▶ Feasibility（阻断性风险，不是「可行性中等」这种废话）
A3 开题报告   ──▶ Proposal（章节骨架 + 写作要点）
```

## 怎么用（独立调用，不需要任何前置阶段）

```python
from copilot.topic import run_topic

report = run_topic(
    title="基于图神经网络的推荐方法研究",
    texts=[...],                    # 领域语料（标题 + 摘要）
    years={0: 2019, 1: 2020},       # 可选：算时间趋势
)
print(report.summary_line())
print(report.to_markdown())
```

## 这个阶段的特殊性：它的产出约束后面所有阶段

题目定不下来 → 文献检索没有方向 → 研究方法无从设计。

但也正因为如此，它是最**不该**被自动化掉判断的一环 ——
所以 A1 只给统计信号，A2 只给「该问的问题」，A3 只给骨架。
**系统不替你决定研究什么。**

## 阶段之间零依赖

A 阶段不 import 任何其他阶段的代码，只复用共享层的分词口径
（``copilot.evidence.text.content_tokens``）——
保证「关键词」这个概念在全系统里是同一个定义。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from copilot.contracts import Finding
from copilot.quality.format.report import Severity
from copilot.stage import StageNode, ok_outcome, run_stage

from copilot.topic.agents import (
    FeasibilityJudge,
    ProposalGenerator,
    assess_feasibility,
    generate_proposal,
)
from copilot.topic.models import Feasibility, Proposal, TopicSignals
from copilot.topic.signals import analyze_topic

__all__ = [
    "run_topic",
    "build_topic_nodes",
    "analyze_topic",
    "assess_feasibility",
    "generate_proposal",
    "TopicSignals",
    "Feasibility",
    "Proposal",
]

STAGE_KEY = "A"
CODE_SIGNAL = "A1-SIGNAL"
CODE_FEAS = "A2-FEAS"
CODE_PROPOSAL = "A3-PROPOSAL"


def build_topic_nodes(
    *,
    texts: Sequence[str] = (),
    years: Mapping[int, int] | None = None,
    conditions: Mapping[str, Any] | None = None,
    judge: FeasibilityJudge | None = None,
    generator: ProposalGenerator | None = None,
    top_k: int = 20,
    min_support: int = 2,
) -> tuple[StageNode, ...]:
    """装配三个节点。"""

    # ── A1 选题分析 ──

    def a1_signals(state: dict[str, Any]):
        if not texts:
            return ok_outcome(
                notes=["没有语料 —— **跳过统计**（不等于「这个方向没东西」）"],
                stats={"corpus": 0},
            )

        signals = analyze_topic(texts, top_k=top_k, min_support=min_support, years=years)
        findings: list[Finding] = []

        if not signals.gaps:
            findings.append(
                Finding(
                    node="A1",
                    code=CODE_SIGNAL,
                    severity=Severity.INFO,
                    message="没有找到候选空白（可能是语料太小，或方向已经很拥挤）",
                    fix="扩充语料；或换更具体的子方向",
                )
            )

        for gap in signals.gaps:
            if not gap.support:
                # 理论上 signals 里已过滤，这里再兜一层
                findings.append(
                    Finding(
                        node="A1",
                        code=CODE_SIGNAL,
                        severity=Severity.WARNING,
                        message=f"空白候选没有支撑文献，已丢弃：{gap.text[:40]}",
                        fix="无支撑的空白是猜测，不该出现在报告里",
                    )
                )
                continue
            findings.append(
                Finding(
                    node="A1",
                    code=CODE_SIGNAL,
                    severity=Severity.INFO,
                    message=gap.text,
                    fix="**候选，不是结论** —— 需人工读文献确认是否真空白",
                    payload=gap.to_dict(),
                )
            )

        return ok_outcome(
            findings=findings,
            stats={
                "corpus": signals.corpus_size,
                "keywords": len(signals.keywords),
                "gaps": len(signals.gaps),
            },
            notes=list(signals.notes),
            patch={"signals": signals},
        )

    # ── A2 可行性 ──

    def a2_feasibility(state: dict[str, Any]):
        feas = assess_feasibility(state["title"], conditions=conditions, judge=judge)
        findings: list[Finding] = []

        if feas.skeleton:
            findings.append(
                Finding(
                    node="A2",
                    code=CODE_FEAS,
                    severity=Severity.INFO,
                    message="可行性评估是**骨架**（未接入模型）—— 下面是该自己回答的问题，不是结论",
                    fix="接入 judge，或逐条回答 blockers / risks",
                )
            )
        elif not feas.blockers:
            findings.append(
                Finding(
                    node="A2",
                    code=CODE_FEAS,
                    severity=Severity.WARNING,
                    message="没有识别出任何阻断性风险",
                    fix="**没有阻断性风险通常意味着评估不到位** —— 任何研究都有前置假设",
                )
            )

        for b in feas.blockers:
            findings.append(
                Finding(
                    node="A2",
                    code=CODE_FEAS,
                    severity=Severity.WARNING if feas.skeleton else Severity.ERROR,
                    message=f"阻断性风险：{b}",
                    fix="先解决这一条，否则题目可能做不下去",
                )
            )

        return ok_outcome(
            findings=findings,
            stats={
                "level": feas.level,
                "blockers": len(feas.blockers),
                "skeleton": feas.skeleton,
            },
            notes=[f"可行性量级：{feas.level}"] if not feas.skeleton else ["骨架模式"],
            patch={"feasibility": feas},
        )

    # ── A3 开题报告 ──

    def a3_proposal(state: dict[str, Any]):
        proposal = generate_proposal(
            state["title"], signals=state.get("signals"), generator=generator
        )
        findings: list[Finding] = [
            Finding(
                node="A3",
                code=CODE_PROPOSAL,
                severity=Severity.INFO,
                message=(
                    f"开题报告 {len(proposal.sections)} 节"
                    + ("（骨架，正文待填）" if proposal.skeleton else "")
                ),
                fix="「研究现状」一节必须逐句绑定文献证据 —— 否则容易编造引用",
            )
        ]

        return ok_outcome(
            findings=findings,
            stats={"sections": len(proposal.sections), "skeleton": proposal.skeleton},
            notes=list(proposal.notes),
            patch={"proposal": proposal},
        )

    return (
        StageNode(
            code="A1", name="选题分析", run=a1_signals,
            description="关键词共现 / 趋势 / 空白候选（纯统计）",
        ),
        StageNode(
            code="A2", name="可行性评估", run=a2_feasibility,
            requires_model=judge is not None,
            description="阻断性风险（缺省给「该问的问题」）",
        ),
        StageNode(
            code="A3", name="开题报告", run=a3_proposal,
            requires_model=generator is not None,
            description="章节骨架 + 写作要点（缺省给骨架）",
        ),
    )


def run_topic(
    *,
    title: str,
    texts: Sequence[str] = (),
    years: Mapping[int, int] | None = None,
    conditions: Mapping[str, Any] | None = None,
    judge: FeasibilityJudge | None = None,
    generator: ProposalGenerator | None = None,
    top_k: int = 20,
    min_support: int = 2,
    only: Sequence[str] | None = None,
    on_node: Callable[[str, Any], None] | None = None,
):
    """**A 阶段的对外入口。** 独立调用，没有任何前置依赖。"""
    nodes = build_topic_nodes(
        texts=texts, years=years, conditions=conditions,
        judge=judge, generator=generator, top_k=top_k, min_support=min_support,
    )
    return run_stage(
        stage=STAGE_KEY,
        label=title[:48] or "<empty>",
        nodes=nodes,
        state={"title": title, "texts": list(texts), "years": dict(years or {})},
        only=only,
        on_node=on_node,
    )
