"""A 阶段的数据模型：选题信号、可行性、开题报告。

## `Gap` 为什么必须带 `support`

一个没有任何语料支撑的"研究空白"是**猜测**。把它写进开题报告，
整篇论文的出发点就是错的。

所以 ``Gap.support`` 是必填的：它记录"这个空白是从哪几篇文献推出来的"。
``strength`` 再区分是**原文明确说了**（``explicit``）还是**我统计推断的**（``inferred``）——
两者可信度差一个数量级，不能混着写。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "KeywordStat",
    "Gap",
    "TopicSignals",
    "Feasibility",
    "ProposalSection",
    "Proposal",
]


@dataclass(frozen=True)
class KeywordStat:
    """一个关键词的统计。"""

    term: str
    count: int
    #: 出现在几个年份里（有年份数据时才有意义）
    years: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"term": self.term, "count": self.count, "years": list(self.years)}


@dataclass(frozen=True)
class Gap:
    """一个**候选**研究空白。"""

    text: str
    #: 支撑它的语料标识（**必须有**；空的不该被输出）
    support: tuple[str, ...]
    strength: Literal["explicit", "inferred"] = "inferred"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "support": list(self.support),
            "strength": self.strength,
        }


@dataclass
class TopicSignals:
    """A1 的产出。**全部是统计量，可复现。**"""

    keywords: list[KeywordStat] = field(default_factory=list)
    #: 高频共现对（说明是成熟子领域）
    cooccurrence: list[tuple[str, str, int]] = field(default_factory=list)
    #: 逐年关键词计数（有年份数据时才有）
    trends: dict[str, dict[int, int]] = field(default_factory=dict)
    gaps: list[Gap] = field(default_factory=list)
    corpus_size: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_size": self.corpus_size,
            "keywords": [k.to_dict() for k in self.keywords],
            "cooccurrence": [
                {"a": a, "b": b, "count": c} for a, b, c in self.cooccurrence
            ],
            "trends": {k: {str(y): n for y, n in v.items()} for k, v in self.trends.items()},
            "gaps": [g.to_dict() for g in self.gaps],
            "notes": list(self.notes),
        }

    def to_markdown(self) -> str:
        lines = ["# 选题信号", "", f"- 语料：{self.corpus_size} 篇", ""]
        if self.keywords:
            lines += ["## 高频关键词", "", "| 关键词 | 出现次数 | 涉及年份 |", "|---|---|---|"]
            for k in self.keywords:
                years = "、".join(str(y) for y in k.years) or "—"
                lines.append(f"| {k.term} | {k.count} | {years} |")
            lines.append("")
        if self.cooccurrence:
            lines += ["## 高频共现（成熟子领域的信号）", "", "| 词 A | 词 B | 共现篇数 |", "|---|---|---|"]
            for a, b, c in self.cooccurrence[:10]:
                lines.append(f"| {a} | {b} | {c} |")
            lines.append("")
        if self.gaps:
            lines += [
                "## 候选研究空白",
                "",
                "> ⚠️ **这些是候选，不是结论。** 全部标记为 `inferred`（统计推断）。",
                "",
                "| 空白 | 支撑文献数 | 强度 |",
                "|---|---|---|",
            ]
            for g in self.gaps:
                lines.append(f"| {g.text} | {len(g.support)} | `{g.strength}` |")
            lines.append("")
        if self.notes:
            lines += ["## 说明", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)


@dataclass
class Feasibility:
    """A2 的产出。"""

    level: str = "unknown"
    """粗略量级：``low`` / ``medium`` / ``high`` / ``unknown``。"""

    blockers: list[str] = field(default_factory=list)
    """**阻断性**风险 —— 这一项为空时"可行性"才有意义。"""

    risks: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    skeleton: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "blockers": list(self.blockers),
            "risks": list(self.risks),
            "suggestions": list(self.suggestions),
            "skeleton": self.skeleton,
        }


@dataclass
class ProposalSection:
    title: str
    points: list[str] = field(default_factory=list)
    body: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "points": list(self.points), "body": self.body}


@dataclass
class Proposal:
    """A3 的产出。"""

    title: str = ""
    sections: list[ProposalSection] = field(default_factory=list)
    skeleton: bool = True
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# 开题报告：{self.title}", ""]
        for s in self.sections:
            lines.append(f"## {s.title}")
            lines.append("")
            if s.body:
                lines.append(s.body)
            else:
                for p in s.points:
                    lines.append(f"- {p}")
            lines.append("")
        if self.notes:
            lines += ["> " + "；".join(self.notes), ""]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "skeleton": self.skeleton,
            "notes": list(self.notes),
            "sections": [s.to_dict() for s in self.sections],
        }
