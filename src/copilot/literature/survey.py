"""B2 综述写作：先分组，再写。

## 为什么综述必须有"分组"这一步

不分组写出来的东西是**流水账**：「A 做了什么，B 做了什么，C 做了什么」。
综述的价值在于**组织** —— 把文献按某个维度摆好，读者才能看出"这个领域
分成几派、各派的共同点和分歧在哪"。

所以 B2 的第一步不是"写"，而是"分组"。分组是确定性的（按年份 / 来源），
写作才需要模型。

## 为什么冲突要并列而不是仲裁

不同文献结论矛盾时，模型很容易"综合"出一个折中说法 ——
但那**抹掉了领域内最该被讨论的东西**。

真实的分歧是综述里最有价值的部分：它告诉读者"这里还没有定论"，
也告诉你"这里可能是个机会"。所以骨架里专门留了「分歧点」一栏。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from copilot.bib import BibEntry

__all__ = ["SurveyGroup", "SurveyPlan", "plan_survey"]


@dataclass
class SurveyGroup:
    """综述里的一个小节（一组文献）。"""

    title: str
    entries: list[BibEntry] = field(default_factory=list)
    #: 写作要点（骨架里给出方向，不代写）
    points: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "count": len(self.entries),
            "entries": [(e.title or e.key) for e in self.entries],
            "points": list(self.points),
        }


@dataclass
class SurveyPlan:
    """综述的写作计划（B2 的骨架产出）。"""

    groups: list[SurveyGroup] = field(default_factory=list)
    #: 需要在正文里并列呈现的分歧点
    conflicts: list[str] = field(default_factory=list)
    #: **去重后的文献篇数**。
    #:
    #: 为什么单独存而不是 ``sum(len(g.entries))``：
    #: 分组是**多维**的（来源一组、年代一组），同一篇会出现在多个组里。
    #: 把各组相加会重复计数 —— 实测 4 篇文献被算成 8 篇。
    papers: int = 0
    skeleton: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        """文献篇数（去重）。"""
        return self.papers or len({(e.title or e.key) for g in self.groups for e in g.entries})

    def to_markdown(self) -> str:
        lines = ["# 综述写作计划", ""]
        lines.append(f"共 {self.total} 篇文献，分 {len(self.groups)} 组。")
        lines.append("")
        for g in self.groups:
            lines.append(f"## {g.title}（{len(g.entries)} 篇）")
            for p in g.points:
                lines.append(f"- {p}")
            if g.entries:
                titles = "；".join((e.title or e.key) for e in g.entries[:6])
                more = "…" if len(g.entries) > 6 else ""
                lines.append(f"- 涉及文献：{titles}{more}")
            lines.append("")
        if self.conflicts:
            lines += ["## 需要并列呈现的分歧", ""]
            lines += [f"- {c}" for c in self.conflicts]
            lines.append("")
        if self.notes:
            lines += ["## 说明", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)


def _decade(year: int | None) -> str:
    if not year:
        return "年份未知"
    start = (year // 5) * 5
    return f"{start}–{start + 4}"


def plan_survey(entries: Sequence[BibEntry]) -> SurveyPlan:
    """按「来源 / 年代」分组，给出写作计划。

    **分组维度选年份 + 来源**，因为这两个字段在参考文献里几乎总是有的；
    按"方法流派"分组更有洞察力，但需要读全文 —— 那是模型/人的工作，
    不在这里假装能做。
    """
    if not entries:
        return SurveyPlan(
            skeleton=True,
            notes=["没有文献，无法分组 —— 先跑 B1 检索或提供参考文献"],
        )

    by_venue: dict[str, list[BibEntry]] = {}
    by_decade: dict[str, list[BibEntry]] = {}
    for e in entries:
        venue = (e.venue or "来源未知").strip()
        by_venue.setdefault(venue, []).append(e)
        by_decade.setdefault(_decade(e.year), []).append(e)

    groups: list[SurveyGroup] = []

    # ① 按来源（>=2 篇才独立成组，否则并进"其他"）
    major = {v: es for v, es in by_venue.items() if len(es) >= 2}
    minor = [e for v, es in by_venue.items() if len(es) < 2 for e in es]

    for venue, es in sorted(major.items(), key=lambda kv: -len(kv[1])):
        groups.append(
            SurveyGroup(
                title=f"来自《{venue}》的工作",
                entries=es,
                points=[
                    "这些工作在方法上有共同点吗？如果有，是什么",
                    "它们的结论方向一致吗？有没有互相矛盾的",
                ],
            )
        )

    if minor:
        groups.append(
            SurveyGroup(
                title="其他来源",
                entries=minor,
                points=["逐篇说明其独特之处（来源分散，往往代表不同的研究传统）"],
            )
        )

    # ② 按年代（>=2 个不同年代才有"演进"可讲）
    if len(by_decade) >= 2:
        years = sorted(by_decade)
        groups.append(
            SurveyGroup(
                title="时间脉络",
                entries=[e for d in years for e in by_decade[d]],
                points=[
                    "按 " + " → ".join(years) + " 的顺序说明研究重点怎么迁移",
                    "近期的文献在解决早期文献留下的什么问题",
                ],
            )
        )

    # ③ 显式标注分歧（骨架给方向，内容留给人/模型）
    conflicts = [
        "同一问题下结论相反的文献需要**并列呈现**，不要综合成折中说法",
        "同一概念用了不同术语的文献需要说明它们的关系（这正是 E6 的 REV001 检查项）",
    ]

    return SurveyPlan(
        groups=groups,
        conflicts=conflicts,
        papers=len(entries),
        skeleton=True,
        notes_placeholder=None,
    ) if False else SurveyPlan(
        groups=groups,
        conflicts=conflicts,
        papers=len(entries),
        skeleton=True,
        notes=[
            "分组维度是「来源 + 年代」—— 参考文献里可靠拿到的字段",
            "按方法流派分组需要读全文，不在本节点假装能做",
            "本节点只给写作骨架，正文需要模型或人工撰写",
        ],
    )
