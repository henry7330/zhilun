"""D 阶段的数据模型：大纲、图表分析结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "OutlineSection",
    "Outline",
    "CaptionFact",
    "ReferenceFact",
    "FigureFix",
    "FigureReport",
]


@dataclass
class OutlineSection:
    """大纲里的一节。"""

    title: str
    level: int = 1
    #: 这一节要写哪几个要点（来自 ResearchDesign 的分配）
    points: list[str] = field(default_factory=list)
    #: 目标字数（按模板下限分配）
    target_words: int = 0
    children: list["OutlineSection"] = field(default_factory=list)

    @property
    def total_words(self) -> int:
        return self.target_words + sum(c.total_words for c in self.children)

    def flat(self) -> list[tuple[int, "OutlineSection"]]:
        out: list[tuple[int, OutlineSection]] = [(self.level, self)]
        for c in self.children:
            out.extend(c.flat())
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "level": self.level,
            "points": list(self.points),
            "target_words": self.target_words,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Outline:
    """一份大纲。"""

    title: str
    sections: list[OutlineSection] = field(default_factory=list)
    skeleton: bool = False
    notes: list[str] = field(default_factory=list)

    def flat(self) -> list[tuple[int, OutlineSection]]:
        out: list[tuple[int, OutlineSection]] = []
        for s in self.sections:
            out.extend(s.flat())
        return out

    @property
    def total_words(self) -> int:
        return sum(s.total_words for s in self.sections)

    def validate(self) -> list[str]:
        """返回大纲的问题清单（空 = 合格）。"""
        problems: list[str] = []
        if not self.sections:
            problems.append("大纲为空")
        for level, s in self.flat():
            if not s.points and level >= 2:
                problems.append(f"章节「{s.title}」没有分配要点")
        if self.total_words <= 0:
            problems.append("没有字数分配")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "skeleton": self.skeleton,
            "notes": list(self.notes),
            "total_words": self.total_words,
            "problems": self.validate(),
            "sections": [s.to_dict() for s in self.sections],
        }

    def to_markdown(self) -> str:
        lines = [f"# 大纲：{self.title}", ""]
        for level, s in self.flat():
            indent = "  " * max(0, level - 1)
            words = f"（目标 {s.target_words} 字）" if s.target_words else ""
            lines.append(f"{indent}- {'#' * level} {s.title}{words}")
            for p in s.points:
                lines.append(f"{indent}  - {p}")
        if self.notes:
            lines += ["", "> " + "；".join(self.notes)]
        lines.append("")
        return "\n".join(lines)


# ─────────────────────────── 图表 ───────────────────────────


@dataclass(frozen=True)
class CaptionFact:
    """一个题注。

    ``number`` 是**解析出来的整数**（``"2-1"`` → 2），``raw_number`` 保留原文。
    为什么两者都留：中文论文常用**按章编号**（图 2-1 / 图 3-1），
    这种编号**不应该**被"重排成 1,2,3" —— 需要看原文才能判断。
    """

    kind: str          # "figure" | "table"
    line: int
    number: int        # 解析出的首个整数（解析不出来则为 -1）
    text: str
    raw_number: str = ""

    @property
    def chapter_based(self) -> bool:
        """是否按章编号（形如 ``2-1``）—— 这种编号不参与顺序重排。"""
        return "-" in self.raw_number or "." in self.raw_number

    @property
    def label(self) -> str:
        return "图" if self.kind == "figure" else "表"


@dataclass(frozen=True)
class ReferenceFact:
    """正文里的一处图表引用。"""

    kind: str          # "figure" | "table"
    line: int
    number: int
    raw: str

    @property
    def label(self) -> str:
        return "图" if self.kind == "figure" else "表"


@dataclass(frozen=True)
class FigureFix:
    """一条**具体的**修复建议（含行号与新旧值）。

    D3 只给建议，**不自动改**。理由见 ``figures.py`` 的模块注释。
    """

    line: int
    what: str
    old: str
    new: str
    why: str

    def describe(self) -> str:
        return f"第 {self.line + 1} 行：{self.what}「{self.old}」→「{self.new}」（{self.why}）"

    def to_dict(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "what": self.what,
            "old": self.old,
            "new": self.new,
            "why": self.why,
        }


@dataclass
class FigureReport:
    """图表现状 + 修复建议。"""

    captions: list[CaptionFact] = field(default_factory=list)
    references: list[ReferenceFact] = field(default_factory=list)
    fixes: list[FigureFix] = field(default_factory=list)
    unreferenced: list[CaptionFact] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "captions": [
                {"kind": c.kind, "line": c.line, "number": c.number, "text": c.text}
                for c in self.captions
            ],
            "references": [
                {"kind": r.kind, "line": r.line, "number": r.number, "raw": r.raw}
                for r in self.references
            ],
            "fixes": [f.to_dict() for f in self.fixes],
            "unreferenced": [c.text for c in self.unreferenced],
        }
