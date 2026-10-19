"""E3 · 统一问题模型 —— 各家报告的"防腐层"。

为什么要有这一层
----------------
知网、维普、PaperPass、GPTZero……每家报告的结构都不一样：字段名不同、
嵌套不同、有的给来源有的不给。如果后面的定位与改写逻辑直接吃各家原始格式，
会变成 N×M 的分支，每加一种格式就要改 N 处。

**统一模型把"会变的东西"（报告格式）挡在外面**，让核心逻辑只面对一种结构。
这与 E1 的文档模型是同一个设计手法。

三个概念的区分
--------------
- :class:`FlaggedItem` —— **报告说了什么**（原文片段、类型、分数、来源）
- :class:`LocatedItem` —— **它在文档的哪里**（位置 + 定位置信度）
- :class:`Checklist` —— 两者的集合，交给下游

**刻意分开而不是合成一个类**：定位可能失败，而"报告说了什么"是确定存在的。
合成一个类就会出现"位置字段是 None 但对象依然存在"的半成品状态，
下游每处都要判空。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

__all__ = [
    "FlagType",
    "LocateMethod",
    "FlaggedItem",
    "LocatedItem",
    "Checklist",
]


class FlagType(str, Enum):
    """报告指出的问题类型。"""

    DUPLICATE = "duplicate"
    """查重报告：与已有文献重复。"""

    AI = "ai"
    """AI 检测报告：疑似 AI 生成。"""

    @property
    def label(self) -> str:
        return {"duplicate": "重复", "ai": "疑似AI"}[self.value]


class LocateMethod(str, Enum):
    """定位方式 —— 决定用户该不该相信这个位置。"""

    EXACT = "exact"
    """规范化后逐字相同。可信。"""

    FUZZY = "fuzzy"
    """模糊匹配（最长公共子串）。**需要用户核对**。"""

    UNLOCATED = "unlocated"
    """没能定位。**不能硬指一个位置**，必须如实报告。"""

    @property
    def label(self) -> str:
        return {"exact": "精确", "fuzzy": "模糊", "unlocated": "未定位"}[self.value]


@dataclass(frozen=True)
class FlaggedItem:
    """报告里的一条问题（尚未定位）。"""

    item_id: str
    #: 报告指出的那段文字
    text: str
    flag_type: FlagType
    #: 报告给的分数（重复率 / AI 概率）。不同工具口径不同，只作参考不作比较
    score: float | None = None
    #: 报告给出的来源文献标题（查重报告常有）
    source_title: str | None = None
    #: 报告给出的**来源原文**（有则降重可以直接度量，见 E4 的说明）
    source_text: str | None = None
    #: 原始记录，便于追溯"这条是从报告哪里来的"
    raw: dict = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("报告片段不能为空")

    @property
    def has_source_text(self) -> bool:
        """报告是否给了来源原文。

        这决定了 E4 降重能否**直接度量**降重效果，
        还是只能拿用户原句做**代理指标**。
        """
        return bool(self.source_text and self.source_text.strip())


@dataclass
class LocatedItem:
    """定位之后的问题条目。"""

    item: FlaggedItem
    method: LocateMethod
    #: 0..1。精确匹配为 1.0；模糊匹配按命中比例；未定位为 0.0
    confidence: float

    char_start: int | None = None
    char_end: int | None = None
    line: int | None = None
    end_line: int | None = None
    #: 文档中该位置的原文（供用户核对）
    doc_text: str | None = None

    #: 未定位时的原因，或模糊匹配时的参考说明
    note: str = ""

    @property
    def located(self) -> bool:
        return self.method is not LocateMethod.UNLOCATED

    @property
    def location(self) -> str:
        if self.char_start is None or self.line is None:
            return "未定位"
        if self.end_line is not None and self.end_line > self.line:
            return f"第 {self.line + 1}–{self.end_line + 1} 行"
        return f"第 {self.line + 1} 行"

    def to_dict(self) -> dict:
        return {
            "item_id": self.item.item_id,
            "flag_type": self.item.flag_type.value,
            "text": self.item.text,
            "score": self.item.score,
            "source_title": self.item.source_title,
            "has_source_text": self.item.has_source_text,
            "method": self.method.value,
            "confidence": round(self.confidence, 4),
            "line": None if self.line is None else self.line + 1,
            "end_line": None if self.end_line is None else self.end_line + 1,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "doc_text": self.doc_text,
            "note": self.note,
        }


@dataclass
class Checklist:
    """一次解析 + 定位的完整结果，交给 E4/E5。"""

    doc_path: str
    doc_text: str
    items: list[LocatedItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 排序：先按位置（用户从上往下改），未定位的排最后
        self.items.sort(
            key=lambda it: (
                it.char_start if it.char_start is not None else 10**9,
                it.char_end if it.char_end is not None else 10**9,
            )
        )

    # ── 分类 ──

    def located(self) -> list[LocatedItem]:
        return [it for it in self.items if it.located]

    def unlocated(self) -> list[LocatedItem]:
        return [it for it in self.items if not it.located]

    def by_type(self, flag_type: FlagType) -> list[LocatedItem]:
        return [it for it in self.items if it.item.flag_type is flag_type]

    def needing_rewrite(self) -> list[LocatedItem]:
        """需要改写的条目：已定位的（未定位的不知道改哪，不能瞎改）。"""
        return self.located()

    # ── 统计 ──

    def counts(self) -> dict[str, int]:
        exact = sum(1 for it in self.items if it.method is LocateMethod.EXACT)
        fuzzy = sum(1 for it in self.items if it.method is LocateMethod.FUZZY)
        return {
            "total": len(self.items),
            "exact": exact,
            "fuzzy": fuzzy,
            "unlocated": len(self.unlocated()),
            "duplicate": len(self.by_type(FlagType.DUPLICATE)),
            "ai": len(self.by_type(FlagType.AI)),
        }

    def summary_line(self) -> str:
        c = self.counts()
        return (
            f"报告共 {c['total']} 处：精确 {c['exact']} / 模糊 {c['fuzzy']} / "
            f"未定位 {c['unlocated']}"
            f"（重复 {c['duplicate']} · 疑似AI {c['ai']}）"
        )

    def by_location(self) -> list[tuple[tuple[int, int] | None, list[LocatedItem]]]:
        """按文档位置归并 —— 同一处的多条报告条目合成一组。

        **为什么下游需要归并**：同一段文字可能同时被判"重复 + 疑似AI"，
        或者被查重报告按不同来源列了多条。若不归并，E4 会对同一段改好几遍。
        """
        groups: dict[tuple[int, int] | None, list[LocatedItem]] = {}
        for it in self.items:
            key = (
                None
                if it.char_start is None or it.char_end is None
                else (it.char_start, it.char_end)
            )
            groups.setdefault(key, []).append(it)
        return list(groups.items())

    # ── 输出 ──

    def to_dict(self) -> dict:
        return {
            "file": self.doc_path,
            "counts": self.counts(),
            "items": [it.to_dict() for it in self.items],
        }

    def to_markdown(self) -> str:
        c = self.counts()
        lines = [
            "# 检测报告解析结果",
            "",
            f"- 文档：`{self.doc_path}`",
            f"- 报告条目：**{c['total']}** 处",
            "",
            f"| 指标 | 数量 |",
            "|---|---:|",
            f"| 精确匹配 | {c['exact']} |",
            f"| 模糊匹配 | {c['fuzzy']} |",
            f"| **未定位** | {c['unlocated']} |",
            f"| 重复类 | {c['duplicate']} |",
            f"| 疑似AI类 | {c['ai']} |",
            "",
        ]

        if c["unlocated"]:
            lines += [
                "## ⚠️ 未能定位的条目",
                "",
                "这些条目在报告里存在，但**没能在文档里找到对应位置** —— ",
                "通常是因为报告生成之后你改过这段文字。",
                "**系统不会替你猜一个位置**，请你自行核对：",
                "",
                "| # | 类型 | 报告片段 | 原因 |",
                "|---|---|---|---|",
            ]
            for i, it in enumerate(self.unlocated(), start=1):
                snippet = " ".join(it.item.text.split())[:60]
                lines.append(
                    f"| {i} | {it.item.flag_type.label} | {snippet} | {it.note or '—'} |"
                )
            lines.append("")

        located = self.located()
        if located:
            lines += [
                "## 已定位的条目",
                "",
                "| 位置 | 方式 | 可信度 | 类型 | 报告给的分数 | 片段 |",
                "|---|---|---:|---|---:|---|",
            ]
            for it in located:
                snippet = " ".join(it.item.text.split())[:50]
                score = "—" if it.item.score is None else f"{it.item.score:.3f}"
                lines.append(
                    f"| {it.location} | {it.method.label} | {it.confidence:.2f} "
                    f"| {it.item.flag_type.label} | {score} | {snippet} |"
                )
            lines.append("")

        lines += [
            "## 说明",
            "",
            "| 定位方式 | 含义 | 该怎么做 |",
            "|---|---|---|",
            "| **精确** | 规范化后逐字相同 | 位置可信，直接改 |",
            "| **模糊** | 靠最长公共子串找到最像的位置 | **先核对再改** —— 文档可能改过 |",
            "| **未定位** | 没找到足够相似的位置 | 系统不猜位置，请自行查找 |",
            "",
        ]
        return "\n".join(lines)
