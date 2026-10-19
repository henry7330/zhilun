"""E3 · 片段定位 —— 本模块的核心。

要解决的问题
------------
报告说"这段文字重复了"，到底是用户文档的第几行到第几行？

**真正的难点：报告来自「旧版本」的文档。**
用户在外部工具跑完检测，通常还会继续改稿。等把报告拿进来时，
文档往往已经和报告生成时不一样了：

    报告片段："本文提出了一种基于图神经网络的推荐方法"
    文档现状："本文提出了一种基于异构图神经网络的推荐方法"   ← 中间加了两个字

精确匹配会失败。这时有三种做法，只有一种是对的：

==============================  ============================================
做法                             后果
==============================  ============================================
静默丢弃                         用户以为都处理了，其实漏了一处 —— **最危险**
硬指一个位置                     用户去改错误的地方，白费功夫还改错
**如实报"未定位" + 给参考位置**    ✅ 用户自己判断该不该改
==============================  ============================================

这与 E2 的「**宁可少绑定，不可绑错**」是同一条原则：
**不确定的时候，把不确定性如实交回给人，不要替人做决定。**

三级策略
--------
1. **精确匹配** —— 规范化 + **去空白**后在全文里查找（偏移最准）
2. **模糊匹配** —— 按"可定位单元"找最长公共子串最像的那一段
3. **未定位** —— 都不够，如实报告，附最接近的位置与命中长度供参考

★ 为什么精确匹配要「去空白」
----------------------------
从 PDF 报告里复制文字，几乎必然带进多余空格、不间断空格、换行符。
如果按原样比对，"方法，  并"和"方法，并"就对不上 ——
而这两个在人眼里**是同一句话**。

做法：构造一个"去掉所有空白之后"的索引，并**记录每个字符回到原文的偏移**。
于是去空白后的查找结果能精确映射回原文位置。

这一步很关键：**去空白是为了比对，但偏移必须回到原文** ——
如果直接在去空白文本上算偏移，定位就会错，而且不报错。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from copilot.overlap import longest_common_substring
from copilot.quality.report_parse.models import (
    FlaggedItem,
    LocateMethod,
    LocatedItem,
)
from copilot.quality.report_parse.segments import DocUnit
from copilot.textnorm import line_of, line_starts, normalize

__all__ = [
    "DocIndex",
    "build_index",
    "locate_item",
    "locate_all",
    "DEFAULT_FUZZY_MIN_RATIO",
    "DEFAULT_FUZZY_MIN_RUN",
]

#: 模糊匹配的**最低命中比例**：命中长度 / 报告片段长度。
#: 低于这个比例就判"未定位" —— 宁可说没找到，也不硬指。
DEFAULT_FUZZY_MIN_RATIO = 0.6

#: 模糊匹配的**绝对最短命中长度**（字符）。防止极短片段靠"巧合"定位。
DEFAULT_FUZZY_MIN_RUN = 10


@dataclass(frozen=True)
class DocIndex:
    """文档的比对索引。

    为什么单独做成一个对象：一次解析可能有几十条报告条目，
    如果每条都重新规范化全文、重建偏移映射，复杂度是 O(条目数 × 全文长度)。
    **索引只建一次**，之后每条定位都是 O(查找)。
    """

    text: str
    #: 规范化后的全文（长度与原文一致）
    norm: str
    #: 去掉所有空白后的规范文本 —— **用于查找**
    stripped: str
    #: ``stripped`` 第 i 个字符 → 原文偏移。**用于把查找结果映射回原文**
    pos_map: tuple[int, ...]
    #: 行起点，避免每次重算
    starts: tuple[int, ...]


def _strip_with_map(text: str) -> tuple[str, list[int]]:
    """去掉空白，同时记录每个保留字符在原文中的偏移。"""
    chars: list[str] = []
    pos_map: list[int] = []
    for i, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch)
            pos_map.append(i)
    return "".join(chars), pos_map


def build_index(text: str) -> DocIndex:
    norm = normalize(text)
    stripped, pos_map = _strip_with_map(norm)
    return DocIndex(
        text=text,
        norm=norm,
        stripped=stripped,
        pos_map=tuple(pos_map),
        starts=tuple(line_starts(text)),
    )


def _line_range(starts: Sequence[int], start: int, end: int) -> tuple[int, int]:
    """给出 ``[start, end)`` 覆盖的行区间 ``(起, 止)``。

    ``end`` 是开区间右端，算结束行要用 ``end - 1``，
    否则当右端正好落在下一行行首时会把行号多算一行。
    """
    s = list(starts)
    return (line_of(s, start), line_of(s, max(start, end - 1)))


def _exact_search(haystack_stripped: str, needle: str) -> int:
    """在去空白文本里查找，返回**第一个非空白字符之前**的位置索引。

    返回 -1 表示没找到。
    """
    if not needle:
        return -1
    return haystack_stripped.find(needle)


def locate_item(
    item: FlaggedItem,
    index: DocIndex,
    units: Sequence[DocUnit],
    *,
    fuzzy_min_ratio: float = DEFAULT_FUZZY_MIN_RATIO,
    fuzzy_min_run: int = DEFAULT_FUZZY_MIN_RUN,
) -> LocatedItem:
    """把一条报告条目定位到文档。

    先试精确匹配（去空白后逐字相同，偏移最准），
    不行再退回模糊匹配，最后如实报未定位。
    """
    if not index.text.strip():
        return LocatedItem(
            item=item,
            method=LocateMethod.UNLOCATED,
            confidence=0.0,
            note="文档为空",
        )

    # ── ① 精确匹配：规范化 + 去空白后逐字相同 ──
    needle_stripped, _ = _strip_with_map(normalize(item.text))
    if needle_stripped:
        hit = _exact_search(index.stripped, needle_stripped)
        if hit >= 0:
            start = index.pos_map[hit]
            end = index.pos_map[hit + len(needle_stripped) - 1] + 1
            first, last = _line_range(index.starts, start, end)
            return LocatedItem(
                item=item,
                method=LocateMethod.EXACT,
                confidence=1.0,
                char_start=start,
                char_end=end,
                line=first,
                end_line=last,
                doc_text=index.text[start:end],
                note="",
            )

    # ── ② 模糊匹配：在可定位单元里找最像的一段 ──
    best: tuple[float, DocUnit, int, int, int] | None = None
    #        命中比例, 单元, 单元内起, needle 内起, 命中长度
    for unit in units:
        n_start, u_start, run = longest_common_substring(needle_stripped, unit.text)
        if run < fuzzy_min_run:
            continue
        ratio = run / max(1, len(needle_stripped))
        if best is None or ratio > best[0]:
            best = (ratio, unit, u_start, n_start, run)

    if best is not None and best[0] >= fuzzy_min_ratio:
        ratio, unit, u_start, _n_start, run = best
        start = unit.char_start + u_start
        end = start + run
        first, last = _line_range(index.starts, start, end)
        return LocatedItem(
            item=item,
            method=LocateMethod.FUZZY,
            confidence=round(ratio, 4),
            char_start=start,
            char_end=end,
            line=first,
            end_line=last,
            doc_text=index.text[start:end],
            note=(
                f"模糊匹配：报告片段 {len(needle_stripped)} 字中命中 {run} 字；"
                "文档可能与报告生成时不同，**请核对后再改**"
            ),
        )

    # ── ③ 未定位：如实报告，给出最接近的位置供参考 ──
    if best is not None:
        ratio, unit, _u, _n, run = best
        note = (
            f"最接近的是第 {unit.line + 1} 行，但只命中 {run} 字"
            f"（{ratio:.0%}），低于 {fuzzy_min_ratio:.0%} 的阈值，故不作为定位结果"
        )
    else:
        note = "未能在文档中找到足够相似的位置"

    return LocatedItem(
        item=item,
        method=LocateMethod.UNLOCATED,
        confidence=0.0,
        note=note,
    )


def locate_all(
    items: Sequence[FlaggedItem],
    index: DocIndex,
    units: Sequence[DocUnit],
    *,
    fuzzy_min_ratio: float = DEFAULT_FUZZY_MIN_RATIO,
    fuzzy_min_run: int = DEFAULT_FUZZY_MIN_RUN,
) -> list[LocatedItem]:
    """批量定位。索引与行起点只建一次。"""
    return [
        locate_item(
            item,
            index,
            units,
            fuzzy_min_ratio=fuzzy_min_ratio,
            fuzzy_min_run=fuzzy_min_run,
        )
        for item in items
    ]
