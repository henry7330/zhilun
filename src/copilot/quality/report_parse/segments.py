"""E3 · 把用户文档切成可用于定位的单元。

与其他模块的分工
----------------
- **长度保持的规范化**、**行号换算** 已提到 ``copilot.textnorm``（全项目共用）；
- **重叠度量**（最长公共子串等）在 ``copilot.overlap``；
- 本文件只负责**E3 特有**的事：把文档切成"能被报告片段命中"的单元。

为什么定位需要"单元"而不是直接在全文里搜
----------------------------------------
外部报告给出的片段往往**不是**文档里的完整句子：它可能只是半句、
可能是跨句的一段、也可能因为用户后来又改过而**对不上**。

直接在全文做子串搜索有两个问题：
1. 报告片段带格式差异（多余空格、全角半角）就搜不到；
2. 命中多处时无法判断该报哪一处。

因此做法是：**先把文档切成有明确边界的单元（句/段）并记录偏移，
再让报告片段在这个集合里找最匹配的单元** —— 命中不了就如实报"未定位"，
而不是硬指一个位置。
"""

from __future__ import annotations

from dataclasses import dataclass

from copilot.textnorm import line_of, line_starts, normalize
from copilot.textseg import iter_segments

__all__ = [
    "DocUnit",
    "segment_document",
    "DEFAULT_MIN_CHARS",
    "DEFAULT_MAX_CHARS",
]

#: 合并后的目标最小长度：太短的单元不足以承载一个"重复片段"
DEFAULT_MIN_CHARS = 12

#: 单元最大长度：太大则定位粒度太粗（"整章重复"没有指导意义）
DEFAULT_MAX_CHARS = 180


@dataclass(frozen=True)
class DocUnit:
    """用户文档里的一个可定位单元。"""

    unit_id: str
    #: 规范化后的文本 —— **用于与报告片段比对**
    text: str
    #: 原文 —— **用于展示与定位**
    raw_text: str
    char_start: int
    char_end: int
    line: int

    def __post_init__(self) -> None:
        if self.char_end - self.char_start != len(self.raw_text):
            raise ValueError(
                "偏移跨度必须等于原文长度："
                f"{self.char_end - self.char_start} != {len(self.raw_text)}"
            )
        if len(self.text) != len(self.raw_text):
            raise ValueError(
                "规范化必须长度保持："
                f"规范化后 {len(self.text)} != 原文 {len(self.raw_text)}\n"
                "（长度不一致会让『比对用规范化文本、定位用原始偏移』的坐标错位）"
            )


def segment_document(
    text: str,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[DocUnit]:
    """把文档切分并合并成可定位单元。

    先按句切（复用共享的 ``iter_segments``，偏移基于原文），
    再把**过短**的相邻句合并，直到达到 ``min_chars``，且不超过 ``max_chars``。

    **合并的是偏移区间**，最终文本由该区间重新切片得到 ——
    这样规范化与原文切片天然对齐，不会出现"手工拼接导致长度不符"。
    """
    if not text:
        return []

    starts = line_starts(text)
    sentences = list(iter_segments(text))
    if not sentences:
        return []

    spans: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = 0

    for start, end, _raw in sentences:
        if cur_start is None:
            cur_start, cur_end = start, end
            continue
        cur_len = cur_end - cur_start
        merged_len = end - cur_start
        if cur_len < min_chars and merged_len <= max_chars:
            cur_end = end                      # 太短 → 与下一句合并
        else:
            spans.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    if cur_start is not None:
        spans.append((cur_start, cur_end))

    units: list[DocUnit] = []
    for i, (start, end) in enumerate(spans, start=1):
        raw = text[start:end]
        if not raw.strip():
            continue
        units.append(
            DocUnit(
                unit_id=f"u{i}",
                text=normalize(raw),
                raw_text=raw,
                char_start=start,
                char_end=end,
                line=line_of(starts, start),
            )
        )
    return units
