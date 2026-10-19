"""E5 · 风格单元切分 —— **段落级**，不是单句。

★ 为什么 E5 不能复用 E4 的单句切分
-----------------------------------
E4 降重按**句**改写是合理的：重复是以句子为单位的，逐句改、逐句验更可控。

但 E5 降 AI **不能这样做** —— 因为它的目标特征本身是**跨句**的：

- **句长变异系数**：一句话没有"方差"可言；
- **套话连接词密度**：需要多句才能看出密度；
- **排比/对仗**：往往跨句成立（"首先…其次…最后"）。

如果按单句切分再送去改写，会出现两个问题：
1. **诊断失效** —— 每个单元只有一句，CV 恒为 0，全都判成"句式过于均匀"；
2. **改写目标错误** —— 让模型去"打散"一个孤立的句子，它只能改同义词。

实测踩过：用单句切分时，每个单元都被诊断出"句式过于均匀"，
但那是**切分方式造成的假象**，不是文本的问题。

因此 E5 用**段落级单元**：至少 ``min_sentences`` 句、不超过 ``max_chars`` 字。
"""

from __future__ import annotations

from copilot.quality.report_parse.models import LocatedItem
from copilot.quality.rewrite.protected import citation_status, quoted_ranges
from copilot.quality.rewrite.units import RewriteUnit
from copilot.textnorm import line_of, line_starts
from copilot.textseg import iter_segments

__all__ = [
    "split_style_units",
    "DEFAULT_MIN_SENTENCES",
    "DEFAULT_MAX_CHARS",
]

#: 一个风格单元至少包含几句。
#: 少于 2 句就谈不上"句式变化"，少于 3 句看不出"连接词密度"。
DEFAULT_MIN_SENTENCES = 2

#: 单元最大长度。太长则诊断粒度粗、改写容易失控。
DEFAULT_MAX_CHARS = 400


def split_style_units(
    item: LocatedItem,
    doc_text: str,
    *,
    min_sentences: int = DEFAULT_MIN_SENTENCES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[RewriteUnit]:
    """把一个已定位的告警条目切成**段落级**风格单元。

    未定位的条目返回空列表 —— 不知道改哪里就不能瞎改。
    """
    if not item.located or item.char_start is None or item.char_end is None:
        return []

    segment = doc_text[item.char_start : item.char_end]
    if not segment.strip():
        return []

    starts = line_starts(doc_text)
    quoted = quoted_ranges(doc_text)

    sentences = [
        (item.char_start + s, item.char_start + e)
        for s, e, t in iter_segments(segment)
        if t.strip()
    ]
    if not sentences:
        return []

    groups: list[tuple[int, int]] = []
    cur_start, cur_end = sentences[0]
    count = 1

    for start, end in sentences[1:]:
        would_be = end - cur_start
        if count >= min_sentences and would_be > max_chars:
            groups.append((cur_start, cur_end))
            cur_start, cur_end, count = start, end, 1
        else:
            cur_end = end
            count += 1
    groups.append((cur_start, cur_end))

    # 最后一段若不足 min_sentences，并入前一段（避免产出"单句单元"）
    if len(groups) >= 2:
        last_start, last_end = groups[-1]
        prev_start, _prev_end = groups[-2]
        last_count = sum(
            1 for s, e in sentences if last_start <= s and e <= last_end
        )
        if last_count < min_sentences:
            groups[-2] = (prev_start, last_end)
            groups.pop()

    units: list[RewriteUnit] = []
    for i, (start, end) in enumerate(groups, start=1):
        text = doc_text[start:end]
        if not text.strip():
            continue
        protected, reason = citation_status(doc_text, start, end, quoted=quoted)
        units.append(
            RewriteUnit(
                unit_id=f"{item.item.item_id}-s{i}",
                text=text,
                char_start=start,
                char_end=end,
                line=line_of(starts, start),
                protected=protected,
                protected_reason=reason,
                skip_reason=(
                    "受保护片段：引用不能被改写（改了就是篡改他人原文）"
                    if protected
                    else ""
                ),
            )
        )
    return units
