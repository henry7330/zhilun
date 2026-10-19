"""E4 · 改写单元切分 —— 把定位到的片段拆成可独立改写的句子。

为什么不整段丢给模型改
----------------------
报告给的片段可能跨好几句。整段改写有三个问题：

1. **保真风险大**：一次改一大段，很容易在某一处悄悄改变原意；
2. **无法定位失败点**：改完发现某句不对，不知道是哪句的问题；
3. **无法部分采用**：用户可能只想改其中一句。

切成句子后，逐句改、逐句验、逐句给效果 —— 可控得多。

受保护片段在这里就被标出来
--------------------------
引用、引号内直接引用、带参考文献标注的句子 —— 这些**从改写范围里排除**。
判定复用 ``protected.py``（与 E2/E3 共用同一份逻辑）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from copilot.quality.report_parse.models import LocatedItem
from copilot.quality.rewrite.protected import citation_status, quoted_ranges
from copilot.textnorm import line_of, line_starts
from copilot.textseg import iter_segments

__all__ = ["RewriteUnit", "split_units", "DEFAULT_MIN_UNIT_CHARS"]

#: 低于此长度的句子不单独改写（太短的句子改写成"换同义词"，反而更糟）。
#: 这类句子会并入相邻句一起改。
DEFAULT_MIN_UNIT_CHARS = 8


@dataclass
class RewriteUnit:
    """一个可独立改写的句子。"""

    unit_id: str
    text: str
    char_start: int
    char_end: int
    line: int
    #: 是否受保护（引用 / 引号内 / 带引用标注）
    protected: bool = False
    protected_reason: str = ""
    #: 是否需要跳过（不改），以及原因
    skip_reason: str = ""
    #: 选定的改写策略名（由 strategy.py 填）
    strategy: str = ""

    @property
    def writable(self) -> bool:
        return not self.protected and not self.skip_reason

    @property
    def length(self) -> int:
        return self.char_end - self.char_start

    @property
    def location(self) -> str:
        return f"第 {self.line + 1} 行"


def _merge_short(
    spans: Sequence[tuple[int, int, str]], *, min_chars: int
) -> list[tuple[int, int]]:
    """把过短的句子与下一句合并。

    为什么不单独改短句：一个 5 字的句子（"因此。"）没有改写空间，
    强行改只会变成同义词替换 —— 那是最典型的**伪改写**。
    """
    out: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = 0
    for start, end, _text in spans:
        if cur_start is None:
            cur_start, cur_end = start, end
            continue
        if (cur_end - cur_start) < min_chars:
            cur_end = end
        else:
            out.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    if cur_start is not None:
        out.append((cur_start, cur_end))
    return out


def split_units(
    item: LocatedItem,
    doc_text: str,
    *,
    min_chars: int = DEFAULT_MIN_UNIT_CHARS,
) -> list[RewriteUnit]:
    """把一个已定位的问题条目拆成若干改写单元。

    若条目**未定位**，返回空列表 —— 不知道改哪里就不能瞎改。
    """
    if not item.located or item.char_start is None or item.char_end is None:
        return []

    segment = doc_text[item.char_start : item.char_end]
    if not segment.strip():
        return []

    starts = line_starts(doc_text)
    quoted = quoted_ranges(doc_text)

    spans = [
        (item.char_start + s, item.char_start + e, t)
        for s, e, t in iter_segments(segment)
    ]
    if not spans:
        return []

    merged = _merge_short(spans, min_chars=min_chars)

    units: list[RewriteUnit] = []
    for i, (start, end) in enumerate(merged, start=1):
        text = doc_text[start:end]
        if not text.strip():
            continue
        protected, reason = citation_status(doc_text, start, end, quoted=quoted)
        units.append(
            RewriteUnit(
                unit_id=f"{item.item.item_id}-u{i}",
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
