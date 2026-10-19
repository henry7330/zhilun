"""共享的文本切分工具：**偏移保真**是这里唯一重要的事。

为什么必须共享而不是各写一份
----------------------------
偏移计算错一次，整条验证链就断了（`verify_span` 的文本比对会失败），
而且失败得很隐蔽 —— 只有回环测试才能发现。
证据绑定与检索切分都需要"带精确偏移的句子切分"，
所以只应存在一份实现。

铁律：**绝不在 NFKC 归一化后的文本上计算偏移。**
归一化会改变字符串长度（全角→半角、连字展开），偏移一旦错位就对不上原文。
评分阶段可以各自归一化，但偏移必须基于原文。
"""

from __future__ import annotations

import re
from typing import Iterator

__all__ = ["SEGMENT_BOUNDARY", "iter_segments", "trimmed_segment"]

#: 句子边界。中英标点 + 换行。
SEGMENT_BOUNDARY = re.compile(r"[。！？；!?;\n]+")


def trimmed_segment(content: str, start: int, end: int) -> Iterator[tuple[int, int, str]]:
    """裁掉首尾空白，并同步调整偏移，保证 ``end - start == len(text)``。"""
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    if end > start:
        yield start, end, content[start:end]


def iter_segments(content: str) -> Iterator[tuple[int, int, str]]:
    """把文本切成句级片段，产出 ``(char_start, char_end, text)``。

    偏移相对于 ``content`` **原文**，可直接用于构造 ``EvidenceSpan``。

    >>> list(iter_segments("第一句。第二句。"))
    [(0, 3, '第一句'), (4, 7, '第二句')]
    """
    if not content:
        return
    pos = 0
    for m in SEGMENT_BOUNDARY.finditer(content):
        yield from trimmed_segment(content, pos, m.start())
        pos = m.end()
    yield from trimmed_segment(content, pos, len(content))
