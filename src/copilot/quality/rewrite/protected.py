"""E4 · 受保护片段识别 —— 降重的第一原则。

降重的第一原则：**不能改的东西绝不能改。**
引用、引号内的直接引用、带参考文献标注的句子 —— 动了它们，
就从"重复率高"变成了**学术不端**，性质严重得多。

本文件的前身是 E3 查重的「引用感知过滤」（把规范引用从重复里豁免掉）。
**逻辑完全相同，用途反过来用**：从"不报为重复"变成"不纳入改写范围"。
复用而不是重写，是因为"哪句话属于规范引用"的判定标准必须只有一份 ——
两份实现迟早会不一致，而那种不一致极难发现。

两个判据（满足其一即视为受保护）
--------------------------------
1. **所在句含引用标注**：``[1]`` / ``[1-3]`` / ``[1,2]`` / ``（张三，2020）``
2. **内容位于引号内**：直接引用

**为什么用「所在句」而不是「所在段」**：一段话里可能既有引用别人的地方，
也有自己的原创表述。按段判定会把整段都保护起来，**漏掉真正该改的部分**。
按句判定才有足够的分辨率。
"""

from __future__ import annotations

import re
from functools import lru_cache

__all__ = [
    "CITATION_MARK",
    "has_citation_mark",
    "quoted_ranges",
    "enclosing_sentence",
    "enclosing_sentence_bounds",
    "citation_status",
]

#: 引用标注。要求含数字，因此不会误匹配参考文献里的类型标识 ``[J]``。
CITATION_MARK = re.compile(
    r"\[\s*\d+(?:\s*[-–—,，]\s*\d+)*\s*\]"          # [1] [1-3] [1,2]
    r"|[（(][^）)]{0,80}?(?:19|20)\d{2}[^）)]{0,80}?[）)]"   # （张三，2020）
    r"|[（(][^）)]{0,80}?et\s+al[^）)]{0,80}?[）)]"          # (Smith et al.)
)

#: 句边界，与 ``copilot.textseg.SEGMENT_BOUNDARY`` 保持一致的语义
_SENTENCE_BOUNDARY = re.compile(r"[。！？；!?;\n]")

#: 成对的引号（中文直角引号与弯引号）。
#: 刻意**不含** ``《》`` —— 那是书名号，标记的是作品名，
#: 把书名号内的文字一律豁免会放过"整段抄一本书"的情况。
_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ("\u201c", "\u201d"),   # “ ”
    ("\u300c", "\u300d"),   # 「 」
    ("\u300e", "\u300f"),   # 『 』
)

#: 直引号：无法区分开合，按出现顺序两两配对
_STRAIGHT_QUOTE = '"'


def has_citation_mark(text: str) -> bool:
    return CITATION_MARK.search(text) is not None


@lru_cache(maxsize=256)
def quoted_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """找出所有被引号包裹的区间 ``(start, end)``。

    这里跑的是**启发式配对**，不是真正的语法解析 —— 引号嵌套、
    未闭合引号都可能配错。因此结果只用于"豁免"这种偏保守的用途：
    配错最多是少豁免一处，不会制造假的重复告警。

    带 ``lru_cache``：同一个文档会被多处查询，重复扫描是浪费。
    """
    ranges: list[tuple[int, int]] = []

    for opener, closer in _QUOTE_PAIRS:
        stack: list[int] = []
        for i, ch in enumerate(text):
            if ch == opener:
                stack.append(i)
            elif ch == closer and stack:
                ranges.append((stack.pop(), i + 1))

    positions = [i for i, ch in enumerate(text) if ch == _STRAIGHT_QUOTE]
    for a, b in zip(positions[0::2], positions[1::2], strict=False):
        ranges.append((a, b + 1))

    return tuple(ranges)


def enclosing_sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """取出 ``[start, end)`` 所在整句的 ``(left, right)`` 偏移。

    向前找最近一个句末标点之后、向后找最近一个句末标点之前。
    """
    left = 0
    for m in _SENTENCE_BOUNDARY.finditer(text, 0, start):
        left = m.end()
    right = len(text)
    for m in _SENTENCE_BOUNDARY.finditer(text, end):
        right = m.start()
        break
    return left, right


def enclosing_sentence(text: str, start: int, end: int) -> str:
    """取出 ``[start, end)`` 所在的整句文本。"""
    left, right = enclosing_sentence_bounds(text, start, end)
    return text[left:right]


#: 句末标点之后还要往后看多少个字符，用于捕捉写在句号之后的引用标注。
#:
#: 中文论文的引用标注**通常**写在句号之前（``……方法[1]。``），
#: 但 ``……方法。[1]`` 这种写法也存在。只看"所在句"会漏掉后者，
#: 于是规范引用被误判成重复。12 个字符足够容纳 ``[1]``、``[1-3]``
#: 这类标注，又不足以把下一句的无关内容卷进来。
_CITATION_TAIL_WINDOW = 12


def citation_status(
    thesis_text: str,
    char_start: int,
    char_end: int,
    *,
    quoted: tuple[tuple[int, int], ...] | None = None,
) -> tuple[bool, str]:
    """判断这处重复是否属于**规范引用**。

    返回 ``(是否正当引用, 理由)``。理由会写进报告，
    这样用户能自己核对"系统为什么放过了这一段"——
    **可解释的豁免**比静默豁免可信得多。

    检查顺序：先引号（直接引用），再句子内的引用标注，
    最后是**句末标点之后**的一小段（兼容 ``……方法。[1]`` 的写法）。
    """
    spans = quoted if quoted is not None else quoted_ranges(thesis_text)

    for q_start, q_end in spans:
        if q_start <= char_start and char_end <= q_end:
            return True, "位于引号内（直接引用）"

    left, right = enclosing_sentence_bounds(thesis_text, char_start, char_end)
    window = thesis_text[left : min(len(thesis_text), right + _CITATION_TAIL_WINDOW)]
    m = CITATION_MARK.search(window)
    if m is not None:
        where = "所在句" if m.start() < (right - left) else "紧随其后"
        return True, f"{where}含引用标注 {m.group(0)}"

    return False, ""
