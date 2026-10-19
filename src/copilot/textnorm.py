"""长度保持的文本规范化 —— 全项目共用的文本口径。

为什么这个工具必须在 00-core
-----------------------------
它被**多个模块**用到：

- **E3 报告解析**：把外部查重/AI 报告里的片段定位回用户文档时，
  两边要先做同样的规范化，否则全角/半角、多余空白的差异会让匹配失败；
- **E4 降重**：改写前后要比较重叠度，比较的两侧必须同一口径。

这与「证据层与检索层必须共用分词口径」是同一条理由：
**两侧口径不一致会产生"看起来差一点"的结果，而且不报错，最难排查。**

核心不变量：长度保持
--------------------
``len(normalize(t)) == len(t)``，因此规范化之后的**偏移量依然有效**。

这条不变量极其重要：整条链路的做法是「**匹配用规范化文本、定位用原始偏移**」。
一旦规范化增删了字符（例如做 NFKC 归一化、把连续空格压成一个），
两步的坐标就不再是同一个坐标系 —— **定位会静默错位**：
报告里的行号看着合理，其实指向别处，且不报任何错。

所以这里**只做逐字符映射，绝不增删**：

- 任何空白 → 一个空格（1 换 1）
- 全角字母/数字/标点 → 半角（1 换 1）

这与 ``copilot/textseg.py`` 里「绝不在归一化后的文本上计算偏移」
是同一条铁律的两种表述。
"""

from __future__ import annotations

import bisect

__all__ = ["normalize", "line_starts", "line_of", "fullwidth_map"]


def _build_fullwidth_map() -> dict[str, str]:
    """构造「全角 → 半角」的逐字符映射。只含 **1 换 1** 的映射。"""
    table: dict[str, str] = {}

    # 全角 ASCII 区（０-９ Ａ-Ｚ ａ-ｚ 及全角标点）
    for code in range(0xFF01, 0xFF5F):
        table[chr(code)] = chr(code - 0xFEE0)

    # 全角空格是特例（不在上面的区间里）
    table["\u3000"] = " "

    # 中文标点 → 半角（这些在全角 ASCII 区之外）
    table.update(
        {
            "。": ".",
            "、": ",",
            "，": ",",
            "；": ";",
            "：": ":",
            "！": "!",
            "？": "?",
            "（": "(",
            "）": ")",
            "【": "[",
            "】": "]",
            "《": "<",
            "》": ">",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "—": "-",
            "－": "-",
            "～": "~",
            "·": ".",
        }
    )
    return table


#: 全角 → 半角映射表（模块级缓存，避免每次调用重建）
_FULLWIDTH: dict[str, str] = _build_fullwidth_map()


def fullwidth_map() -> dict[str, str]:
    """返回全角→半角映射表的副本（供测试或外部检查用）。"""
    return dict(_FULLWIDTH)


def normalize(text: str) -> str:
    """**长度保持**的规范化。

    >>> normalize("ａ，ｂ")
    'a,b'
    >>> len(normalize("你好，世界。")) == len("你好，世界。")
    True
    """
    out: list[str] = []
    for ch in text:
        if ch.isspace():
            out.append(" ")                       # 任何空白 → 一个空格
        else:
            out.append(_FULLWIDTH.get(ch, ch))    # 逐字符映射，找不到就原样
    return "".join(out)


def line_starts(text: str) -> list[int]:
    """每一行起始字符的偏移列表，供 :func:`line_of` 做二分查找。"""
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def line_of(starts: list[int], pos: int) -> int:
    """给出字符偏移对应的 **0 基行号**。

    注意空行也算一行 —— 手工数行号极易数错，所以一律用这个函数算。
    """
    return max(0, bisect.bisect_right(starts, pos) - 1)
