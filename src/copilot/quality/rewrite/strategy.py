"""E4 · 改写策略选择 —— 决定"这一段能怎么改"。

为什么必须有这一层
------------------
**不同内容的改写空间完全不同**，用同一种策略会出错：

========================  ==============  ==============================
内容类型                   改写空间         正确做法
========================  ==============  ==============================
方法描述                   中               术语保留，改句式
结论陈述                   大               重述 + 换表述角度
**定义 / 定理**            **几乎没有**     ★ **建议规范引用，而不是改写**
实验数据                   无               不得改动
========================  ==============  ==============================

**★ 最重要的一条：有些内容本来就不该改写，而应该引用。**

如果系统硬要改写一个定义，产出的东西既不像原文、**也可能不准确** ——
那比"重复率高"糟糕得多。所以策略里必须有一条「建议引用而非改写」，
并且要能让用户理解为什么。

这与 E5 的「保真优先于风格」是同一条原则：
**准确性的优先级永远高于"把重复率降下去"。**
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ContentKind",
    "Strategy",
    "RewriteInstruction",
    "classify",
    "choose_instruction",
]


class ContentKind(str, Enum):
    """内容类型 —— 决定改写空间。"""

    DEFINITION = "definition"
    """定义 / 定理 / 概念界定。改写空间几乎没有。"""

    DATA = "data"
    """数据 / 实验结果 / 公式。**不得改动。**"""

    METHOD = "method"
    """方法描述。术语保留，句式可改。"""

    CONCLUSION = "conclusion"
    """结论陈述。可以换表述角度。"""

    OTHER = "other"

    @property
    def label(self) -> str:
        return {
            "definition": "定义/定理",
            "data": "数据/公式",
            "method": "方法描述",
            "conclusion": "结论陈述",
            "other": "其他",
        }[self.value]


class Strategy(str, Enum):
    """改写策略。"""

    REPHRASE = "rephrase"
    """重述表达方式（术语保留）。"""

    RESTRUCTURE = "restructure"
    """换表述角度 / 重构句子结构。"""

    CITE_INSTEAD = "cite_instead"
    """★ 不建议改写，建议改为规范引用。"""

    SKIP = "skip"
    """不改动。"""


@dataclass(frozen=True)
class RewriteInstruction:
    """交给改写器的指令。"""

    strategy: Strategy
    kind: ContentKind
    #: 给模型看的具体要求（人类也能读懂，便于审计）
    hint: str
    #: 为什么选这个策略 —— 会写进报告，让用户理解
    rationale: str


# ── 分类用的模式 ──

_DEFINITION = re.compile(
    r"定义为|是指|指的是|称为|称作|通常指|所谓|"
    r"定理|引理|推论|公理|命题|定律|"
    r"设\s*\w+\s*为|记作|满足以下条件"
)

_CONCLUSION = re.compile(
    r"结果表明|实验表明|实验显示|数据显示|由此可见|综上|"
    r"因此|因而|所以|这说明|证明了|验证了|"
    r"可以得到|可以看出|可见"
)

_METHOD = re.compile(
    r"本文(提出|设计|采用|引入|构建)|我们(提出|设计|采用|引入|构建)|"
    r"通过.{0,12}(实现|完成)|"
    r"分为.{0,8}(阶段|步骤|部分)|流程如下"
)

#: 数字 / 百分号 / 单位 —— 用于识别"数据密集"的句子
_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|个|倍|次|秒|ms|GB|MB)?")


def _digit_ratio(text: str) -> float:
    """数字字符占非空白字符的比例。"""
    compact = re.sub(r"\s", "", text)
    if not compact:
        return 0.0
    return sum(1 for ch in compact if ch.isdigit()) / len(compact)


def classify(text: str, *, data_ratio_threshold: float = 0.18) -> ContentKind:
    """判断内容类型。

    **顺序有意义**：先判定义与数据 —— 这两类一旦判错，后果最严重
    （改写定义会失准、改写数据会造假）。宁可把它们判得保守一些。

    方法描述排在结论之前：``本文提出…并在…验证了效果`` 这类句子同时命中
    两边，但它的**主要信息是方法**，改写时应保留术语、改句式
    （METHOD 的提示），而不是按结论去换表述角度。

    ``data_ratio_threshold``：数字占比超过这个值就当作数据密集。
    0.18 是保守估计 —— 正常的论述句里数字占比通常远低于此。
    这个值刻意**留得较高**：像「结果表明…提升了 8.3%」这样的句子仍按结论处理，
    其中的数字由 E4 的**保真校验**守住，不必整句禁改。
    """
    if _DEFINITION.search(text):
        return ContentKind.DEFINITION

    numbers = _NUMBER.findall(text)
    if len(numbers) >= 2 and _digit_ratio(text) >= data_ratio_threshold:
        return ContentKind.DATA

    if _METHOD.search(text):
        return ContentKind.METHOD
    if _CONCLUSION.search(text):
        return ContentKind.CONCLUSION
    return ContentKind.OTHER


def choose_instruction(kind: ContentKind) -> RewriteInstruction:
    """按内容类型给出改写指令。"""
    if kind is ContentKind.DEFINITION:
        return RewriteInstruction(
            strategy=Strategy.CITE_INSTEAD,
            kind=kind,
            hint="不要改写：定义/定理的表述是本领域的固定说法，改写会失准。",
            rationale=(
                "这段是定义/定理类表述。改写它会让术语与原文不一致、"
                "甚至改变其含义 —— 建议改为**规范引用**（加引号并标注出处），"
                "而不是重述。"
            ),
        )

    if kind is ContentKind.DATA:
        return RewriteInstruction(
            strategy=Strategy.SKIP,
            kind=kind,
            hint="不要改写：数据与公式必须保持原样。",
            rationale=(
                "这段含具体数据/公式。**数字不能改** —— 改了就是伪造数据。"
                "如果这段确实与其他文献重合，通常是因为实验设置相似，"
                "应通过补充你自己的实验细节来区分，而不是改数字。"
            ),
        )

    if kind is ContentKind.CONCLUSION:
        return RewriteInstruction(
            strategy=Strategy.RESTRUCTURE,
            kind=kind,
            hint=(
                "换一个表述角度重述这个结论：改变句子结构与论证顺序，"
                "但**必须保留结论的方向与限定条件**（提升/下降、在什么条件下成立）。"
            ),
            rationale="结论类表述的改写空间较大，可以换角度重述而保持结论不变。",
        )

    if kind is ContentKind.METHOD:
        return RewriteInstruction(
            strategy=Strategy.REPHRASE,
            kind=kind,
            hint=(
                "重述方法描述：保留专业术语与专有名词不变，"
                "改变句式与叙述顺序。"
            ),
            rationale=(
                "方法描述里的术语不能替换（换了就不准确），"
                "但句式、语序、连接方式可以改。"
            ),
        )

    return RewriteInstruction(
        strategy=Strategy.REPHRASE,
        kind=kind,
        hint="换一种说法重述这段话，保持原意不变。",
        rationale="一般论述，改写空间较大。",
    )
