"""E5 · 改写策略 —— 把诊断结果映射成**具体的改写要求**。

为什么策略要与诊断一一对应
--------------------------
「改得像人一点」不是可执行指令。而
「**删掉这两个连接词，并把 38 字的句子拆成 21 + 17 字**」是可执行的。

策略与诊断一一对应，才能保证"改的是问题所在"，
而不是让模型自由发挥成**同义词替换式的伪改写**。

★ 一条硬约束：不得编造
----------------------
"补充具体信息"这条策略**绝不允许模型生成不存在的数字或事实**。
只能：
- 从上下文里已有的信息提取；
- 或者提示用户"这里可以补充你实验的具体条件"。

**编造数据的危害远大于 AI 味** —— 一段"像 AI 写的"内容只是风格问题，
一段编造的实验数据是学术不端。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from copilot.quality.deai.diagnose import Diagnosis

__all__ = [
    "StyleStrategy",
    "StyleInstruction",
    "DeaiRequest",
    "choose_instruction",
]


class StyleStrategy(str, Enum):
    """针对具体病因的改写策略。"""

    VARY_SENTENCE = "vary_sentence"
    """打散句式：长短交错、拆分或合并。"""

    TRIM_TRANSITIONS = "trim_transitions"
    """删减套话连接词，换成具体逻辑关系。"""

    ADD_SPECIFICS = "add_specifics"
    """补充具体信息（**不得编造**）。"""

    BREAK_PARALLELISM = "break_parallelism"
    """打破排比对仗。"""

    NONE = "none"
    """没有诊断出问题。"""

    @property
    def label(self) -> str:
        return {
            "vary_sentence": "打散句式",
            "trim_transitions": "删减连接词",
            "add_specifics": "补充具体信息",
            "break_parallelism": "打破排比",
            "none": "无需改写",
        }[self.value]


#: 诊断特征 → 策略
_FEATURE_TO_STRATEGY: dict[str, StyleStrategy] = {
    "sentence_length_cv": StyleStrategy.VARY_SENTENCE,
    "transition_density": StyleStrategy.TRIM_TRANSITIONS,
    "specificity": StyleStrategy.ADD_SPECIFICS,
    "parallelism": StyleStrategy.BREAK_PARALLELISM,
}

#: 每条策略对应的具体改写要求
_STRATEGY_HINT: dict[StyleStrategy, str] = {
    StyleStrategy.VARY_SENTENCE: (
        "打散句式：让句子长短有明显差别（例如 30 字以上的长句后接一个短句），"
        "不要连续使用长度相近的句子。"
    ),
    StyleStrategy.TRIM_TRANSITIONS: (
        "删减套话连接词（此外/然而/综上所述/值得注意的是 等），"
        "如果要保留逻辑关系，用**具体的**因果、条件、对比来体现，而不是靠连接词。"
    ),
    StyleStrategy.ADD_SPECIFICS: (
        "让表述更具体：使用原文与上下文里**已经存在的**数字、条件、方法名称。"
        "**绝对不要编造任何数据、数值或事实** —— 如果原文没有具体信息，就保持笼统。"
    ),
    StyleStrategy.BREAK_PARALLELISM: (
        "打破整齐的排比：不要「一方面…另一方面」全部对称展开，"
        "把其中一支说得更具体、更长或更短。"
    ),
}

#: 每条策略的理由（写进报告，让用户理解为什么这么改）
_STRATEGY_RATIONALE: dict[StyleStrategy, str] = {
    StyleStrategy.VARY_SENTENCE: "句长过于均匀是机器文本的常见特征，长短交错读起来更自然。",
    StyleStrategy.TRIM_TRANSITIONS: "过多的连接词会让论述显得空泛，具体逻辑比连接词更有说服力。",
    StyleStrategy.ADD_SPECIFICS: "具体信息是真人写作的显著特征，但**必须是真的**。",
    StyleStrategy.BREAK_PARALLELISM: "完全对称的结构会显得刻意，轻微的不对称更自然。",
}


@dataclass(frozen=True)
class StyleInstruction:
    """交给改写器的风格指令。"""

    strategies: tuple[StyleStrategy, ...]
    hint: str
    rationale: str
    diagnoses: tuple[Diagnosis, ...] = field(default=())

    @property
    def is_empty(self) -> bool:
        return not self.strategies

    def to_dict(self) -> dict:
        return {
            "strategies": [s.value for s in self.strategies],
            "hint": self.hint,
            "rationale": self.rationale,
            "diagnoses": [d.to_dict() for d in self.diagnoses],
        }


@dataclass(frozen=True)
class DeaiRequest:
    """一次降 AI 改写请求。

    形状与 E4 的 ``RewriteRequest`` 一致（都有 ``text`` / ``instruction`` /
    ``context``），因此**可以直接用 E4 的改写器实现**
    —— 见 README 的「与 E4 的共用边界」。
    """

    text: str
    instruction: StyleInstruction
    context: str = ""


def choose_instruction(diagnoses: list[Diagnosis]) -> StyleInstruction:
    """把诊断映射成改写指令。

    多条诊断会合成一条指令（一次改写同时处理多个病因），
    因为分多次改写既费钱，又容易在两次改写之间产生风格断裂。
    """
    if not diagnoses:
        return StyleInstruction(
            strategies=(),
            hint="没有诊断出风格问题。",
            rationale="该片段的风格特征都落在正常范围内，不需要改写。",
            diagnoses=(),
        )

    # 按固定顺序去重：同一策略只出现一次，避免提示词重复
    ordered: list[StyleStrategy] = []
    for d in diagnoses:
        s = _FEATURE_TO_STRATEGY.get(d.feature)
        if s is not None and s not in ordered:
            ordered.append(s)

    hint = "；".join(_STRATEGY_HINT[s] for s in ordered)
    rationale = " ".join(_STRATEGY_RATIONALE[s] for s in ordered)

    return StyleInstruction(
        strategies=tuple(ordered),
        hint=hint,
        rationale=rationale,
        diagnoses=tuple(diagnoses),
    )
