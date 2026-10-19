"""E5 · 异常定位 —— 具体诊断"这段的 AI 味在哪"。

为什么"诊断"不能省
------------------
**"疑似 AI"不是一个可执行的指令。**

不同的病因需要**完全不同的改写方向**：

============================  ==================================
诊断                           该怎么做
============================  ==================================
套话连接词太多                  **删连接词**
句长过于均匀                    **打散句式、长短交错**
缺少具体信息                    补充细节（**不得编造**）
排比过工整                     打破结构对称
============================  ==================================

不诊断就直接让模型"改得像人一点"，产出的往往是**同义词替换式的伪改写** ——
读起来还是那个味儿，只是换了一批词。

刻意的范围限制
--------------
本模块只诊断**外部报告指出的**片段，不做全文扫描定性。
理由是"判断哪段像 AI"是外部工具的职责 —— 本模块越界只会重复
那个本身就不可靠的推断。
"""

from __future__ import annotations

from dataclasses import dataclass

from copilot.quality.deai.features import StyleProfile, profile

__all__ = [
    "Diagnosis",
    "diagnose",
    "CV_LOW",
    "TRANSITION_HIGH",
    "SPECIFICITY_LOW",
    "PARALLELISM_HIGH",
]

#: 句长变异系数低于此值 → 句式过于均匀。
#: 0.35 是经验值：真人写作的句长波动通常明显大于这个数。
CV_LOW = 0.35

#: 每句套话连接词数高于此值 → 偏多。
TRANSITION_HIGH = 0.55

#: 每百字具体信息量低于此值 → 泛泛而谈。
SPECIFICITY_LOW = 0.8

#: 每千字排比框架数高于此值 → 结构过工整。
PARALLELISM_HIGH = 0.5


@dataclass(frozen=True)
class Diagnosis:
    """一条诊断。"""

    feature: str
    label: str
    value: float
    #: 参考基线（全文的同一特征）—— 让用户看到"偏离了多少"
    reference: float
    direction: str
    explanation: str
    advice: str

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "label": self.label,
            "value": round(self.value, 3),
            "reference": round(self.reference, 3),
            "direction": self.direction,
            "explanation": self.explanation,
            "advice": self.advice,
        }

    def __str__(self) -> str:
        return f"{self.label}：{self.explanation}"


def diagnose(
    text: str,
    *,
    doc_text: str | None = None,
    doc_profile: StyleProfile | None = None,
) -> list[Diagnosis]:
    """诊断一段文字的风格问题。

    ``doc_text`` / ``doc_profile`` 提供全文基线，用于在解释里给出"偏离多少"。
    没有基线时也能诊断（用绝对阈值），只是解释里少了对比数字。
    """
    p = profile(text)
    ref = doc_profile if doc_profile is not None else (profile(doc_text) if doc_text else None)

    out: list[Diagnosis] = []

    if p.sentences >= 2 and p.sentence_length_cv < CV_LOW:
        out.append(
            Diagnosis(
                feature="sentence_length_cv",
                label="句式过于均匀",
                value=p.sentence_length_cv,
                reference=ref.sentence_length_cv if ref else CV_LOW,
                direction="low",
                explanation=(
                    f"句长变异系数只有 {p.sentence_length_cv:.2f}"
                    f"（低于 {CV_LOW}），句子长度过于整齐"
                ),
                advice="把长短句混起来：拆开一句长句，或把两句短的合并；避免连续等长的句子",
            )
        )

    if p.transition_density > TRANSITION_HIGH:
        out.append(
            Diagnosis(
                feature="transition_density",
                label="套话连接词偏多",
                value=p.transition_density,
                reference=ref.transition_density if ref else TRANSITION_HIGH,
                direction="high",
                explanation=(
                    f"平均每句 {p.transition_density:.2f} 个套话连接词"
                    f"（超过 {TRANSITION_HIGH}）"
                ),
                advice=(
                    "删掉可有可无的「此外/然而/综上所述」，"
                    "或者换成**具体的逻辑关系**（因果、对比、条件）而不是笼统的连接词"
                ),
            )
        )

    if p.specificity < SPECIFICITY_LOW:
        out.append(
            Diagnosis(
                feature="specificity",
                label="缺少具体信息",
                value=p.specificity,
                reference=ref.specificity if ref else SPECIFICITY_LOW,
                direction="low",
                explanation=(
                    f"每百字只有 {p.specificity:.2f} 个具体信息点"
                    f"（数字/专名，低于 {SPECIFICITY_LOW}）"
                ),
                advice=(
                    "补充**你实际有的**具体信息：实验条件、数据规模、方法名称。"
                    "**不要编造数字** —— 编造的危害比 AI 味大得多"
                ),
            )
        )

    if p.parallelism > PARALLELISM_HIGH:
        out.append(
            Diagnosis(
                feature="parallelism",
                label="排比/对仗过工整",
                value=p.parallelism,
                reference=ref.parallelism if ref else PARALLELISM_HIGH,
                direction="high",
                explanation=(
                    f"每千字 {p.parallelism:.2f} 处排比框架"
                    f"（超过 {PARALLELISM_HIGH}）"
                ),
                advice="打破对称结构：不必「一方面…另一方面」全用满，把其中一支说得更具体",
            )
        )

    return out
