"""E4 · 降重效果度量 —— 用数字回答"到底降了多少"。

为什么降重必须有度量
--------------------
**降重不能靠"感觉换了个说法"。**

没有度量会有两个后果：
- 无法判断改写是否有效，**也就无法重试**；
- 报告里给不出任何依据，用户只能盲信。

**直接度量 vs 代理指标（必须先分清）**
--------------------------------------
降重的目标是降低与**原出处**文字的重叠。但"原出处"未必拿得到：

- 报告给了**来源原文** → 直接度量「改写后 ↔ 来源原文」= **直接度量** ✅
- 报告只有**用户自己那段重复文字** → 度量「改写后 ↔ 用户原句」= **代理指标** ⚠️

代理指标**不等于**真实降重率：改写后与自己的原句不重合，
**大概率**也与来源不重合 —— 但这是推理，不是测量。

因此本模块把这件事记在 :attr:`RewriteEffect.measured_against` 里，
报告会照原样写出来，**不把代理指标冒充成真实降重率**。
"""

from __future__ import annotations

from dataclasses import dataclass

from copilot.overlap import longest_common_run, overlap_ratio, ngram_containment

__all__ = [
    "RewriteEffect",
    "measure_effect",
    "DEFAULT_MAX_RUN",
    "DEFAULT_MAX_CONTAINMENT",
]

#: 达标线：最长连续重叠不超过这么多字符。
#: 取 12 是因为 12 个字已经不太可能是"通用短语的巧合重合"。
DEFAULT_MAX_RUN = 12

#: 达标线：n-gram 包含率不超过这个值。
#: 0.35 意味着新句子里最多三分之一的词组还来自原句。
DEFAULT_MAX_CONTAINMENT = 0.35


@dataclass
class RewriteEffect:
    """一次改写的效果数字。"""

    #: 改之前：原句自身的长度（作为参照）
    original_length: int
    #: 改写后与原句的最长连续重叠（字符数）
    after_run: int
    #: 改写后与原句的 n-gram 包含率
    after_containment: float

    #: 度量对象：``"source"``（来源原文，直接度量）或 ``"original"``（代理指标）
    measured_against: str
    #: 达标线
    max_run: int = DEFAULT_MAX_RUN
    max_containment: float = DEFAULT_MAX_CONTAINMENT

    @property
    def meets_target(self) -> bool:
        """两个指标**都**达标才算成功。

        只看一个会被骗：把长句拆成短句能压低连续重叠，
        但词组还是抄的（``00-core/tests/test_overlap.py`` 里有专门验证）。
        """
        return self.after_run <= self.max_run and self.after_containment <= self.max_containment

    @property
    def is_proxy(self) -> bool:
        """是否用的是代理指标。"""
        return self.measured_against == "original"

    def summary(self) -> str:
        base = (
            f"最长连续重叠 {self.after_run} 字"
            f"（达标线 {self.max_run}）；"
            f"n-gram 包含率 {self.after_containment:.0%}"
            f"（达标线 {self.max_containment:.0%}）"
        )
        if self.is_proxy:
            return base + "；⚠️ 代理指标（报告未给来源原文，以原句为基准）"
        return base + "；直接度量（对照来源原文）"

    def to_dict(self) -> dict:
        return {
            "measured_against": self.measured_against,
            "original_length": self.original_length,
            "after_run": self.after_run,
            "after_containment": round(self.after_containment, 4),
            "max_run": self.max_run,
            "max_containment": self.max_containment,
            "meets_target": self.meets_target,
            "is_proxy": self.is_proxy,
        }


def measure_effect(
    rewritten: str,
    original: str,
    *,
    source_text: str | None = None,
    max_run: int = DEFAULT_MAX_RUN,
    max_containment: float = DEFAULT_MAX_CONTAINMENT,
) -> RewriteEffect:
    """度量一次改写的效果。

    ``source_text`` 是报告给出的来源原文；**有就用它做基准**（直接度量），
    没有才退回用 ``original``（代理指标）。
    """
    use_source = bool(source_text and source_text.strip())
    baseline = source_text if use_source else original

    return RewriteEffect(
        original_length=len(original),
        after_run=longest_common_run(rewritten, baseline),
        after_containment=ngram_containment(rewritten, baseline),
        measured_against="source" if use_source else "original",
        max_run=max_run,
        max_containment=max_containment,
    )


def overlap_snapshot(text: str, baseline: str) -> dict[str, float]:
    """给报告用的原始快照（含"最长连续重叠占多长比例"）。"""
    return {
        "longest_run": float(longest_common_run(text, baseline)),
        "overlap_ratio": overlap_ratio(text, baseline),
        "ngram_containment": ngram_containment(text, baseline),
    }
