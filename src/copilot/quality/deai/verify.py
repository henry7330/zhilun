"""E5 · 风格与保真双校验。

为什么必须**两项都查**
----------------------
======================  ==========================  ==================================
只做其中一项              结果                          问题
======================  ==========================  ==================================
只做风格校验              改得"像人写的"但**意思变了**    这是最坏的结果
只做保真校验              **什么都没改也能通过**          保真当然通过
**两项都做**              既改善风格，又不失真             ✅
======================  ==========================  ==================================

**冲突时保真优先**：宁可不改，也不能改动原意。
这与 E4 的「定义类建议引用而非改写」是同一条原则 ——
**准确性的优先级永远高于降风格。**
"""

from __future__ import annotations

from dataclasses import dataclass, field

from copilot.quality.deai.diagnose import Diagnosis
from copilot.quality.deai.features import StyleProfile, profile
from copilot.quality.rewrite.fidelity import FidelityResult, check_fidelity

__all__ = [
    "FeatureChange",
    "StyleVerification",
    "verify_style",
    "FEATURE_DIRECTION",
    "MIN_IMPROVEMENT",
]

#: 每个特征"往哪个方向改才算改善"。
#: ``+1`` = 越大越好；``-1`` = 越小越好。
FEATURE_DIRECTION: dict[str, int] = {
    "sentence_length_cv": +1,   # 变异系数越大越自然
    "transition_density": -1,   # 连接词越少越好
    "specificity": +1,          # 具体信息越多越好
    "parallelism": -1,          # 排比越少越好
}

#: 认为"确有改善"的最小相对变化。
#: 太小的变化没有实际意义 —— 那只是措辞抖动，不是风格改变。
MIN_IMPROVEMENT = 0.10


@dataclass
class FeatureChange:
    """一个特征的前后变化。"""

    feature: str
    before: float
    after: float
    #: +1 越大越好 / -1 越小越好
    direction: int

    @property
    def relative_change(self) -> float:
        if self.before == 0:
            return 0.0 if self.after == 0 else float("inf")
        return (self.after - self.before) / abs(self.before)

    @property
    def improved(self) -> bool:
        """是否朝正确方向有**足够大**的变化。"""
        change = self.relative_change
        if self.direction > 0:
            return change >= MIN_IMPROVEMENT
        return -change >= MIN_IMPROVEMENT

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "before": round(self.before, 3),
            "after": round(self.after, 3),
            "direction": self.direction,
            "relative_change": (
                None if self.relative_change == float("inf") else round(self.relative_change, 4)
            ),
            "improved": self.improved,
        }

    def describe(self) -> str:
        arrow = "↑" if self.after > self.before else ("↓" if self.after < self.before else "=")
        flag = "✅" if self.improved else "⚪"
        return f"{flag} {self.feature} {self.before:.3f} → {self.after:.3f} {arrow}"


@dataclass
class StyleVerification:
    """双校验结果。"""

    before: StyleProfile
    after: StyleProfile
    changes: list[FeatureChange] = field(default_factory=list)
    fidelity: FidelityResult | None = None
    #: 针对的诊断特征
    targeted: tuple[str, ...] = ()

    @property
    def style_improved(self) -> bool:
        """**针对的特征里至少有一项**确有改善。

        为什么不要求全部改善：一次改写通常只能改善一两个维度，
        要求全部改善会把大量可用的候选拒掉。
        """
        return any(c.improved for c in self.changes if c.feature in self.targeted)

    @property
    def fidelity_ok(self) -> bool:
        return self.fidelity is not None and self.fidelity.ok

    @property
    def ok(self) -> bool:
        """两项都过才算通过。**保真是否决条件。**"""
        return self.fidelity_ok and self.style_improved

    def summary(self) -> str:
        parts = [c.describe() for c in self.changes]
        fid = "保真通过" if self.fidelity_ok else (
            f"保真未通过：{self.fidelity.summary() if self.fidelity else '未校验'}"
        )
        style = "风格有改善" if self.style_improved else "风格无明显改善"
        return f"{style}；{fid}。变化：{'; '.join(parts) if parts else '—'}"

    def to_dict(self) -> dict:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "changes": [c.to_dict() for c in self.changes],
            "targeted": list(self.targeted),
            "style_improved": self.style_improved,
            "fidelity_ok": self.fidelity_ok,
            "ok": self.ok,
            "fidelity": None if self.fidelity is None else self.fidelity.to_dict(),
        }


def verify_style(
    before_text: str,
    after_text: str,
    *,
    diagnoses: list[Diagnosis],
    source_text: str | None = None,
) -> StyleVerification:
    """对一次降 AI 改写做双校验。"""
    before = profile(before_text)
    after = profile(after_text)
    targeted = tuple(d.feature for d in diagnoses)

    changes = [
        FeatureChange(
            feature=feature,
            before=getattr(before, feature),
            after=getattr(after, feature),
            direction=FEATURE_DIRECTION[feature],
        )
        for feature in FEATURE_DIRECTION
        if hasattr(before, feature)
    ]

    return StyleVerification(
        before=before,
        after=after,
        changes=changes,
        fidelity=check_fidelity(after_text, before_text, source_text=source_text),
        targeted=targeted,
    )
