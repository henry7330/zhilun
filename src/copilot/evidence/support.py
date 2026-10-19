"""支持程度判断：证据是否支持结论，以及支持到什么程度。

核心洞察：很大一部分「不支持」是**确定性可判的**
------------------------------------------------
人们常以为"证据是否支持结论"必须交给模型判断。但有一类矛盾完全不需要模型：

- **极性矛盾**：结论说"准确率提升"，原文说"性能下降" —— 二者不相容；
- **数字缺失**：结论声称"提升 5%"，原文里根本没有 5% 这个数。

这两类恰好是幻觉最常见的形式（把结论写反、把数字写错）。
把它们做成规则，就能得到一个**离线、可复现、零成本**的基线，
再用它与模型方案做消融对照 —— 这才回答得出"模型到底带来了多少增益"。

剩余的部分（同义改写、跨句推理、隐含前提）确实需要语义理解，
规则做不到，这一点在 KDoc 里明说，不含糊。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Protocol

from copilot.evidence.models import Claim, SupportStatus
from copilot.evidence.text import coverage, numbers_in, polarity_conflicts

__all__ = [
    "SupportSignals",
    "SupportJudgment",
    "SupportJudge",
    "RuleBasedJudge",
]


@dataclass(frozen=True)
class SupportSignals:
    """判断依据的**可审计**快照。

    保留全部信号（而不只是最终状态）有两个用处：
    1. 线上出问题时能解释"为什么判成这个状态"；
    2. 调阈值时能回放同一批样本，看改动的影响。
    """

    coverage: float
    claim_numbers: tuple[str, ...]
    evidence_numbers: tuple[str, ...]
    missing_numbers: tuple[str, ...]
    polarity_conflicts: tuple[tuple[str, str], ...]

    @property
    def has_polarity_conflict(self) -> bool:
        return bool(self.polarity_conflicts)


@dataclass
class SupportJudgment:
    status: SupportStatus
    signals: SupportSignals
    reasons: list[str] = field(default_factory=list)


class SupportJudge(Protocol):
    """支持程度判断接口。规则基线与模型方案都应实现它，以便互换对照。"""

    def judge(self, claim: Claim) -> SupportJudgment: ...


class RuleBasedJudge:
    """规则式支持判断：**离线、确定性、零调用成本**。

    判定顺序（从强到弱）：

    1. 无证据 → ``UNSUPPORTED``（缺信息，不是编造）
    2. 极性冲突 → ``FABRICATED``（确定性矛盾）
    3. 覆盖率 >= ``supported_threshold`` 且**结论中的数字都能在证据里找到**
       → ``SUPPORTED``
    4. 覆盖率 >= ``partial_threshold`` → ``PARTIALLY_SUPPORTED``
    5. 其余 → ``UNSUPPORTED``

    阈值取舍：调高 ``supported_threshold`` 会降低误判为"支持"的概率，
    但会把更多真实支持的结论降级。这是精确率/召回率的权衡，
    应当在固定评测集上调节，而不是凭感觉 —— 见 P9。
    """

    def __init__(
        self,
        *,
        supported_threshold: float = 0.70,
        partial_threshold: float = 0.35,
    ) -> None:
        if not 0.0 <= partial_threshold <= supported_threshold <= 1.0:
            raise ValueError(
                "阈值必须满足 0 <= partial_threshold <= supported_threshold <= 1；"
                f"当前 partial={partial_threshold}, supported={supported_threshold}"
            )
        self.supported_threshold = supported_threshold
        self.partial_threshold = partial_threshold

    # ── 内部 ────────────────────────────────────────────────

    def _signals(self, claim: Claim) -> SupportSignals:
        evidence_text = "\n".join(s.text for s in claim.evidence)
        claim_nums = numbers_in(claim.text)
        ev_nums = numbers_in(evidence_text)

        # 数字比较用归一化后的集合：证据里出现即可，不要求顺序
        ev_set = set(ev_nums)
        missing = tuple(n for n in claim_nums if n not in ev_set)

        return SupportSignals(
            coverage=round(coverage(claim.text, evidence_text), 4),
            claim_numbers=claim_nums,
            evidence_numbers=ev_nums,
            missing_numbers=missing,
            polarity_conflicts=polarity_conflicts(claim.text, evidence_text),
        )

    # ── 对外 ────────────────────────────────────────────────

    def judge(self, claim: Claim) -> SupportJudgment:
        if not claim.evidence:
            empty = SupportSignals(0.0, numbers_in(claim.text), (), numbers_in(claim.text), ())
            return SupportJudgment(
                SupportStatus.UNSUPPORTED, empty, ["该 claim 没有绑定任何证据"]
            )

        signals = self._signals(claim)
        reasons: list[str] = []

        # 1) 极性矛盾：确定性判定，优先级最高
        if signals.has_polarity_conflict:
            pairs = "、".join(f"{a}↔{b}" for a, b in signals.polarity_conflicts)
            reasons.append(f"结论与证据极性相反（{pairs}），二者不相容")
            return SupportJudgment(SupportStatus.FABRICATED, signals, reasons)

        # 2) 覆盖率与数字一致性
        if signals.coverage >= self.supported_threshold:
            if signals.missing_numbers:
                missing = "、".join(signals.missing_numbers)
                reasons.append(
                    f"文本覆盖率达 {signals.coverage:.2f}，但结论中的数字（{missing}）"
                    "未在证据中出现，因此不能判为完全支持"
                )
                return SupportJudgment(SupportStatus.PARTIALLY_SUPPORTED, signals, reasons)
            reasons.append(f"文本覆盖率达 {signals.coverage:.2f}，且数字一致")
            return SupportJudgment(SupportStatus.SUPPORTED, signals, reasons)

        if signals.coverage >= self.partial_threshold:
            reasons.append(
                f"文本覆盖率仅 {signals.coverage:.2f}（低于完全支持阈值 "
                f"{self.supported_threshold}），只能算部分支持"
            )
            return SupportJudgment(SupportStatus.PARTIALLY_SUPPORTED, signals, reasons)

        reasons.append(
            f"文本覆盖率仅 {signals.coverage:.2f}，证据与结论几乎没有重叠"
        )
        return SupportJudgment(SupportStatus.UNSUPPORTED, signals, reasons)
