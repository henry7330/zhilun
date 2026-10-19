"""F2 合并改写：**一次改写同时满足降重与降AI**。

## 为什么合并

在旧结构里降重（E4）与降AI（E5）是两条独立流水线，各写一套提示词与校验。
但它们做的是同一件事 —— **改这段文字** —— 只是**目标不同**：

======================  ============================  ==========================
目标                     度量                            达标线
======================  ============================  ==========================
**降重（overlap）**     与原句/来源的连续重叠、n-gram   重叠 ≤ 12 字，包含 ≤ 35%
**降AI（style）**        句长变异系数、连接词密度、具体性   目标特征改善 ≥ 10%
======================  ============================  ==========================

同一段既被判"重复"又被判"疑似 AI"时，如果分两次改：
第一次改完，第二次又把结构改回去 —— **两个目标互相抵消**。

所以合并成一次改写：**提示词里同时给两组要求，校验时两组都跑，都过才采纳。**

## 校验顺序：保真 → 效果 → 风格

保真是**否决条件**：改了意思的候选，效果再好也弃用。
这条顺序不能反 —— 反过来就会出现"为了降重牺牲准确"。

## 与旧 E4/E5 的关系

本模块**复用**它们的全部零件，不重写：

| 零件 | 来源 |
|---|---|
| 切分（保偏移） | `copilot.quality.rewrite.units.split_units` |
| 受保护判定（引用/引号） | `copilot.quality.rewrite.protected.citation_status` |
| 改写器协议 | `copilot.quality.rewrite.rewriter.Rewriter` |
| 保真校验 | `copilot.quality.rewrite.fidelity.check_fidelity` |
| 重合度量 | `copilot.quality.rewrite.effect.measure_effect` |
| 风格画像与校验 | `copilot.quality.deai.features.profile` / `deai.verify.verify_style` |

**唯一新写的是"合并策略与双校验编排"。**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

from copilot.quality.deai.diagnose import diagnose
from copilot.quality.deai.features import StyleProfile, profile
from copilot.quality.deai.verify import StyleVerification, verify_style
from copilot.quality.rewrite.effect import RewriteEffect, measure_effect
from copilot.quality.rewrite.fidelity import FidelityResult, check_fidelity
from copilot.quality.rewrite.protected import citation_status
from copilot.quality.rewrite.rewriter import RewriteRequest, Rewriter
from copilot.quality.rewrite.strategy import RewriteInstruction, Strategy, classify
from copilot.quality.rewrite.units import RewriteUnit, split_units

__all__ = [
    "Target",
    "ReviseStatus",
    "ReviseSuggestion",
    "build_instruction",
    "revise_unit",
    "revise_units",
    "STATUS_LABEL",
    "HINT_OVERLAP",
    "HINT_STYLE",
]

#: 改写目标。一个片段可能同时有多个目标。
Target = Literal["overlap", "style"]


class ReviseStatus(str):
    """建议的状态。用常量而不是 Enum，便于直接写进报告。"""

    ACCEPTED = "accepted"
    """达标且保真通过。"""

    SKIPPED_PROTECTED = "skipped_protected"
    """规范引用 / 引号内，不改。"""

    NO_CANDIDATE = "no_candidate"
    """改写器没给出候选（通常是没接模型）。"""

    REJECTED_FIDELITY = "rejected_fidelity"
    """保真没过 —— **否决条件**，后面不必再看效果。"""

    REJECTED_EFFECT = "rejected_effect"
    """保真过了但重合度没降下来。"""

    REJECTED_STYLE = "rejected_style"
    """保真过了但风格特征没改善。"""


#: 报告里用的中文标签
STATUS_LABEL: dict[str, str] = {
    ReviseStatus.ACCEPTED: "✅ 采纳",
    ReviseStatus.SKIPPED_PROTECTED: "🔒 引用豁免",
    ReviseStatus.NO_CANDIDATE: "⚪ 无候选",
    ReviseStatus.REJECTED_FIDELITY: "❌ 保真未通过",
    ReviseStatus.REJECTED_EFFECT: "❌ 重合未达标",
    ReviseStatus.REJECTED_STYLE: "❌ 风格未改善",
}

#: 降重要求（与 `effect.py` 的达标线方向一致：提示词更严一档）
HINT_OVERLAP = (
    "改变句子结构，不要只替换同义词 —— 那只算润色。"
    "调整语序、换主语视角、拆分或合并句子、改变分句连接方式。"
    "目标：改写后不要有连续 10 个字以上与原文相同。"
)

#: 降AI 要求（与 `deai.diagnose` 的四个阈值对应）
HINT_STYLE = (
    "打散句式长度（长短句交错）、删掉可有可无的套话连接词"
    "（首先/其次/此外/综上所述/值得注意的是）、"
    "在**不编造**的前提下补充原文已有的具体信息（数字、条件、方法名）。"
)

_RATIONALE = (
    "两个目标同时满足：既要与原文拉开字面距离，又要把风格特征修回正常区间。"
    "数字、专有名词、引用标注一律不得改动。"
)


def build_instruction(unit: RewriteUnit, targets: Iterable[Target]) -> RewriteInstruction:
    """把两个目标拼成**一条**改写指令。

    合并成一条而不是两次调用，是为了避免"第一次改结构、第二次改回去"的互相抵消。
    """
    wanted = set(targets)
    parts: list[str] = []
    if "overlap" in wanted:
        parts.append(HINT_OVERLAP)
    if "style" in wanted:
        parts.append(HINT_STYLE)
    if not parts:
        parts.append(HINT_OVERLAP)

    return RewriteInstruction(
        strategy=Strategy.REPHRASE,
        kind=classify(unit.text),
        hint="\n".join(parts),
        rationale=_RATIONALE,
    )


@dataclass
class ReviseSuggestion:
    """一个片段的改写结果。**三个校验的结论都留在里面，便于复核。**"""

    unit: RewriteUnit
    status: str
    chosen: str | None = None
    attempt: int = 0
    targets: frozenset[str] = field(default_factory=frozenset)
    effect: RewriteEffect | None = None
    style: StyleVerification | None = None
    fidelity: FidelityResult | None = None
    note: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == ReviseStatus.ACCEPTED

    def summary_line(self) -> str:
        head = f"[{STATUS_LABEL.get(self.status, self.status)}] {self.unit.location}"
        if self.effect is not None:
            head += (
                f"  重合 {self.effect.after_run} 字 / {self.effect.after_containment:.0%}"
            )
        return head


def _style_improved(before_text: str, after_text: str) -> StyleVerification:
    """跑风格校验。诊断从**原文**现算，不依赖外部传参。"""
    return verify_style(before_text, after_text, diagnoses=diagnose(before_text))


def revise_unit(
    unit: RewriteUnit,
    *,
    rewriter: Rewriter,
    targets: Iterable[Target] = ("overlap",),
    max_attempts: int = 3,
) -> ReviseSuggestion:
    """改一个片段，最多试 ``max_attempts`` 次。

    判定顺序：**保真 → 重合 → 风格**。保真不过就直接弃用，不再看后两项。
    """
    wanted = frozenset(targets)

    if unit.protected:
        return ReviseSuggestion(
            unit=unit,
            status=ReviseStatus.SKIPPED_PROTECTED,
            targets=wanted,
            note=unit.protected_reason or "受保护片段",
        )

    instruction = build_instruction(unit, wanted)
    request = RewriteRequest(text=unit.text, instruction=instruction, context="")

    last: ReviseSuggestion | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            candidates = rewriter.rewrite(request)
        except Exception as exc:  # noqa: BLE001 - 改写器失败不该带崩整批
            return ReviseSuggestion(
                unit=unit,
                status=ReviseStatus.NO_CANDIDATE,
                attempt=attempt,
                targets=wanted,
                note=f"改写器调用失败：{type(exc).__name__}",
            )

        if not candidates:
            last = ReviseSuggestion(
                unit=unit,
                status=ReviseStatus.NO_CANDIDATE,
                attempt=attempt,
                targets=wanted,
                note="改写器没有给出候选（是否未接入模型？）",
            )
            continue

        for cand in candidates:
            fid = check_fidelity(rewritten=cand, original=unit.text)
            if not fid.ok:
                last = ReviseSuggestion(
                    unit=unit, status=ReviseStatus.REJECTED_FIDELITY, chosen=cand,
                    attempt=attempt, targets=wanted, fidelity=fid, note=fid.summary(),
                )
                continue

            eff = measure_effect(rewritten=cand, original=unit.text)
            style = _style_improved(unit.text, cand) if "style" in wanted else None

            # 两个目标都要过；只要求其一就只看那一个
            overlap_ok = eff.meets_target if "overlap" in wanted else True
            style_ok = (style is not None and style.style_improved) if "style" in wanted else True

            if overlap_ok and style_ok:
                return ReviseSuggestion(
                    unit=unit, status=ReviseStatus.ACCEPTED, chosen=cand,
                    attempt=attempt, targets=wanted, effect=eff, style=style,
                    fidelity=fid, note=f"第 {attempt} 次尝试达标",
                )

            last = ReviseSuggestion(
                unit=unit,
                status=ReviseStatus.REJECTED_EFFECT if not overlap_ok else ReviseStatus.REJECTED_STYLE,
                chosen=cand, attempt=attempt, targets=wanted,
                effect=eff, style=style, fidelity=fid,
                note=(
                    f"重合 {eff.after_run} 字 / {eff.after_containment:.0%}"
                    + ("；风格未改善" if not style_ok else "")
                ),
            )

    return last or ReviseSuggestion(
        unit=unit, status=ReviseStatus.NO_CANDIDATE, targets=wanted,
        note="没有任何候选",
    )


def revise_units(
    units: Sequence[RewriteUnit],
    *,
    rewriter: Rewriter,
    targets: Iterable[Target] = ("overlap",),
    max_attempts: int = 3,
) -> list[ReviseSuggestion]:
    """批量改。单个片段的失败不影响其他片段。"""
    wanted = tuple(targets)
    return [
        revise_unit(u, rewriter=rewriter, targets=wanted, max_attempts=max_attempts)
        for u in units
    ]
