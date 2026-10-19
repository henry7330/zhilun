"""E4 · 降重流水线 —— 改写 → 度量 → 保真 → 重试。

每一步都在防一个具体的失败模式
------------------------------
========================  ==========================================
步骤                       不做会怎样
========================  ==========================================
受保护片段跳过              引用被改写 → 篡改他人原文
策略选择                    硬改定义/数据 → 不准确甚至伪造
**保真校验**                改完意思变了 → 比不降更糟
**效果度量**                改了但没降下来 → 用户以为处理过了
**重试有上限，超限如实报告**  无上限重试烧钱，还掩盖"这段改不动"
========================  ==========================================

判定顺序：**先保真，再效果**
----------------------------
保真是**否决条件**：不保真的候选即使降重效果再好也不能用。
效果是**达标条件**：保真但没降够的候选，可以重试；重试仍不行就如实报告。

这个顺序不能反 —— 反过来会出现"为了降重而牺牲准确"的产出。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from copilot.quality.report_parse.models import Checklist, LocatedItem
from copilot.quality.rewrite.effect import (
    DEFAULT_MAX_CONTAINMENT,
    DEFAULT_MAX_RUN,
    RewriteEffect,
    measure_effect,
)
from copilot.quality.rewrite.fidelity import (
    FidelityResult,
    check_fidelity,
)
from copilot.quality.rewrite.rewriter import RewriteRequest, Rewriter
from copilot.quality.rewrite.strategy import (
    ContentKind,
    Strategy,
    classify,
    choose_instruction,
)
from copilot.quality.rewrite.units import RewriteUnit, split_units

__all__ = [
    "SuggestionStatus",
    "RewriteSuggestion",
    "rewrite_unit",
    "rewrite_item",
    "DEFAULT_MAX_ATTEMPTS",
]

#: 最多尝试几次。
#:
#: 重试只对**随机性改写器**（真实模型）有意义；确定性改写器重试结果相同。
#:
#: ★ 为什么是 3 而不是 2（实测调整）：
#: `eval/run_rewrite_eval.py` 测得 v2 提示词 + 单次通过率 **83%**（10/12），
#: 重试到 3 次后升到 **100%**（12/12）。失败样本本来就不多，
#: 多一次重试的成本远小于"这一段要人工改"的成本。
#:
#: 注意：单看 B→C 这一步的差异**统计不显著**（+17% [+0%,+42%]，p=0.50），
#: 取 3 是工程判断而不是统计结论 —— 报告里把这一点写明了。
DEFAULT_MAX_ATTEMPTS = 3


class SuggestionStatus(str):
    """建议的状态。用常量而不是 Enum，便于直接写进报告。"""

    ACCEPTED = "accepted"
    """改写被采用。"""

    SKIPPED_PROTECTED = "skipped_protected"
    """受保护片段（引用），未改写。"""

    SKIPPED_CITE = "skipped_cite"
    """建议改为规范引用，而不是改写。"""

    SKIPPED_NO_CHANGE = "skipped_no_change"
    """策略要求不改（如数据/公式）。"""

    NO_CANDIDATE = "no_candidate"
    """改写器没有给出候选。"""

    REJECTED_FIDELITY = "rejected_fidelity"
    """保真校验未通过，弃用。"""

    REJECTED_EFFECT = "rejected_effect"
    """降重未达标。"""


@dataclass
class RewriteSuggestion:
    """一条单元的处理结果。"""

    unit: RewriteUnit
    status: str
    kind: ContentKind = ContentKind.OTHER
    strategy: Strategy = Strategy.REPHRASE

    chosen: str = ""
    candidates: list[str] = field(default_factory=list)

    effect: RewriteEffect | None = None
    fidelity: FidelityResult | None = None
    attempts: int = 0
    note: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == SuggestionStatus.ACCEPTED

    @property
    def needs_attention(self) -> bool:
        """需要用户处理（没改成，但不是因为不该改）。"""
        return self.status in (
            SuggestionStatus.REJECTED_FIDELITY,
            SuggestionStatus.REJECTED_EFFECT,
            SuggestionStatus.NO_CANDIDATE,
        )

    def to_dict(self) -> dict:
        return {
            "unit_id": self.unit.unit_id,
            "location": self.unit.location,
            "line": self.unit.line + 1,
            "original": self.unit.text,
            "status": self.status,
            "kind": self.kind.value,
            "strategy": self.strategy.value,
            "chosen": self.chosen,
            "candidates": list(self.candidates),
            "attempts": self.attempts,
            "effect": None if self.effect is None else self.effect.to_dict(),
            "fidelity": None if self.fidelity is None else self.fidelity.to_dict(),
            "note": self.note,
        }


def _best(rejected: list[tuple[RewriteEffect, FidelityResult, str]]) -> tuple | None:
    """从被拒的候选里挑"最接近达标"的一个，用于报告里说明尽力了。"""
    if not rejected:
        return None
    return max(rejected, key=lambda r: (-r[0].after_run, -r[0].after_containment))


def rewrite_unit(
    unit: RewriteUnit,
    *,
    rewriter: Rewriter,
    source_text: str | None = None,
    context: str = "",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_run: int = DEFAULT_MAX_RUN,
    max_containment: float = DEFAULT_MAX_CONTAINMENT,
) -> RewriteSuggestion:
    """处理一个改写单元。"""
    # ── 受保护：不动 ──
    if unit.protected:
        return RewriteSuggestion(
            unit=unit,
            status=SuggestionStatus.SKIPPED_PROTECTED,
            note=unit.protected_reason or "受保护片段",
        )

    # ── 分类 → 策略 ──
    kind = classify(unit.text)
    instruction = choose_instruction(kind)
    unit.strategy = instruction.strategy.value

    if instruction.strategy is Strategy.CITE_INSTEAD:
        return RewriteSuggestion(
            unit=unit,
            status=SuggestionStatus.SKIPPED_CITE,
            kind=kind,
            strategy=instruction.strategy,
            note=instruction.rationale,
        )

    if instruction.strategy is Strategy.SKIP:
        return RewriteSuggestion(
            unit=unit,
            status=SuggestionStatus.SKIPPED_NO_CHANGE,
            kind=kind,
            strategy=instruction.strategy,
            note=instruction.rationale,
        )

    # ── 生成 → 校验 → 重试 ──
    request = RewriteRequest(text=unit.text, instruction=instruction, context=context)
    rejected: list[tuple[RewriteEffect, FidelityResult, str]] = []
    fidelity_failed: list[FidelityResult] = []
    attempts = 0

    for _ in range(max(1, max_attempts)):
        attempts += 1
        candidates = [c for c in rewriter.rewrite(request) if c and c.strip()]
        if not candidates:
            continue

        for cand in candidates:
            effect = measure_effect(
                cand,
                unit.text,
                source_text=source_text,
                max_run=max_run,
                max_containment=max_containment,
            )
            fidelity = check_fidelity(cand, unit.text, source_text=source_text)

            # **先保真，再效果** —— 保真是否决条件
            if not fidelity.ok:
                fidelity_failed.append(fidelity)
                continue

            if effect.meets_target:
                return RewriteSuggestion(
                    unit=unit,
                    status=SuggestionStatus.ACCEPTED,
                    kind=kind,
                    strategy=instruction.strategy,
                    chosen=cand,
                    candidates=candidates,
                    effect=effect,
                    fidelity=fidelity,
                    attempts=attempts,
                    note=instruction.rationale,
                )

            rejected.append((effect, fidelity, cand))

    # ── 没成功，如实报告 ──
    if fidelity_failed and not rejected:
        worst = fidelity_failed[0]
        return RewriteSuggestion(
            unit=unit,
            status=SuggestionStatus.REJECTED_FIDELITY,
            kind=kind,
            strategy=instruction.strategy,
            candidates=[],
            effect=None,
            fidelity=worst,
            attempts=attempts,
            note=(
                "改写候选全部未通过保真校验，已弃用 —— "
                f"{worst.summary()}"
            ),
        )

    best = _best(rejected)
    if best is not None:
        effect, fidelity, cand = best
        return RewriteSuggestion(
            unit=unit,
            status=SuggestionStatus.REJECTED_EFFECT,
            kind=kind,
            strategy=instruction.strategy,
            chosen=cand,
            candidates=[],
            effect=effect,
            fidelity=fidelity,
            attempts=attempts,
            note=(
                f"尝试 {attempts} 次仍未达到降重目标。{effect.summary()}。"
                "**请人工改写这一段** —— 它可能本身就不适合机器改写。"
            ),
        )

    return RewriteSuggestion(
        unit=unit,
        status=SuggestionStatus.NO_CANDIDATE,
        kind=kind,
        strategy=instruction.strategy,
        attempts=attempts,
        note=(
            f"改写器 `{rewriter.name}` 没有产出候选。"
            "若使用的是 StubRewriter，说明当前没有接入模型。"
        ),
    )


def rewrite_item(
    item: LocatedItem,
    doc_text: str,
    *,
    rewriter: Rewriter,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_run: int = DEFAULT_MAX_RUN,
    max_containment: float = DEFAULT_MAX_CONTAINMENT,
) -> list[RewriteSuggestion]:
    """处理一个已定位的问题条目（可能含多句）。"""
    units = split_units(item, doc_text)
    context = doc_text[max(0, (item.char_start or 0) - 200) : (item.char_end or 0) + 200]
    return [
        rewrite_unit(
            unit,
            rewriter=rewriter,
            source_text=item.item.source_text,
            context=context,
            max_attempts=max_attempts,
            max_run=max_run,
            max_containment=max_containment,
        )
        for unit in units
    ]


def rewrite_checklist(
    checklist: Checklist,
    *,
    rewriter: Rewriter,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> list[RewriteSuggestion]:
    """处理整份清单。未定位的条目会被跳过（不知道改哪，不能瞎改）。"""
    out: list[RewriteSuggestion] = []
    for item in checklist.needing_rewrite():
        out.extend(
            rewrite_item(
                item,
                checklist.doc_text,
                rewriter=rewriter,
                max_attempts=max_attempts,
            )
        )
    return out
