"""E5 · 降AI 流水线 —— 诊断 → 策略 → 改写 → 双校验 → 报告。

与 E4 共用改写框架
------------------
切片、受保护片段、改写器接口、重试逻辑 —— 这些 E4 已经实现好了，
这里**直接复用**（``copilot.quality.rewrite``）。

为什么复用而不是各写一套：两者的**流程完全相同**
（切分 → 保护 → 生成 → 校验 → 重试），只有"策略"与"校验目标"不同。
实现两遍意味着两处都要维护重试逻辑、错误处理、模型适配 ——
而它们必须保持一致。

判断标准：**如果一段代码在 E4 和 E5 里长得一样，它就该在共用部分。**
"""

from __future__ import annotations

from dataclasses import dataclass, field

from copilot.quality.deai.diagnose import Diagnosis, diagnose
from copilot.quality.deai.features import StyleProfile, profile
from copilot.quality.deai.strategy import (
    DeaiRequest,
    StyleInstruction,
    choose_instruction,
)
from copilot.quality.deai.verify import StyleVerification, verify_style
from copilot.quality.report_parse.models import Checklist, LocatedItem
# ↓ 复用 E4 的改写框架（改写器接口、受保护判定、RewriteUnit 类型）
from copilot.quality.rewrite.pipeline import DEFAULT_MAX_ATTEMPTS
from copilot.quality.rewrite.rewriter import Rewriter
from copilot.quality.rewrite.units import RewriteUnit
# ↓ 但**不用** E4 的单句切分：E5 需要段落级单元（见 units.py 的说明）
from copilot.quality.deai.units import split_style_units

__all__ = [
    "DeaiStatus",
    "DeaiSuggestion",
    "deai_unit",
    "deai_item",
    "deai_checklist",
]


class DeaiStatus(str):
    """降 AI 建议的状态。"""

    ACCEPTED = "accepted"
    SKIPPED_PROTECTED = "skipped_protected"
    NO_DIAGNOSIS = "no_diagnosis"
    """没诊断出风格问题 —— 不改，而不是硬改。"""

    NO_CANDIDATE = "no_candidate"
    REJECTED_FIDELITY = "rejected_fidelity"
    REJECTED_STYLE = "rejected_style"


@dataclass
class DeaiSuggestion:
    """一条单元的处理结果。"""

    unit: RewriteUnit
    status: str
    diagnoses: list[Diagnosis] = field(default_factory=list)
    instruction: StyleInstruction | None = None

    chosen: str = ""
    verification: StyleVerification | None = None
    attempts: int = 0
    note: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == DeaiStatus.ACCEPTED

    @property
    def needs_attention(self) -> bool:
        return self.status in (
            DeaiStatus.NO_CANDIDATE,
            DeaiStatus.REJECTED_FIDELITY,
            DeaiStatus.REJECTED_STYLE,
        )

    def to_dict(self) -> dict:
        return {
            "unit_id": self.unit.unit_id,
            "line": self.unit.line + 1,
            "original": self.unit.text,
            "status": self.status,
            "strategies": (
                [] if self.instruction is None
                else [s.value for s in self.instruction.strategies]
            ),
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "chosen": self.chosen,
            "attempts": self.attempts,
            "verification": None if self.verification is None else self.verification.to_dict(),
            "note": self.note,
        }


def deai_unit(
    unit: RewriteUnit,
    *,
    rewriter: Rewriter,
    doc_profile: StyleProfile | None = None,
    source_text: str | None = None,
    context: str = "",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> DeaiSuggestion:
    """处理一个改写单元。"""
    if unit.protected:
        return DeaiSuggestion(
            unit=unit,
            status=DeaiStatus.SKIPPED_PROTECTED,
            note=unit.protected_reason or "受保护片段",
        )

    # ── 诊断 ──
    diagnoses = diagnose(unit.text, doc_profile=doc_profile)
    if not diagnoses:
        return DeaiSuggestion(
            unit=unit,
            status=DeaiStatus.NO_DIAGNOSIS,
            note=(
                "该片段的风格特征都落在正常范围内，**不需要改写** —— "
                "为了「降 AI」去硬改正常文字，只会把内容改坏。"
            ),
        )

    instruction = choose_instruction(diagnoses)
    request = DeaiRequest(text=unit.text, instruction=instruction, context=context)

    # ── 生成 → 双校验 → 重试 ──
    fidelity_failed: list[StyleVerification] = []
    style_failed: list[StyleVerification] = []
    attempts = 0
    candidates: list[str] = []

    for _ in range(max(1, max_attempts)):
        attempts += 1
        candidates = [c for c in rewriter.rewrite(request) if c and c.strip()]
        if not candidates:
            continue

        for cand in candidates:
            verification = verify_style(
                unit.text, cand, diagnoses=diagnoses, source_text=source_text
            )
            # **保真是否决条件，先查它**
            if not verification.fidelity_ok:
                fidelity_failed.append(verification)
                continue
            if not verification.style_improved:
                style_failed.append(verification)
                continue
            return DeaiSuggestion(
                unit=unit,
                status=DeaiStatus.ACCEPTED,
                diagnoses=diagnoses,
                instruction=instruction,
                chosen=cand,
                verification=verification,
                attempts=attempts,
                note=instruction.rationale,
            )

    # ── 如实报告失败 ──
    if fidelity_failed:
        v = fidelity_failed[0]
        return DeaiSuggestion(
            unit=unit,
            status=DeaiStatus.REJECTED_FIDELITY,
            diagnoses=diagnoses,
            instruction=instruction,
            attempts=attempts,
            verification=v,
            note=f"改写候选未通过保真校验，已弃用 —— {v.fidelity.summary() if v.fidelity else ''}",
        )

    if style_failed:
        v = style_failed[0]
        return DeaiSuggestion(
            unit=unit,
            status=DeaiStatus.REJECTED_STYLE,
            diagnoses=diagnoses,
            instruction=instruction,
            attempts=attempts,
            verification=v,
            note=(
                f"尝试 {attempts} 次仍未改善风格。{v.summary()}。"
                "**请人工改写** —— 这段可能本来就不适合机器改。"
            ),
        )

    return DeaiSuggestion(
        unit=unit,
        status=DeaiStatus.NO_CANDIDATE,
        diagnoses=diagnoses,
        instruction=instruction,
        attempts=attempts,
        note=(
            f"改写器 `{rewriter.name}` 没有产出候选。"
            "若使用的是 StubRewriter，说明当前没有接入模型。"
        ),
    )


def deai_item(
    item: LocatedItem,
    doc_text: str,
    *,
    rewriter: Rewriter,
    doc_profile: StyleProfile | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> list[DeaiSuggestion]:
    """处理一个已定位的问题条目。"""
    units = split_style_units(item, doc_text)
    context = doc_text[max(0, (item.char_start or 0) - 200) : (item.char_end or 0) + 200]
    return [
        deai_unit(
            unit,
            rewriter=rewriter,
            doc_profile=doc_profile,
            source_text=item.item.source_text,
            context=context,
            max_attempts=max_attempts,
        )
        for unit in units
    ]


def deai_checklist(
    checklist: Checklist,
    *,
    rewriter: Rewriter,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> list[DeaiSuggestion]:
    """处理整份清单。"""
    doc_profile = profile(checklist.doc_text)
    out: list[DeaiSuggestion] = []
    for item in checklist.needing_rewrite():
        out.extend(
            deai_item(
                item,
                checklist.doc_text,
                rewriter=rewriter,
                doc_profile=doc_profile,
                max_attempts=max_attempts,
            )
        )
    return out
