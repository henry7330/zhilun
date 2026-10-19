"""F 阶段 · Agent「降重降AI」—— 一个可以独立调用的服务。

## 这个 Agent 做什么

只做一件事：**按外部检测报告，对"有定位"的问题片段做合并改写。**

```
F1 报告解析定位 ──▶ 清单（精确定位 / 模糊定位 / 未定位 分开）
F2 合并改写     ──▶ 一次改写同时满足降重与降AI，双校验都过才采纳
```

## 三条硬规则（来自你的要求）

**① 必须有外部报告，且只处理有定位的片段。**

没给报告 → 整阶段拒绝运行（`ValueError`），不是"跳过"。
**跳过会让用户以为"检查过了没问题"**，那是这整套系统最想避免的失败模式。

**② 未定位的片段直接提示用户，不猜位置。**

报告可能来自旧版本文档。定位不到就如实说，让用户自己去找 ——
硬指一个位置会让用户改错地方。

**③ 降完不需要再跑格式规整。**

成立的前提是保真校验守住了不可变的东西（数字/专名/否定/长度）。
实测确认：「如图 3 所示」的图号丢失或改错都会被 `check_fidelity` 拦住，
含 `[1]` 的句子整句豁免不改。

## 怎么用（独立调用）

```python
from copilot.revise import run_revise
from copilot.quality.rewrite import LlmRewriter
from copilot.llm import LlmConfig

report = run_revise(
    draft=open("thesis.md", encoding="utf-8").read(),
    report=dup_report,               # 查重报告或 AI 报告
    report_type="duplicate",         # duplicate | ai | both
    rewriter=LlmRewriter(LlmConfig.from_env()),
)
print(report.summary_line())
```

不传 `rewriter` 时用 `StubRewriter` —— 不会改写任何东西，如实报「无候选」。
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Literal, Sequence

from copilot.contracts import Finding, NodeResult
from copilot.quality.format.report import Severity
from copilot.stage import StageNode, ok_outcome, run_stage

from copilot.revise.combined import (
    STATUS_LABEL,
    ReviseStatus,
    ReviseSuggestion,
    build_instruction,
    revise_unit,
    revise_units,
)

__all__ = [
    "run_revise",
    "build_revise_nodes",
    "ReviseStatus",
    "ReviseSuggestion",
    "STATUS_LABEL",
    "build_instruction",
    "revise_unit",
    "revise_units",
]

STAGE_KEY = "F"
CODE_LOCATE = "F1-LOCATE"
CODE_REVISE = "F2-REVISE"

ReportType = Literal["duplicate", "ai", "both"]

_VALID = ("duplicate", "ai", "both")


def _targets_for(report_type: str) -> tuple[str, ...]:
    if report_type == "both":
        return ("overlap", "style")
    if report_type == "ai":
        return ("style",)
    return ("overlap",)


def build_revise_nodes(
    *,
    draft: str,
    report: str,
    report_type: str = "duplicate",
    rewriter: Any = None,
    max_attempts: int = 3,
) -> tuple[StageNode, ...]:
    """装配两个节点。**报告是必填的** —— 没有报告就没有这个阶段的意义。"""

    def f1_locate(state: dict[str, Any]):
        from copilot.quality.report_parse import FlagType, build_checklist

        flag = FlagType.AI if report_type == "ai" else FlagType.DUPLICATE
        checklist = build_checklist(draft, report, flag_type=flag)

        findings: list[Finding] = []
        located = []
        for item in checklist.items:
            method = getattr(getattr(item, "method", None), "value", "unlocated")
            if method == "unlocated":
                findings.append(
                    Finding(
                        node="F1",
                        code=CODE_LOCATE,
                        severity=Severity.WARNING,
                        message="报告里这条片段**在正文中定位不到**（可能是旧版本内容）",
                        location="",
                        fix="请自行在文档里找到这一段并处理 —— 系统不猜位置",
                        payload={"text": item.item.text[:160]},
                    )
                )
            else:
                located.append(item)
                if method != "exact":
                    findings.append(
                        Finding(
                            node="F1",
                            code=CODE_LOCATE,
                            severity=Severity.INFO,
                            message=f"模糊定位（置信度 {getattr(item, 'confidence', 0):.2f}）—— 请核对后再改",
                            location=getattr(item, "location", ""),
                            payload={"text": item.item.text[:120]},
                        )
                    )

        return ok_outcome(
            findings=findings,
            stats={
                "items": len(checklist.items),
                "located": len(located),
                "unlocated": len(checklist.items) - len(located),
            },
            notes=[
                f"共 {len(checklist.items)} 条：定位 {len(located)} / 未定位 {len(checklist.items)-len(located)}",
                "**只处理有定位的片段** —— 未定位的交回给你自己找",
            ],
            patch={"checklist": checklist, "located": located},
        )

    def f2_revise(state: dict[str, Any]):
        from copilot.quality.rewrite import StubRewriter
        from copilot.quality.rewrite.units import split_units

        located = state.get("located") or []
        if not located:
            return ok_outcome(
                notes=["没有可改的片段（全部未定位或报告为空）"],
                stats={"units": 0, "accepted": 0},
            )

        rw = rewriter or StubRewriter()
        targets = _targets_for(report_type)
        findings: list[Finding] = []
        suggestions: list[ReviseSuggestion] = []
        units_total = 0

        for item in located:
            for unit in split_units(item, draft):
                units_total += 1
                s = revise_unit(
                    unit, rewriter=rw, targets=targets, max_attempts=max_attempts
                )
                suggestions.append(s)

                if s.status == ReviseStatus.ACCEPTED:
                    findings.append(
                        Finding(
                            node="F2",
                            code=CODE_REVISE,
                            severity=Severity.INFO,
                            message=(
                                f"可采纳改写（{'/'.join(sorted(s.targets))}）："
                                f"重合 {s.effect.after_run} 字，"
                                f"包含 {s.effect.after_containment:.0%}"
                                if s.effect
                                else "可采纳改写"
                            ),
                            location=unit.location,
                            fix="审阅后决定是否采用",
                            payload={
                                "original": unit.text,
                                "rewritten": s.chosen,
                                "effect": s.effect.to_dict() if s.effect else None,
                            },
                        )
                    )
                elif s.status == ReviseStatus.SKIPPED_PROTECTED:
                    findings.append(
                        Finding(
                            node="F2",
                            code=CODE_REVISE,
                            severity=Severity.INFO,
                            message=f"引用豁免，不改：{s.note}",
                            location=unit.location,
                        )
                    )
                elif s.status in (
                    ReviseStatus.REJECTED_EFFECT,
                    ReviseStatus.REJECTED_STYLE,
                ):
                    findings.append(
                        Finding(
                            node="F2",
                            code=CODE_REVISE,
                            severity=Severity.WARNING,
                            message=f"自动改写未达标，需人工：{s.note}",
                            location=unit.location,
                            fix="人工重写这一段，或补充信息后重试",
                            payload={"original": unit.text},
                        )
                    )
                elif s.status == ReviseStatus.REJECTED_FIDELITY:
                    findings.append(
                        Finding(
                            node="F2",
                            code=CODE_REVISE,
                            severity=Severity.WARNING,
                            message=f"候选改动原意，已弃用：{s.note}",
                            location=unit.location,
                            fix="机械候选不可用 —— 需人工改写",
                            payload={"original": unit.text, "candidate": s.chosen},
                        )
                    )
                # NO_CANDIDATE 不产生 Finding，只进统计

        accepted = sum(1 for s in suggestions if s.accepted)
        no_candidate = sum(1 for s in suggestions if s.status == ReviseStatus.NO_CANDIDATE)

        notes = [f"目标：{' + '.join(sorted(targets))}"]
        if no_candidate:
            notes.append(
                f"{no_candidate} 个片段没有候选 —— 若未接入模型，这是预期的（用 StubRewriter）"
            )

        return ok_outcome(
            findings=findings,
            stats={
                "units": units_total,
                "accepted": accepted,
                "no_candidate": no_candidate,
                "targets": sorted(targets),
            },
            notes=notes,
            patch={"suggestions": suggestions},
        )

    return (
        StageNode(code="F1", name="报告解析定位", run=f1_locate,
                  description="解析外部报告，只保留有定位的片段"),
        StageNode(code="F2", name="合并改写", run=f2_revise,
                  requires_model=rewriter is not None,
                  description="一次改写同时满足降重与降AI，双校验都过才采纳"),
    )


def run_revise(
    *,
    draft: str,
    report: str,
    report_type: str = "duplicate",
    rewriter: Any = None,
    max_attempts: int = 3,
    label: str = "<inline>",
    only: Sequence[str] | None = None,
    on_node: Callable[[str, NodeResult], None] | None = None,
):
    """**F 阶段的对外入口。**

    ``report`` 是**必填**的：没有外部报告就没有这个阶段存在的意义。
    传空字符串会直接抛 ``ValueError`` —— 不做"静默跳过"。
    """
    if report_type not in _VALID:
        raise ValueError(f"report_type 必须是 {_VALID} 之一，收到 {report_type!r}")
    if not (report or "").strip():
        raise ValueError(
            "降重降AI 阶段必须提供外部检测报告（查重或 AI 检测）。"
            "没有报告时本阶段不做任何事 —— 这不是可以「跳过」的检查。"
        )
    if not (draft or "").strip():
        raise ValueError("draft 为空，没有可改的内容")

    nodes = build_revise_nodes(
        draft=draft, report=report, report_type=report_type,
        rewriter=rewriter, max_attempts=max_attempts,
    )
    return run_stage(
        stage=STAGE_KEY,
        label=label,
        nodes=nodes,
        state={"draft": draft, "report": report, "report_type": report_type},
        only=only,
        on_node=on_node,
    )
