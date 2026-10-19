"""E4 · 降重报告 —— 原文 / 改写 / 效果 / 保真，四栏对照。

为什么必须四栏对照
------------------
只给一个"已降重 12 处"的结论，用户**无从核对**。而改写的是**用户的论文**，
最终责任在他身上 —— 他必须能看懂系统到底改了什么、依据是什么。

四个栏位各回答一个问题：

===========  ======================================
栏位          回答
===========  ======================================
原文          这段原来是什么
改写          建议改成什么
效果          **数字**：重叠降了多少（不是"感觉变了"）
保真          为什么可以相信改写没改变原意
===========  ======================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

from copilot.quality.report_parse.models import Checklist
from copilot.quality.rewrite.pipeline import (
    RewriteSuggestion,
    SuggestionStatus,
)

__all__ = ["RewriteReport", "STATUS_LABEL"]

#: 状态码 → 人话
STATUS_LABEL: dict[str, str] = {
    SuggestionStatus.ACCEPTED: "✅ 已改写",
    SuggestionStatus.SKIPPED_PROTECTED: "🔒 引用豁免",
    SuggestionStatus.SKIPPED_CITE: "📌 建议改为引用",
    SuggestionStatus.SKIPPED_NO_CHANGE: "🔒 数据不改",
    SuggestionStatus.NO_CANDIDATE: "⚠️ 无候选",
    SuggestionStatus.REJECTED_FIDELITY: "❌ 保真未通过",
    SuggestionStatus.REJECTED_EFFECT: "❌ 降重未达标",
}


@dataclass
class RewriteReport:
    """一次降重的完整报告。"""

    doc_path: str
    checklist: Checklist
    suggestions: list[RewriteSuggestion] = field(default_factory=list)
    rewriter_name: str = ""

    # ── 统计 ──

    def by_status(self, status: str) -> list[RewriteSuggestion]:
        return [s for s in self.suggestions if s.status == status]

    def counts(self) -> dict[str, int]:
        accepted = len(self.by_status(SuggestionStatus.ACCEPTED))
        needs = sum(1 for s in self.suggestions if s.needs_attention)
        skipped = sum(
            1
            for s in self.suggestions
            if s.status
            in (
                SuggestionStatus.SKIPPED_PROTECTED,
                SuggestionStatus.SKIPPED_CITE,
                SuggestionStatus.SKIPPED_NO_CHANGE,
            )
        )
        return {
            "units": len(self.suggestions),
            "accepted": accepted,
            "skipped": skipped,
            "needs_attention": needs,
            "unlocated": len(self.checklist.unlocated()),
        }

    def summary_line(self) -> str:
        c = self.counts()
        return (
            f"改写 {c['accepted']} 处 / 未改 {c['skipped']} 处 / "
            f"需人工处理 {c['needs_attention']} 处"
            f"（另有 {c['unlocated']} 处报告条目未能定位）"
        )

    # ── 输出 ──

    def to_dict(self) -> dict:
        return {
            "file": self.doc_path,
            "rewriter": self.rewriter_name,
            "counts": self.counts(),
            "suggestions": [s.to_dict() for s in self.suggestions],
            "unlocated": [it.to_dict() for it in self.checklist.unlocated()],
        }

    def to_markdown(self) -> str:
        c = self.counts()
        lines = [
            "# 降重报告",
            "",
            f"- 文档：`{self.doc_path}`",
            f"- 改写器：`{self.rewriter_name}`",
            f"- 输入：{self.checklist.summary_line()}",
            "",
            "## 结论",
            "",
            f"采纳改写 **{c['accepted']}** 处 / "
            f"刻意不改 **{c['skipped']}** 处 / "
            f"**需人工处理 {c['needs_attention']}** 处",
            "",
        ]

        if not self.suggestions:
            lines += ["没有可改写的内容。", ""]
            return "\n".join(lines)

        # ── 一览表 ──
        lines += [
            "## 一览",
            "",
            "| 位置 | 内容类型 | 策略 | 状态 | 效果 |",
            "|---|---|---|---|---|",
        ]
        for s in self.suggestions:
            effect = "—" if s.effect is None else f"连续重叠 {s.effect.after_run} 字"
            lines.append(
                f"| {s.unit.location} | {s.kind.label} | {s.strategy.value} "
                f"| {STATUS_LABEL.get(s.status, s.status)} | {effect} |"
            )
        lines.append("")

        # ── 需要人工处理的，放前面 ──
        attention = [s for s in self.suggestions if s.needs_attention]
        if attention:
            lines += [
                "## ⚠️ 需要你处理的地方",
                "",
                "这些段落**没能自动改好**。系统不会假装成功 —— 请人工改写：",
                "",
            ]
            for s in attention:
                lines += [
                    f"### {s.unit.location}（{s.kind.label}）",
                    "",
                    f"- **原文**：{_q(s.unit.text)}",
                    f"- **状态**：{STATUS_LABEL.get(s.status, s.status)}",
                    f"- **说明**：{s.note}",
                ]
                if s.chosen and s.chosen != s.unit.text:
                    lines.append(f"- **最接近的候选**：{_q(s.chosen)}")
                lines.append("")

        # ── 已采纳的，四栏对照 ──
        accepted = self.by_status(SuggestionStatus.ACCEPTED)
        if accepted:
            lines += [
                "## ✅ 已改写（四栏对照）",
                "",
                "| 位置 | 原文 | 建议改为 |",
                "|---|---|---|",
            ]
            for s in accepted:
                lines.append(
                    f"| {s.unit.location} | {_q(s.unit.text, 80)} | {_q(s.chosen, 80)} |"
                )
            lines += ["", "### 效果与保真依据", ""]
            for s in accepted:
                lines.append(f"- **{s.unit.location}**")
                if s.effect is not None:
                    lines.append(f"  - 效果：{s.effect.summary()}")
                if s.fidelity is not None:
                    lines.append(f"  - 保真：{s.fidelity.summary()}")
                lines.append(f"  - 策略依据：{s.note}")
            lines.append("")

        # ── 刻意不改的 ──
        skipped = [
            s
            for s in self.suggestions
            if s.status
            in (
                SuggestionStatus.SKIPPED_PROTECTED,
                SuggestionStatus.SKIPPED_CITE,
                SuggestionStatus.SKIPPED_NO_CHANGE,
            )
        ]
        if skipped:
            lines += [
                "## 🔒 刻意不改的地方（以及为什么）",
                "",
                "| 位置 | 状态 | 原因 |",
                "|---|---|---|",
            ]
            for s in skipped:
                lines.append(
                    f"| {s.unit.location} | {STATUS_LABEL.get(s.status, s.status)} "
                    f"| {s.note} |"
                )
            lines.append("")

        # ── 未定位 ──
        unlocated = self.checklist.unlocated()
        if unlocated:
            lines += [
                "## ⚠️ 报告里提到、但没能定位的条目",
                "",
                "系统**不会替你猜位置**。请自行在文档里查找：",
                "",
                "| # | 报告片段 | 原因 |",
                "|---|---|---|",
            ]
            for i, it in enumerate(unlocated, start=1):
                lines.append(f"| {i} | {_q(it.item.text, 60)} | {it.note or '—'} |")
            lines.append("")

        lines += [
            "## 怎么读这份报告",
            "",
            "| 项 | 含义 |",
            "|---|---|",
            "| **最长连续重叠** | 改写后与原句/来源一字不差的连续片段长度。越短越好 |",
            "| **n-gram 包含率** | 新句子里还有多少词组来自原句。**只看连续重叠会被「拆短句」骗过** |",
            "| **直接度量 / 代理指标** | 有来源原文时直接对照来源；没有时只能对照原句，属代理指标 |",
            "| **保真** | 数字、否定、长度、专名是否保持。**不通过一律弃用** |",
            "",
            "> 本模块**不做完整的语义等价判断**。保真校验能抓住最常见的失真"
            "（吞数字、翻转否定、丢专名），但抓不住「表述完全变了但字面都合规」的情况。",
            "> 采纳前请自己读一遍改写结果。",
            "",
        ]
        return "\n".join(lines)


def _q(text: str, limit: int = 120) -> str:
    t = " ".join((text or "").split())
    if len(t) > limit:
        t = t[:limit] + "…"
    return f"「{t}」"
