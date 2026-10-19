"""E5 · 降AI 报告。

报告里**只出现特征的数值与变化**，不出现"这段是/不是 AI 写的"这类断言 ——
检测本身不可靠，报告不能被误用为依据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from copilot.quality.deai.pipeline import DeaiStatus, DeaiSuggestion
from copilot.quality.report_parse.models import Checklist

__all__ = ["DeaiReport", "STATUS_LABEL"]

STATUS_LABEL: dict[str, str] = {
    DeaiStatus.ACCEPTED: "✅ 已改写",
    DeaiStatus.SKIPPED_PROTECTED: "🔒 引用豁免",
    DeaiStatus.NO_DIAGNOSIS: "⚪ 无需改写",
    DeaiStatus.NO_CANDIDATE: "⚠️ 无候选",
    DeaiStatus.REJECTED_FIDELITY: "❌ 保真未通过",
    DeaiStatus.REJECTED_STYLE: "❌ 风格未改善",
}


@dataclass
class DeaiReport:
    """一次降 AI 的完整报告。"""

    doc_path: str
    checklist: Checklist
    suggestions: list[DeaiSuggestion] = field(default_factory=list)
    rewriter_name: str = ""

    def by_status(self, status: str) -> list[DeaiSuggestion]:
        return [s for s in self.suggestions if s.status == status]

    def counts(self) -> dict[str, int]:
        return {
            "units": len(self.suggestions),
            "accepted": len(self.by_status(DeaiStatus.ACCEPTED)),
            "skipped": sum(
                1
                for s in self.suggestions
                if s.status in (DeaiStatus.SKIPPED_PROTECTED, DeaiStatus.NO_DIAGNOSIS)
            ),
            "needs_attention": sum(1 for s in self.suggestions if s.needs_attention),
            "unlocated": len(self.checklist.unlocated()),
        }

    def summary_line(self) -> str:
        c = self.counts()
        return (
            f"改写 {c['accepted']} 处 / 无需或未改 {c['skipped']} 处 / "
            f"需人工处理 {c['needs_attention']} 处"
            f"（另有 {c['unlocated']} 处告警条目未能定位）"
        )

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
            "# 降 AI 报告",
            "",
            f"- 文档：`{self.doc_path}`",
            f"- 改写器：`{self.rewriter_name}`",
            f"- 输入：{self.checklist.summary_line()}",
            "",
            "> **本报告不做作者身份判断。** 它报告的是"
            "可测量的风格特征变化，以及外部检测工具指出的片段。",
            "> 「这段文字是否由 AI 生成」在任何工具上都不可靠，本系统不对此下结论。",
            "",
            "## 结论",
            "",
            f"采纳改写 **{c['accepted']}** 处 / "
            f"无需或未改 **{c['skipped']}** 处 / "
            f"**需人工处理 {c['needs_attention']}** 处",
            "",
        ]

        if not self.suggestions:
            lines += ["没有需要处理的内容。", ""]
            return "\n".join(lines)

        lines += [
            "## 一览",
            "",
            "| 位置 | 诊断 | 策略 | 状态 |",
            "|---|---|---|---|",
        ]
        for s in self.suggestions:
            diag = "、".join(d.label for d in s.diagnoses) or "—"
            strat = (
                "、".join(st.value for st in s.instruction.strategies)
                if s.instruction
                else "—"
            )
            lines.append(
                f"| {s.unit.location} | {diag} | {strat} "
                f"| {STATUS_LABEL.get(s.status, s.status)} |"
            )
        lines.append("")

        attention = [s for s in self.suggestions if s.needs_attention]
        if attention:
            lines += [
                "## ⚠️ 需要你处理的地方",
                "",
                "| 位置 | 原文 | 状态 | 说明 |",
                "|---|---|---|---|",
            ]
            for s in attention:
                lines.append(
                    f"| {s.unit.location} | {_q(s.unit.text, 50)} "
                    f"| {STATUS_LABEL.get(s.status, s.status)} | {s.note} |"
                )
            lines.append("")

        accepted = self.by_status(DeaiStatus.ACCEPTED)
        if accepted:
            lines += [
                "## ✅ 已改写（含特征前后对比）",
                "",
            ]
            for s in accepted:
                lines += [
                    f"### {s.unit.location}",
                    "",
                    f"- **原文**：{_q(s.unit.text)}",
                    f"- **改写**：{_q(s.chosen)}",
                    f"- **诊断**：{'；'.join(str(d) for d in s.diagnoses)}",
                    f"- **策略**："
                    + "；".join(
                        st.label for st in (s.instruction.strategies if s.instruction else ())
                    ),
                ]
                if s.verification is not None:
                    lines.append("- **特征变化**：")
                    for ch in s.verification.changes:
                        lines.append(f"  - {ch.describe()}")
                    if s.verification.fidelity is not None:
                        lines.append(f"- **保真**：{s.verification.fidelity.summary()}")
                lines += ["", "| 特征 | 改前 | 改后 |", "|---|---:|---:|"]
                if s.verification is not None:
                    for ch in s.verification.changes:
                        lines.append(
                            f"| {ch.feature} | {ch.before:.3f} | {ch.after:.3f} |"
                        )
                lines.append("")

        no_diag = self.by_status(DeaiStatus.NO_DIAGNOSIS)
        if no_diag:
            lines += [
                "## ⚪ 无需改写的片段",
                "",
                "这些片段被外部报告标出，但**风格特征都正常** —— ",
                "系统不会为了「降 AI」去硬改正常文字（那只会把内容改坏）：",
                "",
                "| 位置 | 片段 |",
                "|---|---|",
            ]
            for s in no_diag:
                lines.append(f"| {s.unit.location} | {_q(s.unit.text, 60)} |")
            lines.append("")

        lines += [
            "## 特征说明",
            "",
            "| 特征 | 含义 | 理想方向 |",
            "|---|---|---|",
            "| `sentence_length_cv` | 句长变异系数。**越小说明句式越均匀** | 越大越自然 |",
            "| `transition_density` | 每句套话连接词数（此外/然而/综上所述…） | 越小越好 |",
            "| `specificity` | 每百字的具体信息量（数字/专名） | 越大越好 |",
            "| `parallelism` | 每千字排比框架数 | 越小越好 |",
            "| `lexical_diversity` | 词汇丰富度（二字组 TTR） | 参考项 |",
            "",
            "> **保真是否决条件**：改写若不保真，即使风格改善明显也一律弃用。",
            "> 风格特征变好**不等于**「检测工具会给低分」—— 那是外部工具的事，本系统不预测。",
            "",
        ]
        return "\n".join(lines)


def _q(text: str, limit: int = 120) -> str:
    t = " ".join((text or "").split())
    if len(t) > limit:
        t = t[:limit] + "…"
    return f"「{t}」"
