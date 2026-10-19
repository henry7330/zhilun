"""E6 · 审稿意见报告。

与 E1 共用同一套 ``Issue`` / ``Severity`` 词汇 ——
所以两份报告可以**直接合并**成一份质检报告，不需要做任何格式转换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from copilot.quality.format.docmodel import DocModel, parse_markdown
from copilot.quality.format.report import Issue, Severity
from copilot.quality.format.template import Template, default_template
from copilot.quality.review.checks import (
    REVIEW_RULES,
    check_figure_refs,
    check_keywords,
    check_section_length,
    check_terms,
)

__all__ = ["ReviewReport", "review_document", "review_text"]


@dataclass
class ReviewReport:
    """一份审稿意见。"""

    doc: DocModel
    issues: list[Issue] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.issues.sort(
            key=lambda i: (i.severity.rank, i.line if i.line is not None else 10**9)
        )

    def by_severity(self, severity: Severity) -> list[Issue]:
        return [i for i in self.issues if i.severity is severity]

    def counts(self) -> dict[str, int]:
        return {
            "error": len(self.by_severity(Severity.ERROR)),
            "warning": len(self.by_severity(Severity.WARNING)),
            "info": len(self.by_severity(Severity.INFO)),
            "total": len(self.issues),
        }

    def codes(self) -> list[str]:
        seen: list[str] = []
        for i in self.issues:
            if i.code not in seen:
                seen.append(i.code)
        return seen

    def summary_line(self) -> str:
        c = self.counts()
        return (
            f"审稿意见：{c['error']} 个错误 / {c['warning']} 个警告 / "
            f"{c['info']} 个提示（共 {c['total']} 项）"
        )

    def to_dict(self) -> dict:
        return {
            "file": str(self.doc.path),
            "counts": self.counts(),
            "issues": [i.to_dict() for i in self.issues],
        }

    def to_markdown(self) -> str:
        c = self.counts()
        lines = [
            "# 审稿意见",
            "",
            f"- 文档：`{self.doc.path}`",
            f"- 规模：{self.doc.body_words} 字 / "
            f"{len(self.doc.flat_sections())} 个章节 / "
            f"{len(self.doc.captions)} 个图表",
            "",
            "## 结论",
            "",
            f"错误 **{c['error']}** / 警告 **{c['warning']}** / 提示 **{c['info']}**",
            "",
        ]

        if not self.issues:
            lines += ["没有发现可判定的问题。", ""]
            return "\n".join(lines)

        lines += [
            "| 严重级 | 规则码 | 位置 | 问题 | 建议 |",
            "|---|---|---|---|---|",
        ]
        for i in self.issues:
            lines.append(
                f"| {i.severity.label} | `{i.code}` | {i.location} "
                f"| {i.message.replace('|', '/')} | {(i.fix or '—').replace('|', '/')} |"
            )

        lines += [
            "",
            "## 本模块**不做**的事",
            "",
            "E6 只做**可判定**的检查：术语一致性、关键词覆盖、章节篇幅、图表引用。",
            "",
            "**「论证链是否断裂」不在其中** —— 那需要跨句语义理解，属模型范畴。",
            "把没做的事写清楚，比假装做了要好：一份声称"
            "「已检查论证逻辑」实则只查了术语的报告，会让用户误以为可以放心提交。",
            "",
            "## 规则码",
            "",
            "| 码 | 含义 |",
            "|---|---|",
            "| `REV001` | 同一概念出现多种叫法 |",
            "| `REV002` | 摘要关键词未在正文出现 |",
            "| `REV003` | 一级章节过短 |",
            "| `FIG00x` | 图表引用问题（与 E1 共用同一套规则码） |",
            "",
        ]
        return "\n".join(lines)


def review_document(
    doc: DocModel,
    template: Template | None = None,
    *,
    include_figures: bool = True,
    min_section_chars: int | None = None,
) -> ReviewReport:
    """对一个已解析的文档做审稿检查。"""
    tpl = template or default_template()
    issues: list[Issue] = []
    issues.extend(check_terms(doc))
    issues.extend(check_keywords(doc))
    if min_section_chars is None:
        issues.extend(check_section_length(doc, template=tpl))
    else:
        issues.extend(check_section_length(doc, min_chars=min_section_chars, template=tpl))
    if include_figures:
        issues.extend(check_figure_refs(doc, tpl))
    return ReviewReport(doc=doc, issues=issues)


def review_text(
    text: str,
    template: Template | None = None,
    *,
    include_figures: bool = True,
    min_section_chars: int | None = None,
) -> ReviewReport:
    """直接对文本做审稿检查（不落地成文件）。"""
    return review_document(
        parse_markdown(text, path="<inline>"),
        template,
        include_figures=include_figures,
        min_section_chars=min_section_chars,
    )
