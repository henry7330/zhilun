"""F2 · 提交前检查。

层级定位::

    src/copilot
    └── delivery
        └── F2 提交前检查      ← 本包
            ├── manifest.py  ① 检查清单（声明式）
            ├── checks.py    ② 占位符 ③ 敏感信息 ④ 匿名性 ⑤（复用 E1 结构）+ 字数
            └── gate.py      门禁判定与报告

回答的问题：**"带着这份稿子能不能交？"**

与 E1 的关系：**结构类检查直接复用 E1 的规则**，只按"能不能交"重新定级
（见 `manifest.SEVERITY_REGRADE`）。检测只做一遍，判决按用途各定各的。

典型用法::

    from copilot.delivery.prcheck import check_submission

    report = check_submission("thesis.md", anonymity_required=True)
    print(report.summary_line())      # ✅/❌ + 阻断项数量
    if not report.can_submit:
        for issue in report.blockers():
            print(issue.code, issue.location, issue.message)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from copilot.delivery.prcheck.checks import (
    ANONYMITY_PATTERNS,
    PLACEHOLDER_PATTERNS,
    SENSITIVE_PATTERNS,
    check_anonymity,
    check_placeholders,
    check_sensitive,
    check_structure,
    check_word_count,
)
from copilot.delivery.prcheck.gate import PreSubmissionReport, build_report
from copilot.delivery.prcheck.manifest import (
    SEVERITY_REGRADE,
    CheckSpec,
    Manifest,
    default_manifest,
)
from copilot.quality.format.docmodel import DocModel, parse_markdown
from copilot.quality.format.report import Issue, Severity
from copilot.quality.format.template import Template, default_template

__all__ = [
    "check_submission",
    "check_document",
    "PreSubmissionReport",
    "Manifest",
    "CheckSpec",
    "default_manifest",
    "SEVERITY_REGRADE",
    # 单项检查
    "check_placeholders",
    "check_sensitive",
    "check_anonymity",
    "check_structure",
    "check_word_count",
    # 模式表（便于用户按自己的情况扩展）
    "PLACEHOLDER_PATTERNS",
    "SENSITIVE_PATTERNS",
    "ANONYMITY_PATTERNS",
]


def _read(source: str | Path) -> tuple[str, str]:
    text = str(source)
    if not text or "\n" in text or len(text) > 260:
        return text, "<inline>"
    p = Path(text)
    try:
        if p.is_file():
            return p.read_text(encoding="utf-8"), str(p)
    except OSError:
        pass
    return text, "<inline>"


def check_document(
    doc: DocModel,
    *,
    manifest: Manifest | None = None,
    template: Template | None = None,
    doc_path: str | None = None,
) -> PreSubmissionReport:
    """对一个已解析的文档做提交前检查。"""
    mf = manifest or default_manifest()
    tpl = template or default_template()

    enabled = {c.code for c in mf.enabled()}
    issues: list[Issue] = []

    if "PRE001" in enabled:
        issues.extend(check_placeholders(doc))
    if "PRE002" in enabled:
        issues.extend(check_sensitive(doc))
    if "PRE003" in enabled and mf.anonymity_required:
        issues.extend(check_anonymity(doc))
    if "PRE004" in enabled:
        issues.extend(check_structure(doc, tpl))
    if "PRE005" in enabled:
        issues.extend(check_word_count(doc, mf.min_body_words))

    return build_report(doc_path or str(doc.path), mf, issues)


def check_submission(
    source: str | Path,
    *,
    manifest: Manifest | None = None,
    template: Template | None = None,
    anonymity_required: bool = False,
    min_body_words: int | None = None,
) -> PreSubmissionReport:
    """对一份草稿做提交前检查。

    ``source`` 可以是路径或直接的文本（与其它模块一致，
    让写作 Agent 不必为检查一段生成结果去创建临时文件）。
    """
    text, label = _read(source)
    doc = parse_markdown(text, path=label)
    mf = manifest or default_manifest(
        anonymity_required=anonymity_required,
        min_body_words=min_body_words,
    )
    return check_document(doc, manifest=mf, template=template, doc_path=label)
