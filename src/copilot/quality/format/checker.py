"""E1 格式校验 Agent 的编排入口。

把「文档模型 + 模板 + 三类规则 + 报告」串成一次调用。

为什么规则要放在一个列表里跑
----------------------------
新增一类规则时，只需写一个 ``check_xxx(doc, template) -> list[Issue]``
函数并加进 :data:`DEFAULT_RULES`，其余部分（汇总、分级、排序、报告）不用动。
这样"扩展规则"的成本是**加一个函数**，而不是改五处。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from copilot.quality.format.docmodel import DocModel, parse_markdown
from copilot.quality.format.report import FormatReport, Issue
from copilot.quality.format.rules.figures import check_figures
from copilot.quality.format.rules.reference import check_references
from copilot.quality.format.rules.structure import check_structure
from copilot.quality.format.template import Template, default_template

__all__ = ["Rule", "DEFAULT_RULES", "check_format", "check_document"]

Rule = Callable[[DocModel, Template], list[Issue]]

#: 默认规则集。顺序即报告里同类问题的排列顺序。
DEFAULT_RULES: tuple[Rule, ...] = (
    check_structure,
    check_references,
    check_figures,
)


def check_document(
    doc: DocModel,
    template: Template | None = None,
    *,
    rules: Sequence[Rule] = DEFAULT_RULES,
) -> FormatReport:
    """对一个已解析的文档跑全部规则。"""
    tpl = template or default_template()
    issues: list[Issue] = []
    for rule in rules:
        issues.extend(rule(doc, tpl))
    return FormatReport(doc=doc, template=tpl, issues=issues)


def check_format(
    source: str | Path,
    template: Template | None = None,
    *,
    rules: Sequence[Rule] = DEFAULT_RULES,
) -> FormatReport:
    """校验一份草稿。

    ``source`` 可以是文件路径，也可以是**直接传入的 Markdown 文本**
    （通过 ``looks_like_path`` 判断）。支持直接传文本是为了让调用方
    不必为了检查一段生成结果而去创建临时文件 —— 写作 Agent 会频繁用到这种用法。
    """
    text_source = str(source)
    if "\n" in text_source or len(text_source) > 260:
        # 含换行或超长，一定是文本而不是路径
        doc = parse_markdown(text_source, path="<inline>")
    else:
        p = Path(text_source)
        if p.exists():
            doc = parse_markdown(p.read_text(encoding="utf-8"), path=p)
        else:
            doc = parse_markdown(text_source, path="<inline>")
    return check_document(doc, template, rules=rules)
