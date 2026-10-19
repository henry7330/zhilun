"""E1 格式校验 Agent。

层级定位（本项目统一的组织方式）::

    src/copilot
    └── quality
        └── E1 格式校验     ← 本包
            ├── ① docmodel.py   文档模型（防腐层）
            ├── ② template.py   模板配置（要求即数据）
            ├── ③ rules/structure.py   结构校验
            ├── ④ rules/reference.py   参考文献格式
            ├── ⑤ rules/figures.py     图表与交叉引用
            └── ⑥ report.py     问题报告（统一词汇 + 汇总）

典型用法::

    from copilot.quality.format import check_format, default_template

    report = check_format("draft.md", default_template())
    print(report.summary_line())
    Path("reports/format_report.md").write_text(report.to_markdown(), encoding="utf-8")
"""

from copilot.quality.format.checker import (
    DEFAULT_RULES,
    Rule,
    check_document,
    check_format,
)
from copilot.quality.format.docmodel import (
    Caption,
    CrossRef,
    DocModel,
    Reference,
    Section,
    count_words,
    parse_markdown,
)
from copilot.quality.format.report import FormatReport, Issue, Severity
from copilot.quality.format.template import (
    SUPPORTED_REF_STYLES,
    Template,
    builtin_templates,
    default_template,
    load_template,
    save_template,
)

__all__ = [
    # 入口
    "check_format",
    "check_document",
    "DEFAULT_RULES",
    "Rule",
    # 文档模型
    "DocModel",
    "Section",
    "Caption",
    "CrossRef",
    "Reference",
    "parse_markdown",
    "count_words",
    # 模板
    "Template",
    "default_template",
    "load_template",
    "save_template",
    "builtin_templates",
    "SUPPORTED_REF_STYLES",
    # 报告
    "FormatReport",
    "Issue",
    "Severity",
]
