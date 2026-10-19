"""E6 · 审稿挑错。

层级定位::

    src/copilot
    └── quality
        └── E6 审稿挑错       ← 本包
            ├── checks.py  三类可判定检查
            │   ├── REV001 术语一致性
            │   ├── REV002 摘要关键词覆盖
            │   ├── REV003 章节篇幅
            │   └── FIG00x 图表引用（复用 E1）
            └── report.py  审稿意见报告

关注的是**结构与内容一致性**，而不是格式 —— 与 E4/E5 的"外部报告驱动"不同，
E6 是**系统自己看整篇文档**。

**复用 E1**：文档解析、Issue/Severity 词汇、图表检查规则全部来自 E1。
判定逻辑只有一份实现。

只做**可判定**的部分：术语、关键词、篇幅、图表引用。
"论证链是否断裂"需要跨句语义理解，属模型范畴，本模块**不假装能做**。

典型用法::

    from copilot.quality.review import review_text

    report = review_text(open("thesis.md").read())
    print(report.summary_line())
"""

from copilot.quality.review.checks import (
    KEYWORD_LINE,
    REVIEW_RULES,
    SYNONYM_GROUPS,
    check_figure_refs,
    check_keywords,
    check_section_length,
    check_terms,
)
from copilot.quality.review.report import ReviewReport, review_document, review_text

__all__ = [
    "ReviewReport",
    "review_document",
    "review_text",
    # 规则
    "check_terms",
    "check_keywords",
    "check_section_length",
    "check_figure_refs",
    "REVIEW_RULES",
    "SYNONYM_GROUPS",
    "KEYWORD_LINE",
]
