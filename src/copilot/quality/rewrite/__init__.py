"""E4 · 降重（根据外部查重报告改写）。

层级定位::

    src/copilot
    └── quality
        └── E4 降重           ← 本包
            ├── units.py     改写单元切分
            ├── protected.py 受保护片段识别（引用不能改）
            ├── strategy.py  改写策略选择（定义类→建议引用，数据类→不改）
            ├── rewriter.py  改写器接口（Stub / Canned / Callable）
            ├── llm.py       ★ 真实模型改写器（LlmRewriter）
            ├── fidelity.py  保真校验 ★
            ├── effect.py    降重效果度量 ★
            ├── pipeline.py  改写→度量→保真→重试
            └── report.py    四栏对照报告

**本模块不做查重。** 输入是 E3 解析定位好的问题清单，
输出是改写建议 + 效果数字 + 保真结论。

典型用法（无模型，只跑通链路）::

    from copilot.quality.report_parse import build_checklist
    from copilot.quality.rewrite import run_rewrite, StubRewriter

    checklist = build_checklist("thesis.md", report_text)
    report = run_rewrite(checklist, rewriter=StubRewriter())

典型用法（接真实模型）::

    from copilot.llm import LlmConfig
    from copilot.quality.rewrite import LlmRewriter, run_rewrite

    cfg = LlmConfig.from_env()          # 密钥只从环境变量取
    report = run_rewrite(checklist, rewriter=LlmRewriter(cfg))

⚠️ ``StubRewriter`` 不产生任何改写（只用于跑通链路）；
``LlmRewriter`` 才真的调模型。两者实现同一个 ``Rewriter`` 接口。

设计文档与讲解见模块文件夹的 `README.md`。
"""

from __future__ import annotations

from copilot.quality.report_parse.models import Checklist
from copilot.quality.rewrite.effect import (
    DEFAULT_MAX_CONTAINMENT,
    DEFAULT_MAX_RUN,
    RewriteEffect,
    measure_effect,
)
from copilot.quality.rewrite.fidelity import (
    FidelityIssue,
    FidelityResult,
    check_fidelity,
)
from copilot.quality.rewrite.llm import (
    STRATEGY_GUIDANCE,
    SYSTEM_PROMPT,
    LlmRewriter,
    build_messages,
    clean_candidate,
)
from copilot.quality.rewrite.pipeline import (
    DEFAULT_MAX_ATTEMPTS,
    RewriteSuggestion,
    SuggestionStatus,
    rewrite_checklist,
    rewrite_item,
    rewrite_unit,
)
from copilot.quality.rewrite.protected import (
    citation_status,
    enclosing_sentence,
    enclosing_sentence_bounds,
    has_citation_mark,
    quoted_ranges,
)
from copilot.quality.rewrite.report import STATUS_LABEL, RewriteReport
from copilot.quality.rewrite.rewriter import (
    CallableRewriter,
    CannedRewriter,
    RewriteRequest,
    Rewriter,
    StubRewriter,
)
from copilot.quality.rewrite.strategy import (
    ContentKind,
    RewriteInstruction,
    Strategy,
    choose_instruction,
    classify,
)
from copilot.quality.rewrite.units import RewriteUnit, split_units

__all__ = [
    # 一站式入口
    "run_rewrite",
    # 单元
    "RewriteUnit",
    "split_units",
    # 受保护
    "citation_status",
    "has_citation_mark",
    "quoted_ranges",
    "enclosing_sentence",
    "enclosing_sentence_bounds",
    # 策略
    "ContentKind",
    "Strategy",
    "RewriteInstruction",
    "classify",
    "choose_instruction",
    # 改写器
    "Rewriter",
    "RewriteRequest",
    "StubRewriter",
    "CannedRewriter",
    "CallableRewriter",
    "LlmRewriter",
    "build_messages",
    "clean_candidate",
    "SYSTEM_PROMPT",
    "STRATEGY_GUIDANCE",
    # 校验与度量
    "check_fidelity",
    "FidelityResult",
    "FidelityIssue",
    "measure_effect",
    "RewriteEffect",
    "DEFAULT_MAX_RUN",
    "DEFAULT_MAX_CONTAINMENT",
    # 流水线
    "rewrite_unit",
    "rewrite_item",
    "rewrite_checklist",
    "RewriteSuggestion",
    "SuggestionStatus",
    "DEFAULT_MAX_ATTEMPTS",
    # 报告
    "RewriteReport",
    "STATUS_LABEL",
]


def run_rewrite(
    checklist: Checklist,
    *,
    rewriter: Rewriter,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> RewriteReport:
    """一站式：清单 → 改写建议 → 报告。

    放在这里（而不是 pipeline.py）是为了避免循环导入：
    ``report.py`` 需要 ``pipeline.SuggestionStatus``，
    若 ``pipeline.py`` 反过来导入 ``report.RewriteReport`` 就成环了。
    """
    suggestions = rewrite_checklist(
        checklist, rewriter=rewriter, max_attempts=max_attempts
    )
    return RewriteReport(
        doc_path=checklist.doc_path,
        checklist=checklist,
        suggestions=suggestions,
        rewriter_name=rewriter.name,
    )
