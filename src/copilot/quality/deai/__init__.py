"""E5 · 降 AI（根据外部 AI 检测报告改写）。

层级定位::

    src/copilot
    └── quality
        └── E5 降AI          ← 本包
            ├── features.py  风格特征提取（六个可测特征）
            ├── diagnose.py  异常定位（★ 先诊断再改写）
            ├── strategy.py  诊断 → 改写策略
            ├── verify.py    风格 + 保真双校验 ★
            ├── pipeline.py  诊断→策略→改写→双校验→重试
            └── report.py    特征前后对比报告

**本模块不做 AI 检测。** 输入是 E3 解析定位好的告警清单，
输出是改写建议 + 特征变化 + 保真结论。

与 E4 的关系：**共用改写框架**（切片、受保护片段、改写器接口、重试逻辑），
只有"策略"与"校验目标"不同。判断标准：
**如果一段代码在 E4 和 E5 里长得一样，它就该在共用部分。**

典型用法::

    from copilot.quality.report_parse import build_checklist, FlagType
    from copilot.quality.deai import run_deai
    from copilot.quality.rewrite import StubRewriter

    checklist = build_checklist(thesis, report_text, flag_type=FlagType.AI)
    report = run_deai(checklist, rewriter=StubRewriter())
    print(report.summary_line())

设计文档与讲解见模块文件夹的 `README.md`。
"""

from __future__ import annotations

from copilot.quality.deai.diagnose import (
    CV_LOW,
    PARALLELISM_HIGH,
    SPECIFICITY_LOW,
    TRANSITION_HIGH,
    Diagnosis,
    diagnose,
)
from copilot.quality.deai.features import (
    PARALLEL_PATTERNS,
    TRANSITIONS,
    StyleProfile,
    profile,
)
from copilot.quality.deai.pipeline import (
    DeaiStatus,
    DeaiSuggestion,
    deai_checklist,
    deai_item,
    deai_unit,
)
from copilot.quality.deai.report import STATUS_LABEL, DeaiReport
from copilot.quality.deai.strategy import (
    DeaiRequest,
    StyleInstruction,
    StyleStrategy,
    choose_instruction,
)
from copilot.quality.deai.verify import (
    FEATURE_DIRECTION,
    MIN_IMPROVEMENT,
    FeatureChange,
    StyleVerification,
    verify_style,
)
from copilot.quality.report_parse.models import Checklist
from copilot.quality.rewrite.rewriter import Rewriter

__all__ = [
    "run_deai",
    # 特征
    "StyleProfile",
    "profile",
    "TRANSITIONS",
    "PARALLEL_PATTERNS",
    # 诊断
    "Diagnosis",
    "diagnose",
    "CV_LOW",
    "TRANSITION_HIGH",
    "SPECIFICITY_LOW",
    "PARALLELISM_HIGH",
    # 策略
    "StyleStrategy",
    "StyleInstruction",
    "DeaiRequest",
    "choose_instruction",
    # 校验
    "StyleVerification",
    "FeatureChange",
    "verify_style",
    "FEATURE_DIRECTION",
    "MIN_IMPROVEMENT",
    # 流水线
    "deai_unit",
    "deai_item",
    "deai_checklist",
    "DeaiSuggestion",
    "DeaiStatus",
    # 报告
    "DeaiReport",
    "STATUS_LABEL",
]


def run_deai(
    checklist: Checklist,
    *,
    rewriter: Rewriter,
    max_attempts: int = 2,
) -> DeaiReport:
    """一站式：清单 → 降 AI 建议 → 报告。"""
    suggestions = deai_checklist(
        checklist, rewriter=rewriter, max_attempts=max_attempts
    )
    return DeaiReport(
        doc_path=checklist.doc_path,
        checklist=checklist,
        suggestions=suggestions,
        rewriter_name=rewriter.name,
    )
