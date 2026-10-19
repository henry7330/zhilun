"""E3 · 外部检测报告解析与定位。

层级定位::

    src/copilot
    └── quality
        └── E3 报告解析       ← 本包
            ├── ① models.py     统一问题模型
            ├── ② parsers.py    报告格式解析器（注册表）
            ├── ③ locate.py     片段定位（三级策略）★ 核心
            ├── ④ checklist.py  问题清单与编排
            └── segments.py     文档切分（定位用的单元）

**本模块不做检测。** 查重与 AI 检测由外部工具完成，
用户把它们的报告拿进来，本模块负责：

1. 解析报告 → 统一的问题条目（哪段文字、什么类型、报告给的分数）
2. 把报告里的片段**定位回用户文档**（第几行到第几行）
3. 输出问题清单，供 E4 降重 / E5 降AI 使用

典型用法::

    from copilot.quality.report_parse import build_checklist, FlagType

    checklist = build_checklist(thesis_path, report_text,
                                flag_type=FlagType.DUPLICATE)
    print(checklist.summary_line())
    for it in checklist.needing_rewrite():
        print(it.location, it.item.text[:40])

设计文档与讲解见模块文件夹的 `README.md`。
"""

from copilot.quality.report_parse.checklist import (
    ReportParseError,
    build_checklist,
    check_from_report,
    read_document,
)
from copilot.quality.report_parse.locate import (
    DEFAULT_FUZZY_MIN_RATIO,
    DEFAULT_FUZZY_MIN_RUN,
    DocIndex,
    build_index,
    locate_all,
    locate_item,
)
from copilot.quality.report_parse.models import (
    Checklist,
    FlaggedItem,
    FlagType,
    LocatedItem,
    LocateMethod,
)
from copilot.quality.report_parse.parsers import (
    CsvReportParser,
    JsonReportParser,
    PlainTextParser,
    ReportParser,
    available_formats,
    parse_report,
)
from copilot.quality.report_parse.segments import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    DocUnit,
    segment_document,
)

__all__ = [
    # 入口
    "build_checklist",
    "check_from_report",
    "parse_report",
    "available_formats",
    "read_document",
    # 模型
    "FlaggedItem",
    "FlagType",
    "LocatedItem",
    "LocateMethod",
    "Checklist",
    # 定位
    "locate_item",
    "locate_all",
    "DocIndex",
    "build_index",
    "DEFAULT_FUZZY_MIN_RATIO",
    "DEFAULT_FUZZY_MIN_RUN",
    # 解析器
    "ReportParser",
    "PlainTextParser",
    "JsonReportParser",
    "CsvReportParser",
    "ReportParseError",
    # 分块
    "DocUnit",
    "segment_document",
    "DEFAULT_MIN_CHARS",
    "DEFAULT_MAX_CHARS",
]
