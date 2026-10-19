"""E3 · 编排入口 —— 报告文本 + 文档 → 问题清单。

一次调用完成：解析报告 → 建文档索引 → 切分单元 → 三级定位 → 汇成清单。

为什么"解析失败"和"定位失败"要分开报
--------------------------------------
两者对用户的意义完全不同：

- **解析失败** —— 报告本身读不懂（格式不认识）。用户该做的是**换一种粘贴方式**；
- **定位失败** —— 报告读懂了，但在文档里找不到对应位置。用户该做的是**自己核对**。

混成一句"处理失败"，用户不知道该改什么。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from copilot.quality.report_parse.locate import (
    DEFAULT_FUZZY_MIN_RATIO,
    DEFAULT_FUZZY_MIN_RUN,
    build_index,
    locate_all,
)
from copilot.quality.report_parse.models import (
    Checklist,
    FlagType,
)
from copilot.quality.report_parse.parsers import ReportParseError, parse_report
from copilot.quality.report_parse.segments import segment_document

__all__ = ["build_checklist", "check_from_report", "read_document"]


def read_document(document: str | Path) -> tuple[str, str]:
    """读入文档，返回 ``(文本, 标签)``。

    传文本而不是路径是为了让写作 Agent 不必为了检查一段生成结果
    去创建临时文件 —— 与 E1/E2 的约定一致。

    ★ 用 ``is_file()`` 而不是 ``exists()``：空字符串会被 ``Path("")``
    解析成当前目录 ``.``，``exists()`` 对它返回 True，于是会去"读目录"
    并抛 PermissionError。``is_file()`` 天然排除了这种情况。
    """
    text = str(document)
    if not text or "\n" in text or len(text) > 260:
        return text, "<inline>"

    path = Path(text)
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8"), str(path)
    except OSError:
        # 路径不可读（权限/超长/非法字符）—— 当成文本处理，不阻断
        pass
    return text, "<inline>"


def build_checklist(
    document: str | Path,
    report: str,
    *,
    flag_type: FlagType = FlagType.DUPLICATE,
    fmt: str = "auto",
    fuzzy_min_ratio: float = DEFAULT_FUZZY_MIN_RATIO,
    fuzzy_min_run: int = DEFAULT_FUZZY_MIN_RUN,
) -> Checklist:
    """从文档 + 报告文本构造问题清单。"""
    doc_text, label = read_document(document)

    items, used_format = parse_report(report, flag_type=flag_type, fmt=fmt)
    # 把实际使用的解析格式记在每条上 —— 自动探测可能选错，出了问题要能查
    for it in items:
        it.raw.setdefault("_parsed_as", used_format)

    index = build_index(doc_text)
    units = segment_document(doc_text)
    located = locate_all(
        items,
        index,
        units,
        fuzzy_min_ratio=fuzzy_min_ratio,
        fuzzy_min_run=int(fuzzy_min_run),
    )
    return Checklist(doc_path=label, doc_text=doc_text, items=located)


def check_from_report(
    document: str | Path,
    report: str,
    *,
    flag_types: Sequence[FlagType] = (FlagType.DUPLICATE, FlagType.AI),
    fmt: str = "auto",
) -> list[Checklist]:
    """对同一份文档解析多份报告（例如查重报告 + AI 检测报告）。

    返回按类型分列的清单 —— 因为它们的改写策略不同
    （E4 降重瞄准"与出处重合"，E5 降AI 瞄准"风格特征"）。
    """
    return [
        build_checklist(document, report, flag_type=ft, fmt=fmt) for ft in flag_types
    ]


__all__.append("ReportParseError")
