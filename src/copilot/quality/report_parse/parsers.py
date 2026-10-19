"""E3 · 报告格式解析器 —— 把各家报告转成统一模型。

为什么用"注册表"而不是 if/else
------------------------------
报告格式会不断增加（新工具、新导出版本），而且**谁也无法预先支持所有格式**。
注册表让"新增一种格式"的成本 = 加一个解析器类并登记，
而不是去改核心逻辑里的分支。

优先支持「用户直接粘贴片段列表」
--------------------------------
PDF 报告的版式解析很脆弱：不同工具导出的 PDF 结构千差万别，还有扫描件。
而"把报告里那几段文字复制粘贴进来"是**零歧义**的。

因此本模块**优先保证纯文本粘贴这条路径可靠**，再逐步加自动解析；
而不是反过来做一个"能自动解析所有 PDF"的脆弱功能。
这与本项目一贯的取舍一致：**先保证确定的路径能用。**
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Protocol, Sequence

from copilot.quality.report_parse.models import FlaggedItem, FlagType

__all__ = [
    "ReportParser",
    "PlainTextParser",
    "JsonReportParser",
    "CsvReportParser",
    "ReportParseError",
    "parse_report",
    "available_formats",
]


class ReportParseError(ValueError):
    """报告无法解析。"""


class ReportParser(Protocol):
    name: str
    #: 形状判定有多确定。``True`` 表示"看起来就是这种格式"（如以 `[` 开头），
    #: 此时解析失败**应当直接报错**，而不是降级成纯文本。
    strict_shape: bool

    def can_parse(self, text: str) -> bool: ...

    def parse(self, text: str, *, flag_type: FlagType) -> list[FlaggedItem]: ...


def _new_id(index: int, flag_type: FlagType) -> str:
    prefix = "dup" if flag_type is FlagType.DUPLICATE else "ai"
    return f"{prefix}-{index}"


def _split_blocks(text: str) -> list[str]:
    """把纯文本拆成若干片段。

    优先按**空行**分块（一个可能跨行的片段）；
    若整段没有空行，则按**行**拆（最常见：一行一条的列表）。
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) == 1 and "\n" in blocks[0]:
        return [ln.strip() for ln in blocks[0].splitlines() if ln.strip()]
    return blocks


class PlainTextParser:
    """纯文本：一行（或一个空行分隔的块）= 一条片段。

    这是最可靠的路径 —— 用户把报告里列出的重复片段复制粘贴进来即可，
    不依赖任何版式解析。
    """

    name = "plain"
    #: 纯文本是最松的形状 —— 任何内容都能当纯文本，所以它永远是兜底
    strict_shape = False

    def can_parse(self, text: str) -> bool:
        return bool(text.strip())

    def parse(self, text: str, *, flag_type: FlagType) -> list[FlaggedItem]:
        blocks = _split_blocks(text)
        if not blocks:
            raise ReportParseError("没有可解析的片段（内容为空）")
        return [
            FlaggedItem(
                item_id=_new_id(i, flag_type),
                text=b,
                flag_type=flag_type,
            )
            for i, b in enumerate(blocks, start=1)
        ]


def _coerce_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None


class JsonReportParser:
    """JSON：``[{"text": "...", "score": 0.31, "source_title": "...", ...}]``。

    也接受外面包一层：``{"items": [...]}`` / ``{"segments": [...]}``。
    字段名做了常见别名兼容（``text``/``content``/``snippet``），
    因为各家导出的字段名并不统一。
    """

    name = "json"
    #: 以 ``[`` / ``{`` 开头**就是** JSON 的形状 —— 解析失败说明内容有问题，
    #: 应当直接报错而不是降级成纯文本（那会把整段 JSON 当成一条重复片段）
    strict_shape = True

    _TEXT_KEYS = ("text", "content", "snippet", "passage", "repeat_text")
    _SCORE_KEYS = ("score", "similarity", "rate", "ratio", "probability")
    _SOURCE_TITLE_KEYS = ("source_title", "source", "title", "origin")
    _SOURCE_TEXT_KEYS = ("source_text", "original", "matched_text")

    def can_parse(self, text: str) -> bool:
        s = text.strip()
        return s.startswith("[") or s.startswith("{")

    @staticmethod
    def _first(data: dict, keys: Sequence[str]):
        for k in keys:
            if k in data and data[k] not in (None, ""):
                return data[k]
        return None

    def parse(self, text: str, *, flag_type: FlagType) -> list[FlaggedItem]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ReportParseError(f"JSON 解析失败：{exc}") from exc

        if isinstance(data, dict):
            for key in ("items", "segments", "records", "data"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                # 单个对象也接受
                data = [data]

        if not isinstance(data, list):
            raise ReportParseError("JSON 顶层必须是数组，或含 items/segments 的对象")

        items: list[FlaggedItem] = []
        for i, entry in enumerate(data, start=1):
            if isinstance(entry, str):
                body = entry.strip()
                if not body:
                    continue
                items.append(
                    FlaggedItem(item_id=_new_id(len(items) + 1, flag_type), text=body,
                                flag_type=flag_type)
                )
                continue
            if not isinstance(entry, dict):
                raise ReportParseError(f"第 {i} 条不是对象或字符串：{type(entry).__name__}")

            body = self._first(entry, self._TEXT_KEYS)
            if not body or not str(body).strip():
                raise ReportParseError(f"第 {i} 条缺少文字内容（可用字段：{', '.join(self._TEXT_KEYS)}）")

            items.append(
                FlaggedItem(
                    item_id=_new_id(len(items) + 1, flag_type),
                    text=str(body).strip(),
                    flag_type=flag_type,
                    score=_coerce_float(self._first(entry, self._SCORE_KEYS)),
                    source_title=(
                        None
                        if self._first(entry, self._SOURCE_TITLE_KEYS) is None
                        else str(self._first(entry, self._SOURCE_TITLE_KEYS))
                    ),
                    source_text=(
                        None
                        if self._first(entry, self._SOURCE_TEXT_KEYS) is None
                        else str(self._first(entry, self._SOURCE_TEXT_KEYS))
                    ),
                    raw=dict(entry),
                )
            )

        if not items:
            raise ReportParseError("JSON 里没有有效的条目")
        return items


class CsvReportParser:
    """CSV：第一列当作片段文字，其余列按表头识别。

    没有表头时，认定为「只有一列文字」。
    """

    name = "csv"
    #: "第一行含逗号"是个很弱的信号（一句中文里也可能有逗号），
    #: 所以 CSV 解析失败时允许退回纯文本
    strict_shape = False

    def can_parse(self, text: str) -> bool:
        s = text.strip()
        return "," in s.splitlines()[0] if s.splitlines() else False

    def parse(self, text: str, *, flag_type: FlagType) -> list[FlaggedItem]:
        reader = csv.reader(io.StringIO(text))
        rows = [r for r in reader if any(c.strip() for c in r)]
        if not rows:
            raise ReportParseError("CSV 为空")

        header = [h.strip().lower() for h in rows[0]]
        has_header = any(k in header for k in ("text", "content", "snippet"))
        data_rows = rows[1:] if has_header else rows

        def col(name: str) -> int | None:
            return header.index(name) if has_header and name in header else None

        text_col = col("text") or col("content") or col("snippet") or 0
        score_col = col("score") or col("similarity") or col("rate")
        title_col = col("source_title") or col("source") or col("title")

        items: list[FlaggedItem] = []
        for row in data_rows:
            if text_col >= len(row):
                continue
            body = row[text_col].strip()
            if not body:
                continue
            items.append(
                FlaggedItem(
                    item_id=_new_id(len(items) + 1, flag_type),
                    text=body,
                    flag_type=flag_type,
                    score=(
                        _coerce_float(row[score_col])
                        if score_col is not None and score_col < len(row)
                        else None
                    ),
                    source_title=(
                        row[title_col].strip()
                        if title_col is not None
                        and title_col < len(row)
                        and row[title_col].strip()
                        else None
                    ),
                    raw={"row": row},
                )
            )

        if not items:
            raise ReportParseError("CSV 里没有有效的条目")
        return items


#: 解析器注册表。顺序即自动探测的优先级（结构化格式优先于纯文本）。
_PARSERS: tuple[ReportParser, ...] = (
    JsonReportParser(),
    CsvReportParser(),
    PlainTextParser(),
)


def available_formats() -> list[str]:
    return [p.name for p in _PARSERS]


def parse_report(
    text: str,
    *,
    flag_type: FlagType,
    fmt: str = "auto",
) -> tuple[list[FlaggedItem], str]:
    """解析报告，返回 ``(条目列表, 实际使用的格式名)``。

    ``fmt="auto"`` 时按注册表顺序探测；显式指定则只用那一个解析器。

    **为什么返回实际格式名**：自动探测可能选错（例如一段含逗号的纯文本
    会被当成 CSV）。把实际用了什么告诉调用方，出了问题才能查。
    这也是"不隐藏内部决策"的一贯做法。
    """
    if fmt != "auto":
        for parser in _PARSERS:
            if parser.name == fmt:
                return parser.parse(text, flag_type=flag_type), parser.name
        raise ReportParseError(
            f"未知格式 {fmt!r}；可用：{', '.join(available_formats())}"
        )

    errors: list[str] = []
    for parser in _PARSERS:
        if not parser.can_parse(text):
            continue
        try:
            return parser.parse(text, flag_type=flag_type), parser.name
        except ReportParseError as exc:
            errors.append(f"{parser.name}: {exc}")
            if getattr(parser, "strict_shape", False):
                # 形状已经明确（例如就是 JSON），解析失败是**内容有问题**。
                # 这时降级成纯文本会把整段 JSON 当成一条重复片段，
                # 用户拿到的定位结果会离奇且难查 —— 不如直接报错。
                raise ReportParseError(
                    f"内容看起来是 {parser.name} 格式，但解析失败：{exc}"
                ) from exc
            continue

    raise ReportParseError(
        "报告无法解析。尝试过的格式：\n  " + "\n  ".join(errors or ["（没有匹配的解析器）"])
    )
