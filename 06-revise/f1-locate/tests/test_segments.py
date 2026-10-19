"""E3 · 文档切分（用于把外部报告片段定位回来）。

要点：切出的单元必须**偏移自洽** —— 区间与文本对不上，
后面"定位到第 N 行"就是错的。
"""

from __future__ import annotations

import pytest

from copilot.quality.report_parse.segments import (
    DocUnit,
    segment_document,
)
from copilot.textnorm import line_starts


def test_units_point_into_original_text() -> None:
    text = "第一句话说得比较长一些。第二句话也不短。"
    units = segment_document(text)
    assert units
    for u in units:
        assert text[u.char_start : u.char_end] == u.raw_text


def test_unit_rejects_offset_mismatch() -> None:
    """区间跨度与原文长度不符 —— 必须立刻报错。"""
    with pytest.raises(ValueError, match="偏移跨度"):
        DocUnit(
            unit_id="u1", text="ab", raw_text="abc", char_start=0, char_end=2, line=0
        )


def test_unit_rejects_non_length_preserving_text() -> None:
    """★ 规范化改变长度必须**立刻报错**。

    这条不变量一旦破了不会报错，只会让定位静默错位 ——
    报告里的行号看着合理，其实指向别处。
    """
    with pytest.raises(ValueError, match="长度保持"):
        DocUnit(
            unit_id="u1", text="ab", raw_text="abc", char_start=0, char_end=3, line=0
        )


def test_short_sentences_are_merged() -> None:
    """过短的句子会被合并 —— 否则"因此。"这类碎片无法承载一个重复片段。"""
    text = "短。很短。这一句话稍微长一点用来凑够阈值。"
    units = segment_document(text, min_chars=12)
    assert len(units) < 3


def test_long_unit_does_not_exceed_max_chars() -> None:
    text = "。".join(["这是一句长度中等的话"] * 20) + "。"
    units = segment_document(text, min_chars=12, max_chars=60)
    assert all(len(u.raw_text) <= 90 for u in units), [len(u.raw_text) for u in units]


def test_line_numbers_point_at_the_right_line() -> None:
    """★ 报出的行号，就是 ``char_start`` 所在行。

    不硬编码行号：空行也算一行，手数极易数错。
    """
    text = "第一行。\n\n第二行内容在这里。\n第三行。"
    starts = line_starts(text)
    units = segment_document(text)
    assert units
    for u in units:
        assert starts[u.line] <= u.char_start
        if u.line + 1 < len(starts):
            assert u.char_start < starts[u.line + 1], (
                f"行号 {u.line} 不是 char_start={u.char_start} 所在行"
            )


def test_empty_text_returns_nothing() -> None:
    assert segment_document("") == []


def test_unit_ids_are_unique_and_ordered() -> None:
    text = "第一段内容在这里。第二段内容也很长一些。第三段同样有内容。"
    units = segment_document(text)
    ids = [u.unit_id for u in units]
    assert len(ids) == len(set(ids))
    starts = [u.char_start for u in units]
    assert starts == sorted(starts)
