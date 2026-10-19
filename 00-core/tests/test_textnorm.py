"""长度保持的规范化 —— 全项目共用的文本口径。

**核心不变量**：``len(normalize(t)) == len(t)``。

这条不变量错了**不会报错**，只会让「比对用规范化文本、定位用原始偏移」
的坐标错位 —— 报告里的行号看着合理，其实指向别处。
所以这里测的是"不变量"，而不只是"几个映射对不对"。
"""

from __future__ import annotations

import pytest

from copilot.textnorm import fullwidth_map, line_of, line_starts, normalize


def test_normalize_is_length_preserving() -> None:
    samples = [
        "你好，世界。",
        "ＡＢＣ１２３",
        "多个   空格\t和\n换行",
        "引号“测试”与「直角」",
        "",
        "mixed 中英 mixed",
        "\u3000全角空格开头",
    ]
    for s in samples:
        assert len(normalize(s)) == len(s), f"长度变了：{s!r}"


def test_normalize_maps_fullwidth_ascii() -> None:
    assert normalize("ＡＢＣ") == "ABC"
    assert normalize("１２３") == "123"


def test_normalize_maps_cjk_punctuation() -> None:
    assert normalize("你好，世界。") == "你好,世界."
    assert normalize("（括号）") == "(括号)"


def test_normalize_collapses_whitespace_without_changing_length() -> None:
    assert normalize("a\t\tb") == "a  b"
    assert normalize("a\n\nb") == "a  b"
    assert normalize("\u3000") == " "


def test_normalize_keeps_unknown_chars_as_is() -> None:
    assert normalize("中文abc") == "中文abc"


def test_fullwidth_map_is_one_to_one() -> None:
    """映射表必须是 1 换 1，否则长度保持无从谈起。"""
    for src, dst in fullwidth_map().items():
        assert len(src) == 1 and len(dst) == 1, f"{src!r} -> {dst!r} 不是 1 换 1"


# ─────────────────────── 行号换算 ───────────────────────


def test_line_starts_counts_blank_lines() -> None:
    """空行也算一行 —— 手工数行号极易数错，一律用这个函数算。

    这里**不硬编码**期望值，而是验证定义：每个 ``\\n`` 之后都有一条新行。
    （我第一版硬编码的期望值就数错了。）
    """
    text = "第一行。\n\n第二行。\n第三行。"
    starts = line_starts(text)
    assert starts[0] == 0
    assert len(starts) == text.count("\n") + 1
    for s in starts[1:]:
        assert text[s - 1] == "\n"


def test_line_of_maps_offset_to_line() -> None:
    text = "第一行。\n\n第二行。\n第三行。"
    starts = line_starts(text)
    assert line_of(starts, 0) == 0          # "第一行。"
    assert line_of(starts, 5) == 1          # 空行
    assert line_of(starts, 6) == 2          # "第二行。"
    assert line_of(starts, 11) == 3         # "第三行。"


def test_line_of_clamps_negative() -> None:
    starts = line_starts("abc")
    assert line_of(starts, -5) == 0


@pytest.mark.parametrize("pos", [0, 1, 3, 5, 6, 11, 15])
def test_line_of_is_monotonic(pos: int) -> None:
    text = "第一行。\n\n第二行。\n第三行。"
    starts = line_starts(text)
    assert 0 <= line_of(starts, pos) < len(starts)
