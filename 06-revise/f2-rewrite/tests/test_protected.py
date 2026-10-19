"""E4 · 受保护片段识别。

降重的第一原则：**不能改的东西绝不能改**。
引用、引号内的直接引用、参考文献标注所在句 —— 改了会变成学术不端，
比"重复率高"严重得多。

本文件原本是 E3 查重的「引用感知过滤」（把规范引用从重复里豁免掉），
用途反过来用：**从改写范围里排除掉**。逻辑是同一个逻辑。
"""

from __future__ import annotations

from copilot.quality.rewrite.protected import (
    citation_status,
    enclosing_sentence,
    enclosing_sentence_bounds,
    has_citation_mark,
    quoted_ranges,
)

UNIQUE = "该方法在三个公开数据集上取得了显著效果，准确率提升了8.3%。"


# ─────────────────────── 引用标注识别 ───────────────────────


def test_has_citation_mark_variants() -> None:
    assert has_citation_mark("如文献[1]所述")
    assert has_citation_mark("见[1-3]")
    assert has_citation_mark("见[1,2]")
    assert has_citation_mark("（张三，2020）")
    assert has_citation_mark("(Smith et al., 2019)")
    assert not has_citation_mark("普通句子，没有引用。")


def test_citation_mark_does_not_match_reference_type_marker() -> None:
    """``[J]`` 是文献类型标识，不是引用标注 —— 误判会让保护范围出错。"""
    assert not has_citation_mark("一种方法[J]. 计算机学报, 2020.")


# ─────────────────────── 引号配对 ───────────────────────


def test_quoted_ranges_finds_cjk_quotes() -> None:
    ranges = quoted_ranges("他说“你好”，又说「再见」。")
    assert len(ranges) == 2


def test_straight_quotes_are_paired_by_order() -> None:
    ranges = quoted_ranges('他说 "hello" 和 "world"。')
    assert len(ranges) == 2


def test_book_title_mark_is_not_a_quote() -> None:
    """★ ``《》`` 是书名号，标记作品名，**不是**引号。

    把它当引号会放过"整段抄一本书" —— 那是最该被改写的部分。
    """
    assert quoted_ranges("参见《深度学习》一书。") == ()


# ─────────────────────── 句边界 ───────────────────────


def test_enclosing_sentence_bounds() -> None:
    text = "第一句。第二句在这里。第三句。"
    left, right = enclosing_sentence_bounds(text, 4, 6)
    assert text[left:right] == "第二句在这里"
    assert enclosing_sentence(text, 4, 6) == "第二句在这里"


def test_enclosing_sentence_stops_at_newline() -> None:
    text = "上一行\n这一行是要看的\n下一行"
    s = enclosing_sentence(text, text.index("这一行"), text.index("这一行") + 3)
    assert "这一行是要看的" in s
    assert "上一行" not in s


# ─────────────────────── 保护判定 ───────────────────────


def test_protected_when_inside_quotes() -> None:
    text = "原文写道“这是一段被引用的内容”，随后是作者自己的话。"
    start = text.index("这是一段被引用")
    protected, reason = citation_status(text, start, start + 8)
    assert protected and "引号" in reason


def test_protected_by_marker_in_sentence() -> None:
    text = "张三等人提出了图神经网络方法[1]，取得了较好效果。"
    start = text.index("张三")
    protected, reason = citation_status(text, start, start + 20)
    assert protected and "[1]" in reason


def test_protected_when_marker_follows_the_period() -> None:
    """★ 引用标注写在**句号之后**（``……效果。[1]``）也要能识别。

    「中文论文的引用标注通常写在句号前」是事实，
    但**"通常"不等于"总是"** —— 按"通常"实现会漏掉真实情况，
    于是该保护的句子被改写，改写后引用关系就断了。
    """
    text = f"前面的内容。{UNIQUE}[1]\n后面的内容。"
    start = text.index("该方法")
    protected, _ = citation_status(text, start, start + 10)
    assert protected, "句号后的引用标注也应视为保护理由"


def test_not_protected_for_plain_sentence() -> None:
    text = "我自己提出了一个新的方法，并且在实验中验证了它。"
    protected, _ = citation_status(text, 0, 10)
    assert not protected


def test_not_protected_when_no_citation_nearby() -> None:
    text = "第一句原创。第二句也是原创内容。第三句同样原创。"
    protected, reason = citation_status(text, 6, 12)
    assert not protected and reason == ""
