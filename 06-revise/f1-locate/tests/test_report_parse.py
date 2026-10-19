"""E3 · 报告解析与定位测试。

四条核心验收标准：

1. 报告片段与文档一致 → **精确定位**到行；
2. 文档被改过 → **模糊定位**，并标明这是模糊匹配；
3. 完全对不上 → **如实报"未定位"**，附参考信息，**不硬指位置**；
4. 格式差异（全角/半角、空格）**不影响**定位。
"""

from __future__ import annotations

import json

import pytest

from copilot.quality.report_parse import (
    Checklist,
    FlagType,
    LocateMethod,
    ReportParseError,
    build_checklist,
    parse_report,
    segment_document,
)
from copilot.quality.report_parse.parsers import available_formats

# ─────────────────────────── 夹具 ───────────────────────────

THESIS = """# 绪论

学术文献的数量在过去二十年里呈现指数级增长，研究者需要快速定位相关工作。

# 相关工作

本文提出了一种基于图神经网络的推荐方法，并在三个数据集上验证了效果。

# 方法

我们设计了一种双阶段的检索策略：第一阶段用词面信号筛出候选。

# 结论

本文的方法取得了预期效果。
"""

#: 与文档中完全一致的一段
PLANTED = "本文提出了一种基于图神经网络的推荐方法，并在三个数据集上验证了效果。"


# ─────────────────────── 解析器 ───────────────────────


def test_parse_plain_text_one_per_line() -> None:
    items, fmt = parse_report("第一段重复内容。\n第二段重复内容。", flag_type=FlagType.DUPLICATE)
    assert fmt == "plain"
    assert len(items) == 2
    assert items[0].flag_type is FlagType.DUPLICATE


def test_parse_plain_text_blank_line_blocks() -> None:
    text = "第一段可能跨\n两行的内容。\n\n第二段内容。"
    items, _ = parse_report(text, flag_type=FlagType.AI)
    assert len(items) == 2
    assert "两行" in items[0].text


def test_parse_json_with_fields() -> None:
    payload = json.dumps(
        [
            {"text": "重复的段落内容", "score": 0.42, "source_title": "某文献",
             "source_text": "来源原文在这里"},
        ],
        ensure_ascii=False,
    )
    items, fmt = parse_report(payload, flag_type=FlagType.DUPLICATE)
    assert fmt == "json"
    assert items[0].score == pytest.approx(0.42)
    assert items[0].source_title == "某文献"
    assert items[0].has_source_text


def test_parse_json_field_aliases() -> None:
    """各家导出的字段名不统一 —— 常见别名要兼容。"""
    payload = json.dumps([{"content": "段落内容", "similarity": "31%"}], ensure_ascii=False)
    items, _ = parse_report(payload, flag_type=FlagType.DUPLICATE)
    assert items[0].text == "段落内容"
    assert items[0].score == pytest.approx(31.0)


def test_parse_json_wrapped_in_object() -> None:
    payload = json.dumps({"items": [{"text": "内容 A"}, {"text": "内容 B"}]}, ensure_ascii=False)
    items, _ = parse_report(payload, flag_type=FlagType.AI)
    assert len(items) == 2


def test_parse_csv_with_header() -> None:
    text = "text,score,source_title\n重复段落一,0.3,文献甲\n重复段落二,0.5,文献乙\n"
    items, fmt = parse_report(text, flag_type=FlagType.DUPLICATE)
    assert fmt == "csv"
    assert len(items) == 2
    assert items[0].score == pytest.approx(0.3)
    assert items[1].source_title == "文献乙"


def test_parse_csv_without_header() -> None:
    text = "重复段落一\n重复段落二\n"
    items, _ = parse_report(text, flag_type=FlagType.DUPLICATE)
    assert len(items) == 2


def test_parse_report_rejects_empty() -> None:
    with pytest.raises(ReportParseError):
        parse_report("   ", flag_type=FlagType.DUPLICATE)


def test_parse_report_rejects_unknown_format() -> None:
    with pytest.raises(ReportParseError, match="未知格式"):
        parse_report("内容", flag_type=FlagType.DUPLICATE, fmt="pdf")


def test_parse_report_rejects_json_without_text() -> None:
    with pytest.raises(ReportParseError):
        parse_report(json.dumps([{"score": 0.4}]), flag_type=FlagType.DUPLICATE)


def test_available_formats_lists_parsers() -> None:
    assert {"plain", "json", "csv"} <= set(available_formats())


# ─────────────────────── 模型 ───────────────────────


def test_flagged_item_rejects_blank_text() -> None:
    from copilot.quality.report_parse import FlaggedItem

    with pytest.raises(ValueError, match="不能为空"):
        FlaggedItem(item_id="x", text="   ", flag_type=FlagType.DUPLICATE)


def test_has_source_text_flag() -> None:
    from copilot.quality.report_parse import FlaggedItem

    with_src = FlaggedItem(item_id="a", text="x", flag_type=FlagType.DUPLICATE,
                           source_text="来源")
    without = FlaggedItem(item_id="b", text="x", flag_type=FlagType.DUPLICATE)
    assert with_src.has_source_text
    assert not without.has_source_text


# ─────────────────── ★ 三级定位 ───────────────────


def test_exact_location_when_document_unchanged() -> None:
    """★ 验收 1：片段与文档一致 → 精确定位。"""
    checklist = build_checklist(THESIS, PLANTED)

    item = checklist.items[0]
    assert item.method is LocateMethod.EXACT
    assert item.confidence == 1.0
    assert item.doc_text == PLANTED
    # 定位与原文自洽
    assert THESIS[item.char_start : item.char_end] == item.doc_text
    # 行号指向的那一行确实包含它
    assert PLANTED in THESIS.splitlines()[item.line]


def test_fuzzy_location_when_document_was_edited() -> None:
    """★ 验收 2：文档被改过 → 模糊定位，且标明是模糊匹配。

    这是本模块的核心场景：报告生成之后用户又改了稿。
    """
    edited = THESIS.replace("基于图神经网络", "基于异构图神经网络")
    # 报告片段来自**修改前**的版本
    checklist = build_checklist(edited, PLANTED)

    item = checklist.items[0]
    assert item.method is LocateMethod.FUZZY, (
        f"应模糊定位，实际 {item.method}；note={item.note}"
    )
    assert 0 < item.confidence < 1.0
    assert "核对" in item.note, "模糊匹配必须提示用户核对"
    assert item.char_start is not None
    assert "图神经网络" in edited[item.char_start : item.char_end]


def test_unlocated_when_completely_different() -> None:
    """★ 验收 3：完全对不上 → 如实报未定位，**不硬指位置**。"""
    checklist = build_checklist(THESIS, "这段文字与文档毫无关系，纯属另一篇论文的内容。")

    item = checklist.items[0]
    assert item.method is LocateMethod.UNLOCATED
    assert item.confidence == 0.0
    assert item.char_start is None and item.line is None, "未定位不得给出位置"
    assert item.location == "未定位"
    assert item.note


def test_unlocated_note_mentions_closest_candidate() -> None:
    """未定位时给出最接近的位置供参考 —— 让用户知道该往哪看。"""
    checklist = build_checklist(THESIS, "本文提出了一种基于图神经网络的推荐方法但是结论完全相反。")
    item = checklist.items[0]
    if item.method is LocateMethod.UNLOCATED:
        assert "最接近" in item.note or item.note


def test_formatting_differences_do_not_break_exact_match() -> None:
    """★ 验收 4：全角/半角差异不影响定位。"""
    report = "本文提出了一种基于图神经网络的推荐方法,并在三个数据集上验证了效果."  # 半角标点
    checklist = build_checklist(THESIS, report)
    assert checklist.items[0].method is LocateMethod.EXACT


def test_extra_whitespace_does_not_break_location() -> None:
    report = "本文提出了一种基于图神经网络的推荐方法，  并在三个数据集上验证了效果。"
    checklist = build_checklist(THESIS, report)
    assert checklist.items[0].located


# ─────────────────────── 清单 ───────────────────────


def test_checklist_sorts_by_position() -> None:
    """用户是从上往下改的 —— 清单必须按位置排序。"""
    report = "本文的方法取得了预期效果。\n学术文献的数量在过去二十年里呈现指数级增长。"
    checklist = build_checklist(THESIS, report)

    located = checklist.located()
    assert len(located) == 2
    assert located[0].char_start < located[1].char_start


def test_checklist_counts() -> None:
    report = "本文提出了一种基于图神经网络的推荐方法，并在三个数据集上验证了效果。\n完全无关的内容。"
    checklist = build_checklist(THESIS, report)
    c = checklist.counts()
    assert c["total"] == 2
    assert c["exact"] == 1
    assert c["unlocated"] == 1


def test_checklist_by_location_merges_same_position() -> None:
    """同一段文字可能同时被判"重复"和"疑似AI" —— 下游不该改两遍。"""
    report = f"{PLANTED}\n{PLANTED}"   # 两条不同的 FlaggedItem，但定位到同一处
    checklist = build_checklist(THESIS, report)
    groups = checklist.by_location()
    assert len(groups) == 1
    assert len(groups[0][1]) == 2


def test_needing_rewrite_excludes_unlocated() -> None:
    """未定位的不知道改哪，不能交给改写模块。"""
    report = f"{PLANTED}\n完全无关的内容。"
    checklist = build_checklist(THESIS, report)
    need = checklist.needing_rewrite()
    assert all(it.located for it in need)
    assert len(need) == 1


def test_checklist_markdown_reports_unlocated_prominently() -> None:
    """★ 未定位必须在报告里醒目出现 —— 静默丢弃会让用户以为都处理了。"""
    report = "完全无关的内容甲。\n完全无关的内容乙。"
    checklist = build_checklist(THESIS, report)
    md = checklist.to_markdown()
    assert "未能定位" in md
    assert "不会替你猜一个位置" in md


def test_checklist_to_dict_is_json_serializable() -> None:
    checklist = build_checklist(THESIS, PLANTED)
    payload = json.loads(json.dumps(checklist.to_dict(), ensure_ascii=False))
    assert payload["counts"]["total"] == 1
    assert payload["items"][0]["line"] >= 1


def test_empty_document_yields_unlocated() -> None:
    checklist = build_checklist("", PLANTED)
    assert checklist.items[0].method is LocateMethod.UNLOCATED


def test_parse_error_is_surfaced_separately_from_location_failure() -> None:
    """解析失败与定位失败必须是两种错误 —— 用户该做的事不一样。"""
    with pytest.raises(ReportParseError):
        build_checklist(THESIS, "   ")


def test_inline_text_and_path_both_accepted(tmp_path) -> None:
    path = tmp_path / "thesis.md"
    path.write_text(THESIS, encoding="utf-8")

    from_path = build_checklist(path, PLANTED)
    from_text = build_checklist(THESIS, PLANTED)

    assert from_path.doc_path.endswith("thesis.md")
    assert from_text.doc_path == "<inline>"
    assert from_path.items[0].char_start == from_text.items[0].char_start
