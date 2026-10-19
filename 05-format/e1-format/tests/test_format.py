"""E1 格式校验 Agent 测试。

两条最重要的验收标准：

1. **一份埋了已知问题的草稿 → 逐条全部检出**；
2. **一份合格草稿 → 误报为 0**。

第 2 条比第 1 条更重要：规则写得激进时第 1 条很容易满足，
但用户会被假报警淹没，然后不再相信这个工具。
"""

from __future__ import annotations

import pytest

from copilot.quality.format import (
    FormatReport,
    Severity,
    Template,
    check_format,
    count_words,
    default_template,
    load_template,
    parse_markdown,
    save_template,
)
from copilot.quality.format.rules.structure import normalize_heading

# ─────────────────────────── 夹具 ───────────────────────────

#: 测试模板：关掉字数下限。默认模板要求 8000 字，
#: 任何测试用的短草稿都会命中 SEC004，掩盖真正要测的东西。
TPL = Template(
    name="测试模板",
    required_sections=["摘要", "绪论", "方法", "结论", "参考文献"],
    section_order_strict=True,
    reference_style="gbt7714",
    min_body_words=None,
    max_authors_before_etal=3,
    max_heading_level=3,
)

#: 完全合格的草稿 —— 必须产出 0 个问题
GOOD_DRAFT = """# 摘要

本文研究了一种新的检索方法，并在多个数据集上进行了验证。

# 绪论

如 图 1 所示，本文的整体框架分为三个阶段。

**图 1** 研究框架

# 方法

参数设置见 表 1。

**表 1** 参数设置

# 结论

本文提出的方法在实验中取得了预期效果。

# 参考文献

[1] 张三, 李四. 一种新的检索方法[J]. 计算机学报, 2020, 43(2): 1-10.
[2] 王五. 深度学习导论[M]. 北京: 科学出版社, 2021.
"""

#: 故意埋了问题的草稿
BAD_DRAFT = """# 摘要

摘要内容。

# 绪论

如 图 3 所示，该方法有效。

# 结论

结论内容。

# 参考文献

[1] 张三. 一种方法. 计算机学报.
[2] 李四, 王五, 赵六, 钱七. 深度学习. 科学出版社.
[4] 孙八. 某种技术[J]. 软件学报, 2019.
"""


# ─────────────────── ★ 两条核心验收标准 ★ ───────────────────


def test_good_draft_has_zero_issues() -> None:
    """★ 合格草稿必须零误报。"""
    report = check_format(GOOD_DRAFT, TPL)
    assert report.issues == [], (
        "合格草稿产生了误报：\n"
        + "\n".join(f"  {i.code} {i.location} {i.message}" for i in report.issues)
    )
    assert report.passed


def test_bad_draft_detects_every_planted_issue() -> None:
    """★ 埋了的问题必须逐条检出。"""
    report = check_format(BAD_DRAFT, TPL)
    codes = set(report.codes())

    expected = {
        "SEC001",  # 缺少「方法」章节
        "FIG002",  # 正文引用了不存在的 图 3
        "REF004",  # 序号 1,2,4 不连续
        "REF001",  # [1][2] 缺文献类型标识
        "REF003",  # [1][2] 缺年份
        "REF006",  # [2] 四位作者未用「等」
    }
    missing = expected - codes
    assert not missing, f"漏检：{sorted(missing)}；实际检出 {sorted(codes)}"
    assert report.has_error


# ─────────────────────── 子部分 ① 文档模型 ───────────────────────


def test_parse_sections_builds_tree() -> None:
    doc = parse_markdown("# 一\n\n## 一点一\n\n# 二\n")
    assert [s.title for s in doc.sections] == ["一", "二"]
    assert [c.title for c in doc.sections[0].children] == ["一点一"]


def test_section_span_covers_until_next_heading() -> None:
    doc = parse_markdown("# 一\n\n内容\n\n# 二\n")
    first = doc.sections[0]
    assert first.span[0] == 0
    assert first.span[1] == 4  # 到「# 二」所在行为止


def test_numbered_heading_extracts_number() -> None:
    doc = parse_markdown("# 2.1 研究方法\n")
    s = doc.sections[0]
    assert s.number == "2.1"
    assert normalize_heading(s.title) == "研究方法"


def test_normalize_heading_strips_various_number_styles() -> None:
    for raw in ("1.2 研究方法", "二、研究方法", "（2）研究方法", "研究方法"):
        assert normalize_heading(raw) == "研究方法"


def test_normalize_heading_handles_chinese_chapter_prefix() -> None:
    """★ 回归：「第一章 绪论」必须能归一化成「绪论」。

    曾经的编号正则只覆盖 ``1.2`` / ``一、`` / ``（2）``，
    **漏了中文论文最常用的「第X章」**，导致这类草稿的每个必需章节
    都被误报为"缺失" —— 一个模板匹配问题伪装成了 5 条内容问题。
    """
    for raw in ("第一章 绪论", "第1章 绪论", "第 2 章 绪论", "第一部分 绪论"):
        assert normalize_heading(raw) == "绪论", raw


def test_sec001_not_triggered_by_chinese_chapter_headings() -> None:
    """★ 回归（端到端）：用「第X章」写的草稿不应误报章节缺失。"""
    doc = (
        "# 摘要\n\nA\n\n# 第一章 绪论\n\nB\n\n# 第二章 方法\n\nC\n\n"
        "# 第三章 结论\n\nD\n\n# 参考文献\n\n[1] 甲. A[J]. 学报, 2020.\n"
    )
    tpl = Template(
        name="t",
        required_sections=["摘要", "绪论", "方法", "结论", "参考文献"],
        min_body_words=None,
    )
    report = check_format(doc, tpl)
    assert "SEC001" not in report.codes(), [i.message for i in report.issues]


def test_parse_references_extracts_fields() -> None:
    doc = parse_markdown(
        "# 参考文献\n\n[1] 张三, 李四. 一种方法[J]. 计算机学报, 2020, 43(2): 1-10.\n"
    )
    assert len(doc.references) == 1
    ref = doc.references[0]
    assert ref.index == 1
    assert ref.ref_type == "J"
    assert ref.year == 2020
    assert ref.authors == ["张三", "李四"]


def test_parse_references_supports_plain_numbering() -> None:
    doc = parse_markdown("# 参考文献\n\n1. 张三. 一种方法[J]. 学报, 2020.\n")
    assert len(doc.references) == 1
    assert doc.references[0].index == 1


def test_reference_block_is_excluded_from_crossref_scan() -> None:
    """★ 参考文献里的「图 1」不能被当成正文交叉引用。

    这是本模块最容易踩的假阳性陷阱：文献题名里出现"图"太常见了
    （如《基于图神经网络的…》），一旦误扫就会报出一堆"引用了不存在的图"。
    """
    doc = parse_markdown(
        "# 正文\n\n没有引用任何图。\n\n# 参考文献\n\n[1] 张三. 基于图 1 结构的模型[J]. 学报, 2020.\n"
    )
    assert doc.cross_refs == [], f"误从参考文献中扫出交叉引用：{doc.cross_refs}"


def test_caption_line_is_not_counted_as_crossref() -> None:
    """题注行自身的「图 1」是声明，不是引用。"""
    doc = parse_markdown("# 正文\n\n**图 1** 框架图\n")
    assert len(doc.captions) == 1
    assert doc.cross_refs == []


def test_bold_caption_is_recognized() -> None:
    doc = parse_markdown("# 正文\n\n**图 1** 研究框架\n")
    assert len(doc.captions) == 1
    assert doc.captions[0].text == "研究框架"


def test_prose_line_starting_with_figure_number_is_not_a_caption() -> None:
    """★ 回归：以「图 N」开头的**正文句子**不能被误判成题注。

    「图 3 给出了两类方法的对比结果。」与题注「图 3 方法对比」在词法上
    无法区分 —— 实测踩过：误判为题注后它不再算作"引用"，
    于是反向误报出「图 3 未被正文引用」。

    判据：中文题注按惯例不加句号，故以句末标点结尾的行视为正文。
    """
    doc = parse_markdown("# 正文\n\n图 3 给出了两类方法的对比结果。\n")
    assert doc.captions == [], f"被误判为题注：{doc.captions}"
    assert len(doc.cross_refs) == 1


def test_english_caption_with_trailing_period_is_recognized() -> None:
    """英文题注常以句点结尾，不能一并当作正文 —— 用长度兜底。"""
    doc = parse_markdown("# Body\n\nFigure 1. Overall framework.\n")
    assert len(doc.captions) == 1


def test_crossref_detected_when_following_chinese_text() -> None:
    """★ 回归：「如图 3 所示」必须能扫出交叉引用。

    曾经用 ``(?<![\\w])`` 做前瞻排除，而 Python 的 ``\\w`` **匹配汉字**，
    于是"图"前面是汉字时前瞻恒不成立，**所有中文交叉引用都漏扫**。

    注意原来的单测写的是「如 图 9 所示」（**中间有空格**）——
    恰好是唯一能绕过该 bug 的写法，所以单测全绿而实际功能是坏的。
    这个回归用例刻意**不加空格**。
    """
    doc = parse_markdown("# 正文\n\n如图 3 所示，效果显著。\n")
    assert len(doc.cross_refs) == 1, f"漏扫：{doc.cross_refs}"
    assert doc.cross_refs[0].number == "3"


def test_fig002_detects_missing_figure_without_spaces() -> None:
    """★ 回归（端到端）：无空格写法下也必须报出「引用了不存在的图」。"""
    report = check_format("# 正文\n\n如图 9 所示。\n", Template(name="t", min_body_words=None))
    assert "FIG002" in report.codes(), [i.message for i in report.issues]


def test_fig003_not_reported_when_referenced_without_spaces() -> None:
    """★ 回归：确实被引用的图**不应**被报"未被引用"。

    这是同一个 bug 的反向表现 —— 漏扫导致"明明引用了却说没引用"。
    """
    doc = "# 正文\n\n如图 1 所示。\n\n**图 1** 框架\n"
    report = check_format(doc, Template(name="t", min_body_words=None))
    assert "FIG003" not in report.codes(), [i.message for i in report.issues]


def test_count_words_handles_cjk_and_western() -> None:
    # 中文按字、西文按词
    assert count_words("你好世界") == 4
    assert count_words("hello world") == 2
    assert count_words("你好 hello") == 3


def test_body_words_excludes_references() -> None:
    """参考文献块不计入正文字数。

    用「同一内容加/不加参考文献」对比，而不是硬编码一个字数 ——
    硬编码会把「标题算不算正文字数」这种无关细节一起锁进断言，
    将来调整统计口径就会莫名其妙地失败。
    """
    with_refs = parse_markdown(
        "# 正文\n\n你好世界\n\n# 参考文献\n\n[1] 张三. 一种方法[J]. 学报, 2020.\n"
    )
    without_refs = parse_markdown("# 正文\n\n你好世界\n")
    assert with_refs.body_words == without_refs.body_words
    assert "张三" not in with_refs.body_text


# ─────────────────────── 子部分 ② 模板 ───────────────────────


def test_template_roundtrip(tmp_path) -> None:
    p = save_template(TPL, tmp_path / "t.json")
    loaded = load_template(p)
    assert loaded == TPL


def test_template_rejects_unknown_field() -> None:
    """写错字段名却不报错，会让用户以为配置生效了 —— 必须显式拒绝。"""
    with pytest.raises(ValueError, match="未知字段"):
        Template.from_json({"name": "x", "min_words": 100})


def test_template_rejects_unsupported_reference_style() -> None:
    with pytest.raises(ValueError, match="参考文献风格"):
        Template(name="x", reference_style="mla")


def test_default_template_is_valid() -> None:
    t = default_template()
    assert t.required_sections
    assert t.reference_style == "gbt7714"


# ─────────────────────── 子部分 ③ 结构规则 ───────────────────────


def test_sec001_missing_required_section() -> None:
    report = check_format("# 摘要\n\n内容\n", TPL)
    titles = [i.message for i in report.issues if i.code == "SEC001"]
    assert any("绪论" in m for m in titles)


def test_sec002_detects_wrong_order() -> None:
    tpl = Template(name="t", required_sections=["绪论", "摘要"], min_body_words=None)
    doc = "# 摘要\n\nA\n\n# 绪论\n\nB\n"
    report = check_format(doc, tpl)
    assert "SEC002" in report.codes()


def test_sec002_not_reported_when_sections_missing() -> None:
    """章节有缺失时不判顺序 —— 否则会因同一根因刷出连锁误报。"""
    tpl = Template(name="t", required_sections=["甲", "乙", "丙"], min_body_words=None)
    report = check_format("# 甲\n\nA\n\n# 丙\n\nC\n", tpl)
    assert "SEC002" not in report.codes()


def test_sec003_heading_too_deep() -> None:
    tpl = Template(name="t", max_heading_level=2)
    report = check_format("# 一\n\n### 太深了\n\n内容\n", tpl)
    assert "SEC003" in report.codes()


def test_sec004_word_count_below_minimum() -> None:
    tpl = Template(name="t", min_body_words=100)
    report = check_format("# 一\n\n太短。\n", tpl)
    assert "SEC004" in report.codes()


def test_sec005_empty_section() -> None:
    report = check_format("# 摘要\n\n内容\n\n# 绪论\n\n# 结论\n", TPL)
    empty = [i for i in report.issues if i.code == "SEC005"]
    assert any("绪论" in i.message for i in empty)


def test_sec005_not_triggered_when_section_has_only_subsections() -> None:
    """只有子章节、自己没正文的章节**不应**被报为空 —— 这是常见结构。"""
    doc = "# 一\n\n## 一点一\n\n有内容\n"
    report = check_format(doc, Template(name="t", min_body_words=None))
    assert "SEC005" not in report.codes(), [i.message for i in report.issues]


# ─────────────────────── 子部分 ④ 参考文献规则 ───────────────────────


def test_ref001_missing_type_marker() -> None:
    report = check_format("# 参考文献\n\n[1] 张三. 一种方法. 学报. 2020.\n", TPL)
    assert "REF001" in report.codes()


def test_ref002_invalid_type_marker() -> None:
    report = check_format("# 参考文献\n\n[1] 张三. 一种方法[XQ]. 学报, 2020.\n", TPL)
    assert "REF002" in report.codes()


def test_ref003_missing_year() -> None:
    report = check_format("# 参考文献\n\n[1] 张三. 一种方法[J]. 学报.\n", TPL)
    assert "REF003" in report.codes()


def test_ref004_discontinuous_index() -> None:
    report = check_format(
        "# 参考文献\n\n[1] 甲. A[J]. 学报, 2020.\n[3] 乙. B[J]. 学报, 2021.\n", TPL
    )
    assert "REF004" in report.codes()


def test_ref005_empty_reference_list() -> None:
    report = check_format("# 参考文献\n\n暂无。\n", TPL)
    assert "REF005" in report.codes()


def test_ref006_too_many_authors_without_etal() -> None:
    report = check_format(
        "# 参考文献\n\n[1] 甲, 乙, 丙, 丁. 一种方法[J]. 学报, 2020.\n", TPL
    )
    assert "REF006" in report.codes()


def test_ref006_not_triggered_when_etal_used() -> None:
    report = check_format(
        "# 参考文献\n\n[1] 甲, 乙, 丙, 等. 一种方法[J]. 学报, 2020.\n", TPL
    )
    assert "REF006" not in report.codes(), [i.message for i in report.issues]


def test_reference_rules_tolerate_english_entries() -> None:
    """英文文献不应因为"抽不出中文式字段"而被大量误报。

    只断言不报 REF001（它确实有 [J]），不假装能完整校验英文格式 ——
    那是尚未实现的能力，测试不替它吹牛。
    """
    report = check_format(
        "# 参考文献\n\n[1] Smith J, Lee K. A method[J]. Journal of Testing, 2020, 5(1): 1-9.\n",
        TPL,
    )
    assert "REF001" not in report.codes()
    assert "REF003" not in report.codes()


# ─────────────────────── 子部分 ⑤ 图表规则 ───────────────────────


def test_fig001_discontinuous_figure_numbering() -> None:
    doc = "# 正文\n\n**图 1** A\n\n**图 3** C\n"
    report = check_format(doc, Template(name="t", min_body_words=None))
    assert "FIG001" in report.codes()


def test_fig002_reference_to_missing_figure() -> None:
    report = check_format("# 正文\n\n如 图 9 所示。\n", Template(name="t", min_body_words=None))
    codes = report.codes()
    assert "FIG002" in codes
    fig002 = [i for i in report.issues if i.code == "FIG002"][0]
    assert fig002.severity is Severity.ERROR


def test_fig003_unreferenced_caption_is_info_only() -> None:
    """未被引用的图只给 INFO —— 是否要引用是作者的决定，规则不替他判断。"""
    report = check_format("# 正文\n\n没有引用。\n\n**图 1** A\n", Template(name="t", min_body_words=None))
    fig003 = [i for i in report.issues if i.code == "FIG003"]
    assert fig003 and fig003[0].severity is Severity.INFO


def test_fig001_supports_chapter_based_numbering() -> None:
    """「图 2-1 / 2-2」应按章内序号连续，而不是要求全文从 1 连续。"""
    doc = "# 正文\n\n**图 2-1** A\n\n**图 2-2** B\n"
    report = check_format(doc, Template(name="t", min_body_words=None))
    assert "FIG001" not in report.codes(), [i.message for i in report.issues]


def test_figure_and_table_numbering_are_independent() -> None:
    """图与表各自编号，互不干扰。"""
    doc = "# 正文\n\n**图 1** A\n\n**表 1** B\n"
    report = check_format(doc, Template(name="t", min_body_words=None))
    assert "FIG001" not in report.codes()


# ─────────────────────── 子部分 ⑥ 报告 ───────────────────────


def test_report_sorts_by_severity_then_line() -> None:
    report = check_format(BAD_DRAFT, TPL)
    ranks = [i.severity.rank for i in report.issues]
    assert ranks == sorted(ranks)


def test_report_counts_and_codes() -> None:
    report = check_format(BAD_DRAFT, TPL)
    c = report.counts()
    assert c["total"] == len(report.issues)
    assert c["error"] + c["warning"] + c["info"] == c["total"]
    assert len(report.codes()) == len(set(report.codes()))


def test_report_markdown_contains_every_issue() -> None:
    report = check_format(BAD_DRAFT, TPL)
    md = report.to_markdown()
    for issue in report.issues:
        assert issue.code in md
    assert "格式校验报告" in md


def test_report_to_dict_is_serializable() -> None:
    import json

    report = check_format(BAD_DRAFT, TPL)
    payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    assert payload["passed"] is False
    assert len(payload["issues"]) == len(report.issues)
    assert payload["stats"]["references"] == 3


def test_passed_ignores_warnings_and_info() -> None:
    """只有 ERROR 会判定为不通过 —— 否则满篇 INFO 会把系统变成噪声源。"""
    report = FormatReport(doc=parse_markdown("# x"), template=TPL, issues=[])
    assert report.passed
