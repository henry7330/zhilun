"""E6 · 审稿挑错测试。

核心验收：
1. 术语不一致 → **检出**（人工极难全文核对的那种）
2. 摘要关键词正文没出现 → **检出**
3. 一级章节过短 → 检出（且**不**对小节噪声报警）
4. 图表引用断裂 → 复用 E1 规则检出
5. 合格文档 → **零误报**
6. 报告**写明没做的事**（不假装检查了论证逻辑）
"""

from __future__ import annotations

import json

import pytest

from copilot.quality.format.report import Severity
from copilot.quality.format.template import Template
from copilot.quality.review import review_text

# ─────────────────────────── 夹具 ───────────────────────────

#: 一份"有问题"的草稿：
#: - 摘要关键词「鲁棒性」正文没出现
#: - 同时用了「召回率」和「查全率」
#: - 「结论」章过短
#: - 正文引用了不存在的图 5
BAD_DOC = """# 摘要

本文研究了一种检索增强方法，并在多个数据集上完成了验证。

**关键词**：检索增强；鲁棒性；推荐系统

# 绪论

检索增强生成在近年受到关注。相关工作讨论了召回率的提升方式。
如 图 5 所示，该方法效果显著。

# 方法

我们采用两阶段流程：先做粗筛，再做精排。整个流程的查全率在实验中得到了验证。

# 结论

有效。

# 参考文献

[1] 张三. 一种方法[J]. 计算机学报, 2020, 43(2): 1-10.
"""

#: 一份"合格"的草稿 —— 必须零误报
GOOD_DOC = """# 摘要

本文研究了一种检索增强方法，并在多个数据集上完成了验证。

**关键词**：检索增强；推荐系统

# 绪论

检索增强生成在近年受到关注。已有工作讨论了检索质量对最终效果的影响。

# 方法

我们采用两阶段流程：先做粗筛，再做精排。粗筛用词面信号召回候选，
精排用语义信号重新排序，两阶段之间用一个小型重排模块连接。
该流程在推荐系统的排序任务上做了初步验证。

# 结论

本文的方法在多个数据集上取得了稳定效果。

# 参考文献

[1] 张三. 一种方法[J]. 计算机学报, 2020, 43(2): 1-10.
"""


def tpl() -> Template:
    return Template(
        name="测试模板",
        required_sections=["摘要", "绪论", "方法", "结论", "参考文献"],
        min_body_words=None,
    )


# ─────────────────────── 术语一致性 ───────────────────────


def test_detects_terminology_inconsistency() -> None:
    """★ 同时用了「召回率」和「查全率」—— 人工几乎不可能核对出来。"""
    report = review_text(BAD_DOC, tpl())
    rev001 = [i for i in report.issues if i.code == "REV001"]
    assert rev001
    assert "召回率" in rev001[0].message and "查全率" in rev001[0].message


def test_terminology_inconsistency_is_warning_not_error() -> None:
    """只提示确认，**不判定对错** —— 作者可能是有意区分。"""
    report = review_text(BAD_DOC, tpl())
    rev001 = [i for i in report.issues if i.code == "REV001"]
    assert rev001[0].severity is Severity.WARNING


def test_consistent_terminology_not_flagged() -> None:
    report = review_text(GOOD_DOC, tpl())
    assert "REV001" not in report.codes()


# ─────────────────────── 关键词覆盖 ───────────────────────


def test_detects_keyword_missing_from_body() -> None:
    """★ 摘要列了「鲁棒性」，正文一次都没出现。"""
    report = review_text(BAD_DOC, tpl())
    rev002 = [i for i in report.issues if i.code == "REV002"]
    assert rev002
    assert "鲁棒性" in rev002[0].message


def test_keyword_present_in_body_not_flagged() -> None:
    report = review_text(GOOD_DOC, tpl(), min_section_chars=10)
    assert "REV002" not in report.codes()


def test_no_keyword_line_yields_nothing() -> None:
    doc = "# 摘要\n\n没有关键词行。\n"
    report = review_text(doc, tpl())
    assert "REV002" not in report.codes()


# ─────────────────────── 章节篇幅 ───────────────────────


def test_short_top_level_section_flagged() -> None:
    """「结论」只有两个字 → 论述不充分。"""
    report = review_text(BAD_DOC, tpl())
    rev003 = [i for i in report.issues if i.code == "REV003"]
    assert any("结论" in i.message for i in rev003)


def test_short_sections_are_info_not_error() -> None:
    report = review_text(BAD_DOC, tpl())
    rev003 = [i for i in report.issues if i.code == "REV003"]
    assert all(i.severity is Severity.INFO for i in rev003)


def test_subsection_length_not_checked() -> None:
    """只查一级章节 —— 对小节设阈值只会制造噪声。"""
    doc = "# 一\n\n## 一点一\n\n短。\n\n## 一点二\n\n也很短。\n"
    report = review_text(doc, tpl(), include_figures=False)
    assert "REV003" not in report.codes()


# ─────────────────── 图表引用（复用 E1） ───────────────────


def test_figure_reference_check_is_reused_from_e1() -> None:
    """★ 复用 E1 的规则 —— 同一套能力只有一份实现，规则码也一致。"""
    report = review_text(BAD_DOC, tpl())
    assert "FIG002" in report.codes(), f"应复用 E1 的 FIG002；实际 {report.codes()}"


def test_figures_can_be_skipped() -> None:
    report = review_text(BAD_DOC, tpl(), include_figures=False)
    assert "FIG002" not in report.codes()


# ─────────────────────── 整体 ───────────────────────


def test_good_document_has_no_review_issues() -> None:
    """★ 验收 5：合格文档零误报（只看 E6 自己的规则，不含图表与模板检查）。"""
    report = review_text(GOOD_DOC, tpl(), include_figures=False, min_section_chars=10)
    review_codes = [c for c in report.codes() if c.startswith("REV")]
    assert review_codes == [], [i.message for i in report.issues]


def test_report_sorts_by_severity_then_line() -> None:
    report = review_text(BAD_DOC, tpl())
    ranks = [i.severity.rank for i in report.issues]
    assert ranks == sorted(ranks)


def test_report_states_what_it_does_not_do() -> None:
    """★ 验收 6：**写明没做的事** —— 不假装检查了论证逻辑。

    一份声称"已检查论证逻辑"实则只查了术语的报告，
    会让用户误以为可以放心提交。
    """
    md = review_text(BAD_DOC, tpl()).to_markdown()
    assert "论证链是否断裂" in md
    assert "不做" in md


def test_report_to_dict_is_serializable() -> None:
    payload = json.loads(
        json.dumps(review_text(BAD_DOC, tpl()).to_dict(), ensure_ascii=False)
    )
    assert payload["counts"]["total"] >= 1


def test_codes_are_unique_and_ordered() -> None:
    report = review_text(BAD_DOC, tpl())
    codes = report.codes()
    assert len(codes) == len(set(codes))

# ─────────── ★ 标题+章节（# 题目 + ## 章节）结构下的篇幅检查 ───────────


def test_section_length_works_with_title_heading() -> None:
    """★ 很多论文是 `# 题目` + `## 章节` 的写法。

    此时顶层只有 1 个节点，真正的章节在第二层。
    早期实现只看顶层 → **篇幅检查被完全静默跳过**（不报错、不提示），
    看起来像"这篇论文篇幅都合格"。

    这个 bug 是建评测集才发现的：单测夹具都用 `# 章节` 写法，所以没暴露。
    """
    md = "# 论文题目\n\n## 摘要\n\n内容。\n\n## 结论\n\n短。\n"
    report = review_text(md, min_section_chars=50)
    assert "REV003" in report.codes(), [i.message for i in report.issues]
    assert any("结论" in i.message for i in report.issues)


def test_section_length_still_reads_top_level_when_multiple() -> None:
    """多个顶层章节时行为不变 —— 修 bug 不能改变原有正确路径。"""
    md = "# 摘要\n\n内容一。\n\n# 结论\n\n短。\n"
    report = review_text(md, min_section_chars=50)
    assert "REV003" in report.codes()
