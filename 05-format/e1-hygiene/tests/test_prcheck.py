"""F2 · 提交前检查测试。

核心验收：
1. **占位符必被查出**（TODO / 待补 / 空占位 / 空章节）；
2. **敏感信息必被查出**（邮箱 / 手机号 / 身份证 / 路径 / 密钥），且**报告里要脱敏**；
3. **匿名性问题必被查出**（致谢非空 / 基金号 / 机构名）；
4. **门禁给明确的 yes/no**，并区分阻断项与警告；
5. **合格文档零误报**。
"""

from __future__ import annotations

import pytest

from copilot.delivery.prcheck import (
    check_anonymity,
    check_placeholders,
    check_sensitive,
    check_submission,
    default_manifest,
)
from copilot.quality.format import parse_markdown
from copilot.quality.format.report import Severity
from copilot.quality.format.template import Template

# ─────────────────────────── 夹具 ───────────────────────────

DIRTY = """# 一篇还有问题的论文

## 摘要

本文研究了一种方法。

## 绪论

TODO: 这里要补充研究背景。

联系方式：zhangsan@example.com，手机 13800138000。

实验数据放在 C:\\Users\\zhang\\data\\results.csv。

## 方法

我们采用两阶段流程。

## 结论

（待补）

## 致谢

感谢清华大学李教授的指导。本工作受国家自然科学基金（No. 12345678）资助。

## 参考文献

[1] 张三. 一种方法[J]. 计算机学报, 2020, 43(2): 1-10.
"""

CLEAN = """# 一篇可以提交的论文

## 摘要

本文研究了一种用于文献检索的增强方法，并在三个公开数据集上完成了验证。

**关键词**：检索增强；推荐系统

## 绪论

检索增强生成在近年受到广泛关注。已有工作讨论了检索质量对最终效果的影响，
本文在此基础上进一步分析了长尾查询上的表现差异。

## 方法

我们采用两阶段流程：先做粗筛，再做精排。粗筛用词面信号召回候选，
精排用语义信号重新排序，两阶段之间用一个小型重排模块连接。

## 结论

本文的方法在多个数据集上取得了稳定效果，在低资源场景下仍有提升空间。

## 参考文献

[1] 张三. 一种方法[J]. 计算机学报, 2020, 43(2): 1-10.
"""


LENIENT = Template(
    name="宽松模板",
    required_sections=["摘要", "结论", "参考文献"],
    min_body_words=None,
)


@pytest.fixture
def dirty_doc():
    return parse_markdown(DIRTY)


@pytest.fixture
def clean_doc():
    return parse_markdown(CLEAN)


# ─────────────────── 占位符 ───────────────────


def test_detects_todo(dirty_doc) -> None:
    issues = check_placeholders(dirty_doc)
    assert issues
    assert any("TODO" in i.message for i in issues)


def test_detects_chinese_placeholder(dirty_doc) -> None:
    issues = check_placeholders(dirty_doc)
    assert any("待补" in i.message or "中文占位" in i.message for i in issues)


def test_placeholders_are_blockers(dirty_doc) -> None:
    """占位符是**阻断项** —— 带着 TODO 提交是低级事故。"""
    assert all(i.severity is Severity.ERROR for i in check_placeholders(dirty_doc))


def test_clean_doc_has_no_placeholder(clean_doc) -> None:
    assert check_placeholders(clean_doc) == []


# ─────────────────── ★ 敏感信息 ───────────────────


def test_detects_email(dirty_doc) -> None:
    issues = check_sensitive(dirty_doc)
    assert any("邮箱" in i.message for i in issues)


def test_detects_phone(dirty_doc) -> None:
    issues = check_sensitive(dirty_doc)
    assert any("手机号" in i.message for i in issues)


def test_detects_absolute_path(dirty_doc) -> None:
    issues = check_sensitive(dirty_doc)
    assert any("路径" in i.message for i in issues)


def test_sensitive_values_are_masked(dirty_doc) -> None:
    """★ 报告本身也不能泄露 —— 否则泄露面从正文扩大到报告。"""
    issues = check_sensitive(dirty_doc)
    joined = " ".join(i.message for i in issues)
    assert "13800138000" not in joined
    assert "*" in joined


def test_detects_id_card() -> None:
    doc = parse_markdown("# 论文\n\n身份证 110101199001011234 用于登记。\n")
    assert any("身份证" in i.message for i in check_sensitive(doc))


def test_detects_api_key_shape() -> None:
    doc = parse_markdown("# 论文\n\n配置里写了 sk-abcdefghijklmnopqrstuvwx 这个值。\n")
    assert any("密钥" in i.message for i in check_sensitive(doc))


def test_clean_doc_has_no_sensitive(clean_doc) -> None:
    assert check_sensitive(clean_doc) == []


# ─────────────────── ★ 匿名性 ───────────────────


def test_detects_non_empty_acknowledgement(dirty_doc) -> None:
    issues = check_anonymity(dirty_doc)
    assert any("致谢" in i.message for i in issues)


def test_detects_grant_number(dirty_doc) -> None:
    issues = check_anonymity(dirty_doc)
    assert any("基金" in i.message for i in issues)


def test_detects_institution_name(dirty_doc) -> None:
    issues = check_anonymity(dirty_doc)
    assert any("机构" in i.message for i in issues)


def test_anonymity_only_runs_when_required(clean_doc) -> None:
    """非双盲时不跑匿名检查（默认关闭）。"""
    report = check_submission(CLEAN, anonymity_required=False, template=LENIENT)
    assert "PRE003" not in [i.code for i in report.issues]


def test_anonymity_enabled_by_flag() -> None:
    text = "# 论文\n\n本论文由清华大学完成。\n"
    report = check_submission(text, anonymity_required=True)
    assert "PRE003" in [i.code for i in report.issues]


# ─────────────────── 结构复用与重定级 ───────────────────


def test_empty_section_is_upgraded_to_blocker() -> None:
    """★ E1 把空章节判为 INFO（格式上没问题），F2 要升为 **ERROR**。

    同一次检测，两种判决：E1 管"格式合不合规"，F2 管"能不能交"。
    """
    text = (
        "# 论文\n\n## 摘要\n\n有内容在这里。\n\n## 结论\n\n## 参考文献\n\n"
        "[1] 张三. 一种方法[J]. 计算机学报, 2020.\n"
    )
    report = check_submission(text)
    empty = [i for i in report.issues if i.code == "SEC005"]
    assert empty, [i.code for i in report.issues]
    assert empty[0].severity is Severity.ERROR


def test_structure_checks_can_be_disabled() -> None:
    mf = default_manifest()
    mf.include_format_checks = False
    mf.checks = [c for c in mf.checks if c.code != "PRE004"]
    report = check_submission(DIRTY, manifest=mf)
    assert "SEC005" not in [i.code for i in report.issues]


# ─────────────────── ★ 门禁 ───────────────────


def test_dirty_doc_cannot_submit() -> None:
    report = check_submission(DIRTY, anonymity_required=True)
    assert not report.can_submit
    assert report.blockers()
    assert "不能提交" in report.verdict()


def test_verdict_gives_clear_yes_no() -> None:
    """只说"有 12 个问题"没有决策价值 —— 必须给 yes/no。"""
    report = check_submission(DIRTY, anonymity_required=True)
    verdict = report.verdict()
    assert verdict.startswith("❌") or verdict.startswith("✅")


def test_clean_doc_can_submit() -> None:
    """★ 验收 5：合格文档零误报。"""
    report = check_submission(CLEAN, anonymity_required=False, template=LENIENT)
    blockers = [i for i in report.blockers() if i.code.startswith("PRE")]
    assert blockers == [], [f"{i.code} {i.message}" for i in blockers]
    assert report.can_submit, report.summary_line()


def test_report_markdown_lists_blockers_first() -> None:
    report = check_submission(DIRTY, anonymity_required=True)
    md = report.to_markdown()
    assert "阻断项" in md
    assert md.index("阻断项") < md.index("警告")


def test_report_to_dict_is_serializable() -> None:
    import json

    report = check_submission(DIRTY, anonymity_required=True)
    payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    assert payload["can_submit"] is False
    assert payload["counts"]["blocker"] > 0


def test_default_template_is_strict_by_design() -> None:
    """★ 默认模板是**本科论文的严格要求**（9 个必需章节 + 8000 字）。

    所以 `check_submission` 不传模板时，一份"内容不错但只有 200 字"的草稿
    会被判为不可提交 —— 这是**正确行为**，不是 bug。
    测试与演示里要显式传宽松模板才能验证"零误报"。
    """
    from copilot.quality.format import default_template

    tpl = default_template()
    assert len(tpl.required_sections) >= 8
    assert (tpl.min_body_words or 0) >= 5000


def test_manifest_is_declarative() -> None:
    """清单是数据不是代码 —— 可以按投稿要求调整。"""
    mf = default_manifest(anonymity_required=True, min_body_words=50000)
    codes = [c.code for c in mf.enabled()]
    assert "PRE003" in codes
    assert mf.min_body_words == 50000
