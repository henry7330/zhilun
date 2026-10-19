"""E 阶段（格式规整）测试。

核心验收：
1. **没有格式要求时如实说"未做版式规整"**，不假装做过；
2. 有一段式 formatter 时，产物**必须有 diff**（否则改了什么谁也不知道）；
3. **未覆盖项清单**必须出现 —— 防"规整过了 = 格式没问题"的错觉；
4. 导出产物要能回读校验。
"""

from __future__ import annotations

import pytest

from copilot.format_stage import COMMON_FORMAT_ITEMS, run_format

DRAFT = """# 摘要

本文研究推荐系统中的长尾问题。

# 方法

我们设计了两阶段检索策略。

# 结论

方法有效。

# 参考文献

[1] 张三. 一种方法[J]. 计算机学报, 2020.
"""

REQUIREMENT = """标题：黑体三号居中
正文：宋体小四，行距 1.5 倍
参考文献：按 GB/T 7714 著录
"""


def test_empty_draft_is_rejected() -> None:
    with pytest.raises(ValueError, match="draft"):
        run_format(draft="   ")


def test_without_requirement_does_not_pretend_to_normalize() -> None:
    """★ 没给格式要求就不该说"规整过了"。"""
    report = run_format(draft=DRAFT)
    e1 = report.by_node()["E1"]
    messages = " ".join(f.message for f in e1.findings)
    assert "未提供格式要求" in messages
    assert e1.stats["formatter"] == "none"
    assert e1.stats["changed"] == 0


def test_formatter_output_must_come_with_diff() -> None:
    """★ 一段式的唯一可核查物就是 diff —— 没有它等于黑箱。"""
    report = run_format(
        draft=DRAFT,
        requirement=REQUIREMENT,
        formatter=lambda text, req: text.replace("我们设计了", "本文设计了"),
    )
    e1 = report.by_node()["E1"]
    assert e1.stats["changed"] > 0
    changed = [f for f in e1.findings if f.payload.get("diff")]
    assert changed, "有改动就必须给 diff"
    assert any("本文设计了" in line for line in changed[0].payload["diff"])


def test_uncovered_items_are_listed() -> None:
    """★ 只改用户要求的项，其余**如实列出没处理**。"""
    report = run_format(draft=DRAFT, requirement=REQUIREMENT)
    uncovered = [
        f for f in report.by_node()["E1"].findings
        if "没提到" in f.message
    ]
    assert uncovered, "应当列出未覆盖的格式项"
    # 要求里没提页眉页脚
    assert any("页眉页脚" in f.message for f in uncovered)


def test_common_items_are_declared_upfront() -> None:
    assert len(COMMON_FORMAT_ITEMS) >= 8
    assert "参考文献著录格式" in COMMON_FORMAT_ITEMS


def test_nodes_run_in_declared_order() -> None:
    order: list[str] = []
    run_format(draft=DRAFT, on_node=lambda code, _r: order.append(code))
    assert order == ["E1", "E2", "E3"]


def test_consistency_node_runs() -> None:
    report = run_format(draft=DRAFT)
    e2 = report.by_node()["E2"]
    assert e2.ok
    assert "issues" in e2.stats


def test_export_node_produces_entries() -> None:
    report = run_format(draft=DRAFT)
    e3 = report.by_node()["E3"]
    assert e3.ok
    assert e3.stats["entries"] >= 1
    assert e3.stats["tex_chars"] > 0


def test_formatter_failure_is_reported_not_swallowed() -> None:
    def boom(text: str, req: str) -> str:
        raise RuntimeError("模型超时")

    report = run_format(draft=DRAFT, requirement=REQUIREMENT, formatter=boom)
    e1 = report.by_node()["E1"]
    assert any("调用失败" in f.message for f in e1.findings)
    # 未改动文件 —— 不能说"规整好了"
    assert e1.stats["changed"] == 0


def test_placeholder_is_detected_by_hygiene() -> None:
    """占位符属于"内容卫生"，归 E1 的检查器。"""
    draft = DRAFT.replace("方法有效。", "方法有效 TODO 补充实验。")
    report = run_format(draft=draft)
    messages = " ".join(f.message for f in report.by_node()["E1"].findings)
    assert "TODO" in messages or "占位" in messages


def test_report_marks_stage_e() -> None:
    assert run_format(draft=DRAFT).stage == "E"
