"""E4 · 降重测试。

核心验收：
1. 好的改写 → 被采纳，且报告给出**效果数字**；
2. 改写导致失真（丢数字/翻否定）→ **保真拦下，即使降重效果很好**；
3. 受保护片段（引用）→ **不改**；
4. 定义类 → **建议引用而不是改写**；
5. 数据类 → **不改**；
6. 改不动 → **如实报"未达标"，不假装成功**。
"""

from __future__ import annotations

import json

import pytest

from copilot.quality.report_parse import FlagType, build_checklist
from copilot.quality.rewrite import (
    CallableRewriter,
    CannedRewriter,
    ContentKind,
    RewriteRequest,
    Strategy,
    StubRewriter,
    SuggestionStatus,
    check_fidelity,
    choose_instruction,
    classify,
    measure_effect,
    run_rewrite,
    split_units,
)

# ─────────────────────────── 夹具 ───────────────────────────

DUP = "本文提出了一种基于图神经网络的推荐方法，并在三个数据集上验证了效果。"

THESIS = f"""# 绪论

学术文献的数量在过去二十年里呈现指数级增长。

# 相关工作

{DUP}

# 方法

我们设计了一种双阶段的检索策略。

# 结论

本文的方法取得了预期效果。
"""


def make_checklist(report: str = DUP):
    return build_checklist(THESIS, report, flag_type=FlagType.DUPLICATE)


# ─────────────────────── 单元切分 ───────────────────────


def test_split_units_splits_by_sentence() -> None:
    checklist = make_checklist()
    item = checklist.items[0]
    units = split_units(item, THESIS)
    assert units
    for u in units:
        assert THESIS[u.char_start : u.char_end] == u.text


def test_split_units_marks_quoted_text_protected() -> None:
    text = '他说“本文提出了一种基于图神经网络的推荐方法并在三个数据集上验证了效果”这句话。'
    checklist = build_checklist(text, "本文提出了一种基于图神经网络的推荐方法并在三个数据集上验证了效果")
    units = split_units(checklist.items[0], text)
    assert any(u.protected for u in units), [u.text for u in units]


def test_split_units_returns_empty_for_unlocated() -> None:
    """未定位的条目不知道改哪 —— 不能瞎改。"""
    checklist = build_checklist(THESIS, "完全无关的内容，与文档毫无关系。")
    units = split_units(checklist.items[0], THESIS)
    assert units == []


def test_split_units_merges_very_short_sentences() -> None:
    text = "因此。本文提出了一种基于图神经网络的推荐方法并在三个数据集上验证了效果。"
    checklist = build_checklist(text, "本文提出了一种基于图神经网络的推荐方法并在三个数据集上验证了效果")
    units = split_units(checklist.items[0], text, min_chars=8)
    assert len(units) >= 1


# ─────────────────────── 策略分类 ───────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("精度定义为正确样本数除以总样本数。", ContentKind.DEFINITION),
        ("定理：在满足上述条件时该估计量是无偏的。", ContentKind.DEFINITION),
        # 纯数据陈述（数字密集）→ DATA，整句不该改
        ("准确率 91.3%，召回率 88.7%，F1 值 90.0%。", ContentKind.DATA),
        # 结论里带数字 → 仍按结论处理（数字由保真校验守住，不必整句禁改）
        ("结果表明该方法在三个数据集上提升了 8.3%，召回率提升了 12.5%。", ContentKind.CONCLUSION),
        ("由此可见，稀疏信号与稠密信号具有互补性。", ContentKind.CONCLUSION),
        ("本文提出了一种基于图神经网络的推荐方法。", ContentKind.METHOD),
        ("这是一段普通的话。", ContentKind.OTHER),
    ],
)
def test_classify_content_kind(text: str, expected: ContentKind) -> None:
    assert classify(text) is expected


def test_definition_suggests_citation_not_rewrite() -> None:
    """★ 定义类不该改写，而该建议引用 —— 硬改定义会既不准确又不像原文。"""
    instruction = choose_instruction(ContentKind.DEFINITION)
    assert instruction.strategy is Strategy.CITE_INSTEAD
    assert "不要改写" in instruction.hint
    assert "引用" in instruction.rationale


def test_data_suggests_no_change() -> None:
    """★ 数据不能改 —— 改了就是伪造数据。"""
    instruction = choose_instruction(ContentKind.DATA)
    assert instruction.strategy is Strategy.SKIP
    assert "数字" in instruction.rationale


def test_method_keeps_terms() -> None:
    instruction = choose_instruction(ContentKind.METHOD)
    assert instruction.strategy is Strategy.REPHRASE
    assert "术语" in instruction.hint


# ─────────────────────── 改写器 ───────────────────────


def test_stub_rewriter_returns_original() -> None:
    stub = StubRewriter()
    req = RewriteRequest(text=DUP, instruction=choose_instruction(ContentKind.METHOD))
    assert stub.rewrite(req) == [DUP]


def test_canned_rewriter_returns_mapped() -> None:
    canned = CannedRewriter({DUP: ["完全不同的表述方式。"]})
    req = RewriteRequest(text=DUP, instruction=choose_instruction(ContentKind.METHOD))
    assert canned.rewrite(req) == ["完全不同的表述方式。"]


def test_canned_rewriter_returns_empty_for_unknown() -> None:
    canned = CannedRewriter({})
    req = RewriteRequest(text="没预设过的句子。", instruction=choose_instruction(ContentKind.OTHER))
    assert canned.rewrite(req) == []


def test_callable_rewriter_wraps_function() -> None:
    rw = CallableRewriter(lambda req: [req.text + "（改）"])
    req = RewriteRequest(text="原文", instruction=choose_instruction(ContentKind.OTHER))
    assert rw.rewrite(req) == ["原文（改）"]


# ─────────────────────── 效果度量 ───────────────────────


def test_effect_targets_met_for_real_rewrite() -> None:
    rewritten = "我们的方案借助图结构对用户与物品的关联建模，在三个公开数据集上完成了评估。"
    effect = measure_effect(rewritten, DUP)
    assert effect.meets_target, effect.summary()
    assert effect.measured_against == "original"
    assert effect.is_proxy, "没有来源原文时必须标为代理指标"


def test_effect_uses_source_when_available() -> None:
    """★ 有来源原文时用来源做基准（直接度量），而不是原句（代理指标）。"""
    source = "We propose a graph neural network based recommendation method."
    effect = measure_effect("译文", DUP, source_text=source)
    assert effect.measured_against == "source"
    assert not effect.is_proxy


def test_effect_fails_for_unchanged_text() -> None:
    effect = measure_effect(DUP, DUP)
    assert not effect.meets_target


def test_effect_proxy_is_labeled_in_summary() -> None:
    effect = measure_effect(DUP, DUP)
    assert "代理指标" in effect.summary()


# ─────────────────── ★ 保真校验 ───────────────────


def test_fidelity_passes_for_faithful_rewrite() -> None:
    rewritten = "我们的方案借助图结构对用户与物品的关联建模，在三个公开数据集上完成了评估。"
    result = check_fidelity(rewritten, DUP)
    assert result.ok, result.summary()


def test_fidelity_blocks_lost_number() -> None:
    """★ 最常见的改写事故：把数字吞了。"""
    original = "准确率提升了 8.3%，召回率提升了 12.5%。"
    rewritten = "准确率有明显提升，召回率也有所改善。"
    result = check_fidelity(rewritten, original)
    assert not result.ok
    assert any(i.kind == "number" for i in result.issues)


def test_fidelity_blocks_invented_number() -> None:
    original = "该方法在数据集上表现良好。"
    rewritten = "该方法在数据集上准确率提升了 37%。"
    result = check_fidelity(rewritten, original)
    assert not result.ok
    assert any(i.kind == "number" for i in result.issues)


def test_fidelity_blocks_negation_flip() -> None:
    """★ 把结论说反 —— 比不降重严重得多。"""
    original = "该方法并不优于基线方法。"
    rewritten = "该方法优于基线方法。"
    result = check_fidelity(rewritten, original)
    assert not result.ok
    assert any(i.kind == "negation" for i in result.issues)


def test_fidelity_blocks_lost_latin_term() -> None:
    original = "我们使用 BERT 作为编码器。"
    rewritten = "我们使用一个预训练模型作为编码器。"
    result = check_fidelity(rewritten, original)
    assert not result.ok
    assert any(i.kind == "term" for i in result.issues)


def test_fidelity_blocks_too_short_rewrite() -> None:
    result = check_fidelity("很短。", DUP)
    assert not result.ok
    assert any(i.kind == "length" for i in result.issues)


def test_fidelity_blocks_empty_rewrite() -> None:
    result = check_fidelity("   ", DUP)
    assert not result.ok


def test_fidelity_against_source_blocks_invented_number() -> None:
    source = "准确率提升了 8.3%。"
    rewritten = "准确率提升了 99%。"
    result = check_fidelity(rewritten, "准确率提升。", source_text=source)
    assert not result.ok


def test_fidelity_records_which_checks_ran() -> None:
    """用户要知道"通过"是基于什么检查 —— 不能只给一个 ok。"""
    result = check_fidelity("我们的方案借助图结构建模，在多个数据集上完成评估。", DUP)
    assert result.checks_run
    assert "数字保留" in result.checks_run


# ─────────────────────── 流水线 ───────────────────────


def test_pipeline_accepts_good_rewrite() -> None:
    """★ 验收 1：好的改写被采纳，并给出效果数字。"""
    good = "我们的方案借助图结构对用户与物品的关联建模，在三个公开数据集上完成了评估。"
    report = run_rewrite(make_checklist(), rewriter=CannedRewriter({DUP: [good]}))

    accepted = report.by_status(SuggestionStatus.ACCEPTED)
    assert accepted, [s.status for s in report.suggestions]
    s = accepted[0]
    assert s.effect is not None and s.effect.meets_target
    assert s.fidelity is not None and s.fidelity.ok
    assert s.chosen == good


def test_pipeline_rejects_unfaithful_even_if_effect_good() -> None:
    """★ 验收 2：**保真是否决条件** —— 效果再好也不能用。"""
    original = "准确率提升了 8.3%，召回率提升了 12.5%，在三个数据集上验证了效果。"
    text = f"# 方法\n\n{original}\n"
    checklist = build_checklist(text, original)
    # 这个候选把数字全删了：降重效果会很好，但失真
    bad = "有提升，也在若干数据集上做过验证。"
    report = run_rewrite(checklist, rewriter=CannedRewriter({original: [bad]}))

    assert not report.by_status(SuggestionStatus.ACCEPTED)
    rejected = report.by_status(SuggestionStatus.REJECTED_FIDELITY)
    assert rejected, [s.status for s in report.suggestions]
    assert "保真" in rejected[0].note


def test_pipeline_reports_failure_when_no_improvement() -> None:
    """★ 验收 6：改不动就如实说 —— 不假装成功。"""
    report = run_rewrite(make_checklist(), rewriter=StubRewriter())
    assert not report.by_status(SuggestionStatus.ACCEPTED)
    failed = report.by_status(SuggestionStatus.REJECTED_EFFECT)
    assert failed
    assert "人工改写" in failed[0].note


def test_pipeline_skips_protected_quoted_text() -> None:
    """★ 验收 3：引用不能被改写。"""
    text = '如文献所述“本文提出了一种基于图神经网络的推荐方法并在三个数据集上验证了效果”，可见其有效。'
    quoted = "本文提出了一种基于图神经网络的推荐方法并在三个数据集上验证了效果"
    checklist = build_checklist(text, quoted)
    report = run_rewrite(checklist, rewriter=StubRewriter())

    skipped = report.by_status(SuggestionStatus.SKIPPED_PROTECTED)
    assert skipped, [s.status for s in report.suggestions]
    assert "引用" in skipped[0].note


def test_pipeline_suggests_citation_for_definition() -> None:
    """★ 验收 4：定义类 → 建议引用，而不是改写。"""
    definition = "精确率定义为正确预测的正样本数除以所有预测为正的样本数。"
    text = f"# 方法\n\n{definition}\n"
    checklist = build_checklist(text, definition)
    report = run_rewrite(checklist, rewriter=StubRewriter())

    skipped = report.by_status(SuggestionStatus.SKIPPED_CITE)
    assert skipped, [s.status for s in report.suggestions]
    assert "引用" in skipped[0].note


def test_pipeline_skips_data_sentences() -> None:
    """★ 验收 5：数据类不改。"""
    data = "准确率 91.3%，召回率 88.7%，F1 值 90.0%。"
    text = f"# 实验\n\n{data}\n"
    checklist = build_checklist(text, data)
    report = run_rewrite(checklist, rewriter=StubRewriter())

    skipped = report.by_status(SuggestionStatus.SKIPPED_NO_CHANGE)
    assert skipped, [s.status for s in report.suggestions]


def test_pipeline_skips_unlocated_items() -> None:
    """未定位的条目不会被交给改写器。"""
    checklist = build_checklist(THESIS, "完全无关的内容。")
    report = run_rewrite(checklist, rewriter=StubRewriter())
    assert report.suggestions == []
    assert report.counts()["unlocated"] == 1


# ─────────────────────── 报告 ───────────────────────


def test_report_markdown_has_four_columns() -> None:
    good = "我们的方案借助图结构对用户与物品的关联建模，在三个公开数据集上完成了评估。"
    report = run_rewrite(make_checklist(), rewriter=CannedRewriter({DUP: [good]}))
    md = report.to_markdown()
    assert "降重报告" in md
    assert "已改写" in md
    assert "保真" in md
    assert "效果" in md


def test_report_markdown_lists_attention_items() -> None:
    report = run_rewrite(make_checklist(), rewriter=StubRewriter())
    md = report.to_markdown()
    assert "需要你处理的地方" in md


def test_report_to_dict_is_serializable() -> None:
    good = "我们的方案借助图结构对用户与物品的关联建模，在三个公开数据集上完成了评估。"
    report = run_rewrite(make_checklist(), rewriter=CannedRewriter({DUP: [good]}))
    payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    assert payload["counts"]["accepted"] == 1
    assert payload["suggestions"][0]["effect"]["meets_target"] is True


def test_report_states_semantic_limit() -> None:
    """必须写明本模块**不做完整语义等价判断** —— 不能让人以为保真=万无一失。"""
    report = run_rewrite(make_checklist(), rewriter=StubRewriter())
    md = report.to_markdown()
    assert "不做完整的语义等价判断" in md
