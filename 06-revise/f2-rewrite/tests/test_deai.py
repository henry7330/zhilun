"""E5 · 降AI 测试。

核心验收：
1. 特征**可测且可复核**；
2. 诊断**可解释**（说出偏离多少，而不是"像 AI"）；
3. 改写后特征**有改善**，并给出前后数字；
4. **保真是否决条件** —— 风格改善再明显，改动原意就弃用；
5. 没有风格问题 → **不改**（不为了"降 AI"硬改正常文字）；
6. 报告**不做身份判断**。
"""

from __future__ import annotations

import json

import pytest

from copilot.quality.deai import (
    CV_LOW,
    DeaiStatus,
    FeatureChange,
    StyleStrategy,
    choose_instruction,
    diagnose,
    profile,
    run_deai,
    verify_style,
)
from copilot.quality.report_parse import FlagType, build_checklist
from copilot.quality.rewrite import CannedRewriter, StubRewriter

# ─────────────────────────── 夹具 ───────────────────────────

#: 典型的"机器味"片段：句式整齐 + 套话连接词多 + 无具体信息
AI_LIKE = "首先，该方法是有效的。其次，该方法具有优势。此外，该方法易于实现。最后，该方法值得推广。"

#: 人手写的"好改写"：长短句交错、无套话连接词
GOOD_REWRITE = (
    "这个方法管用。我们在两个真实业务场景里跑过，准确率明显高于原来的规则方案，"
    "维护成本也低了不少。核心逻辑不到一百行，团队里两个人两天就能接手。"
)

#: 假"改写"：加了原文没有的数字（编造）
INVENTED = "该方法有效。在 37 个数据集上准确率达到 94.2%。实现简单。"

THESIS = f"""# 方法

{AI_LIKE}

# 结论

本文的方法取得了预期效果。
"""


def make_checklist(report: str = AI_LIKE):
    return build_checklist(THESIS, report, flag_type=FlagType.AI)


# ─────────────────────── 风格特征 ───────────────────────


def test_profile_measures_sentence_length_cv() -> None:
    uniform = "这是一句。那也是句。这是三句。那是四句。"
    varied = "短句。这一句话非常长，包含了很多修饰成分和限定条件，一直写到这里还没有结束的意思。又短。"
    assert profile(varied).sentence_length_cv > profile(uniform).sentence_length_cv


def test_profile_measures_transition_density() -> None:
    heavy = "首先，A。其次，B。此外，C。最后，D。"
    light = "A 是可行的。B 也值得一试。C 还需要观察。"
    assert profile(heavy).transition_density > profile(light).transition_density


def test_profile_measures_specificity() -> None:
    vague = "该方法效果不错，值得推广。"
    specific = "该方法把召回率从 71.2% 提到 78.5%，在 BERT 上做了消融。"
    assert profile(specific).specificity > profile(vague).specificity


def test_profile_counts_are_reproducible() -> None:
    """同一段文字，任何人算出来的值应当一致 —— 这是"客观特征"与"推断"的区别。"""
    a = profile(AI_LIKE).to_dict()
    b = profile(AI_LIKE).to_dict()
    assert a == b


def test_profile_handles_empty_text() -> None:
    p = profile("")
    assert p.chars == 0
    assert p.sentences == 0
    assert p.transition_density == 0.0


# ─────────────────────── 诊断 ───────────────────────


def test_diagnose_flags_uniform_sentences() -> None:
    diags = diagnose(AI_LIKE)
    features = {d.feature for d in diags}
    assert "sentence_length_cv" in features


def test_diagnose_flags_excess_transitions() -> None:
    diags = diagnose(AI_LIKE)
    features = {d.feature for d in diags}
    assert "transition_density" in features


def test_diagnose_flags_low_specificity() -> None:
    diags = diagnose(AI_LIKE)
    features = {d.feature for d in diags}
    assert "specificity" in features


def test_diagnosis_is_explainable_with_numbers() -> None:
    """★ 诊断必须说出**偏离多少**，而不是"这段像 AI"。"""
    diags = diagnose(AI_LIKE)
    for d in diags:
        assert str(d.value) or True
        assert any(ch.isdigit() for ch in d.explanation), (
            f"诊断缺乏数字依据：{d.explanation}"
        )


def test_normal_text_has_no_diagnosis() -> None:
    """风格正常的文字不该被诊断出问题 —— 否则所有内容都会被硬改。"""
    normal = (
        "我们在两个真实业务场景里跑过，准确率比原来的规则方案高。"
        "维护成本也低了不少。核心逻辑不到一百行。"
    )
    assert diagnose(normal) == [], [str(d) for d in diagnose(normal)]


# ─────────────────────── 策略 ───────────────────────


def test_choose_instruction_maps_each_diagnosis() -> None:
    diags = diagnose(AI_LIKE)
    instr = choose_instruction(diags)
    assert StyleStrategy.VARY_SENTENCE in instr.strategies or \
           StyleStrategy.TRIM_TRANSITIONS in instr.strategies


def test_choose_instruction_dedupes() -> None:
    diags = diagnose(AI_LIKE)
    instr = choose_instruction(diags)
    assert len(instr.strategies) == len(set(instr.strategies))


def test_empty_diagnosis_gives_empty_instruction() -> None:
    instr = choose_instruction([])
    assert instr.is_empty
    assert "不需要改写" in instr.rationale


def test_add_specifics_forbids_fabrication() -> None:
    """★ "补充具体信息"必须明确禁止编造 —— 编造数据的危害远大于 AI 味。"""
    from copilot.quality.deai.strategy import _STRATEGY_HINT

    hint = _STRATEGY_HINT[StyleStrategy.ADD_SPECIFICS]
    assert "不要编造" in hint


# ─────────────────── ★ 双校验 ───────────────────


def test_verify_improves_style() -> None:
    diags = diagnose(AI_LIKE)
    v = verify_style(AI_LIKE, GOOD_REWRITE, diagnoses=diags)
    assert v.style_improved, v.summary()
    assert v.fidelity_ok, v.summary()
    assert v.ok


def test_verify_blocks_invented_numbers() -> None:
    """★ 保真是否决条件：加了原文没有的数字 → 弃用。"""
    diags = diagnose(AI_LIKE)
    v = verify_style(AI_LIKE, INVENTED, diagnoses=diags)
    assert not v.fidelity_ok
    assert not v.ok


def test_verify_blocks_when_style_unchanged() -> None:
    diags = diagnose(AI_LIKE)
    v = verify_style(AI_LIKE, AI_LIKE, diagnoses=diags)
    assert not v.style_improved
    assert not v.ok


def test_verify_records_feature_changes() -> None:
    diags = diagnose(AI_LIKE)
    v = verify_style(AI_LIKE, GOOD_REWRITE, diagnoses=diags)
    assert v.changes
    assert any(c.improved for c in v.changes)
    # 每个变化都带前后数值
    for c in v.changes:
        assert isinstance(c.before, float)
        assert isinstance(c.after, float)


def test_feature_change_direction_logic() -> None:
    # transition_density 越小越好
    down = FeatureChange("transition_density", before=1.0, after=0.2, direction=-1)
    up = FeatureChange("transition_density", before=1.0, after=2.0, direction=-1)
    assert down.improved
    assert not up.improved


# ─────────────────────── 流水线 ───────────────────────


def test_pipeline_accepts_good_style_rewrite() -> None:
    """★ 验收 3：改写后特征有改善 → 采纳，并给出前后数字。"""
    report = run_deai(make_checklist(), rewriter=CannedRewriter({AI_LIKE: [GOOD_REWRITE]}))

    accepted = report.by_status(DeaiStatus.ACCEPTED)
    assert accepted, [s.status for s in report.suggestions]
    s = accepted[0]
    assert s.verification is not None and s.verification.ok
    assert any(c.improved for c in s.verification.changes)


def test_pipeline_rejects_fabricated_rewrite() -> None:
    """★ 验收 4：编造数字 → 保真拦下。"""
    report = run_deai(make_checklist(), rewriter=CannedRewriter({AI_LIKE: [INVENTED]}))

    assert not report.by_status(DeaiStatus.ACCEPTED)
    rejected = report.by_status(DeaiStatus.REJECTED_FIDELITY)
    assert rejected, [s.status for s in report.suggestions]


def test_pipeline_skips_text_without_style_problems() -> None:
    """★ 验收 5：风格正常 → **不改**。

    为了"降 AI"去硬改正常文字，只会把内容改坏。
    """
    normal = (
        "我们在两个真实业务场景里跑过，准确率比原来的规则方案高。"
        "维护成本也低了不少。核心逻辑不到一百行。"
    )
    text = f"# 方法\n\n{normal}\n"
    checklist = build_checklist(text, normal, flag_type=FlagType.AI)
    report = run_deai(checklist, rewriter=StubRewriter())

    assert report.by_status(DeaiStatus.NO_DIAGNOSIS)
    assert not report.by_status(DeaiStatus.ACCEPTED)


def test_pipeline_reports_no_candidate_with_stub() -> None:
    report = run_deai(make_checklist(), rewriter=StubRewriter())
    assert not report.by_status(DeaiStatus.ACCEPTED)
    assert report.counts()["needs_attention"] >= 1


def test_pipeline_skips_protected_quoted_text() -> None:
    text = f'如文献所述“{AI_LIKE}”，可见其有效。'
    checklist = build_checklist(text, AI_LIKE, flag_type=FlagType.AI)
    report = run_deai(checklist, rewriter=StubRewriter())
    assert report.by_status(DeaiStatus.SKIPPED_PROTECTED)


def test_pipeline_skips_unlocated() -> None:
    checklist = build_checklist(THESIS, "完全无关的内容。", flag_type=FlagType.AI)
    report = run_deai(checklist, rewriter=StubRewriter())
    assert report.suggestions == []
    assert report.counts()["unlocated"] == 1


# ─────────────────────── 报告 ───────────────────────


def test_report_makes_no_identity_claim() -> None:
    """★ 验收 6：报告只说特征变化，**不说"这段是/不是 AI 写的"**。"""
    report = run_deai(make_checklist(), rewriter=StubRewriter())
    md = report.to_markdown()
    assert "不做作者身份判断" in md
    assert "不可靠" in md
    # 不应出现这类断言
    assert "这段是 AI 写的" not in md
    assert "判定为 AI" not in md


def test_report_shows_feature_changes() -> None:
    report = run_deai(
        make_checklist(), rewriter=CannedRewriter({AI_LIKE: [GOOD_REWRITE]})
    )
    md = report.to_markdown()
    assert "特征变化" in md
    assert "sentence_length_cv" in md or "transition_density" in md


def test_report_to_dict_is_serializable() -> None:
    report = run_deai(
        make_checklist(), rewriter=CannedRewriter({AI_LIKE: [GOOD_REWRITE]})
    )
    payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    assert payload["counts"]["accepted"] == 1


def test_report_lists_needs_attention() -> None:
    report = run_deai(make_checklist(), rewriter=StubRewriter())
    assert "需要你处理的地方" in report.to_markdown()

# ─────────────────── ★ 中文数字计数的 idiom 假阳性 ───────────────────


def test_idiom_jin_yi_bu_is_not_a_specific_point() -> None:
    """★ 「进一步」不是具体信息点。

    实测踩过：`_CN_NUMBER` 把「进一步」匹配成「一步」（`步` 在量词表里），
    于是一个纯 idiom 被算成"具体信息点"，`specificity` 虚高，
    本来该被判"缺少具体信息"的段落逃过诊断。

    这个 bug 是在接真实模型做 E5 改写时暴露的：模型改写里多了个「进一步」，
    specificity 从 0.000 跳到 3.448。
    """
    with_idiom = profile("该方法在低资源场景下仍有进一步提升的空间。")
    without = profile("该方法在低资源场景下仍有提升的空间。")
    assert with_idiom.specificity == without.specificity == 0.0


def test_idiom_does_not_hide_low_specificity() -> None:
    """带 idiom 的泛泛表述，仍然应该被判"缺少具体信息"。"""
    text = "该方法有进一步提升空间。此外，它还有优化余地。"
    assert "specificity" in {d.feature for d in diagnose(text)}


def test_real_chinese_quantity_still_counted() -> None:
    """真实数量仍要数进去 —— 修 bug 不能把正确的也去掉。"""
    assert profile("我们跑了三个数据集。").specificity > 0
    assert profile("该方法需要两步处理。").specificity > 0
