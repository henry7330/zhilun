"""P1 语义层测试：文本特征 / claim 原子化 / 支持程度判断。

重点覆盖两类**确定性可判**的幻觉形式：

- 极性矛盾（结论写反了）："提升" vs "下降"
- 数字缺失（数字编错了）：声称 5%，原文没有 5%

另外刻意测两类边界，避免规则过度反应：

- 含对冲词（"可能"）不等于没有证据
- 覆盖率低不等于编造（只是没找到支持）
"""

from __future__ import annotations

import pytest

from copilot.evidence.extract import RuleBasedExtractor, split_clauses, split_sentences
from copilot.evidence.models import Claim, EvidenceSpan, SupportStatus
from copilot.evidence.support import RuleBasedJudge
from copilot.evidence.text import (
    content_tokens,
    coverage,
    has_hedge,
    has_negation,
    numbers_in,
    polarity_conflicts,
)

# ─────────────────────────── 辅助 ───────────────────────────


def span(text: str, doc_id: str = "doc-1") -> EvidenceSpan:
    """构造一个结构合法的 span。

    支持判断只读取 ``text``，不校验来源，所以这里不需要真实文档。
    结构合法性仍由 ``EvidenceSpan.__post_init__`` 保证。
    """
    return EvidenceSpan(
        doc_id=doc_id,
        source_hash="0" * 64,
        char_start=0,
        char_end=len(text),
        text=text,
    )


def claim_of(text: str, evidence: list[str] | None = None) -> Claim:
    return Claim(
        claim_id="c1",
        text=text,
        evidence=[span(e) for e in (evidence or [])],
    )


# ─────────────────────── 文本特征 text.py ───────────────────────


def test_content_tokens_uses_cjk_bigrams() -> None:
    tokens = content_tokens("准确率提升")
    assert "准确" in tokens and "确率" in tokens and "提升" in tokens


def test_content_tokens_handles_latin_words() -> None:
    tokens = content_tokens("Accuracy Improved 5%")
    assert "accuracy" in tokens and "improved" in tokens


def test_content_tokens_normalizes_fullwidth() -> None:
    """全角/半角不应造成假不匹配。"""
    assert content_tokens("ＡＢＣ") == content_tokens("ABC")


def test_coverage_is_bounded_and_directional() -> None:
    assert coverage("准确率提升", "实验表明准确率提升明显") > 0.5
    assert coverage("完全无关的内容", "实验表明准确率提升明显") < 0.2


def test_coverage_empty_needle_is_zero() -> None:
    """空字符串不应被视为「完全覆盖」。"""
    assert coverage("", "任意证据") == 0.0


def test_numbers_in_extracts_and_normalizes() -> None:
    nums = numbers_in("提升 5 %，下降12％，共 3 倍")
    assert "5%" in nums
    assert "12%" in nums
    assert "3倍" in nums


def test_numbers_in_deduplicates_but_keeps_order() -> None:
    assert numbers_in("5% 然后 5% 再 12%") == ("5%", "12%")


def test_polarity_conflicts_detects_inverted_pair() -> None:
    conflicts = polarity_conflicts("准确率提升", "准确率下降")
    assert ("提升", "下降") in conflicts


def test_polarity_conflicts_is_symmetric() -> None:
    """方向相反也要能检测到 —— 用「轴」而非两两配对，避免枚举遗漏。"""
    assert polarity_conflicts("降低延迟", "延迟上升") == (("降低", "上升"),)


def test_polarity_conflicts_skips_internally_mixed_claim() -> None:
    """claim 自身同时含两种极性时，方向不可判定，应跳过而不是误报。

    宁可漏报不可误报：这类句子本应由原子化拆开；若没拆开，
    规则层不应凭猜测制造冲突。
    """
    mixed = "该方法提升了准确率但降低了延迟"
    assert polarity_conflicts(mixed, "该方法提升了准确率") == ()


def test_polarity_conflicts_does_not_cross_axes() -> None:
    """不同语义轴之间不应判为矛盾（"改善"与"上升"并非对立）。"""
    assert polarity_conflicts("效果改善", "数值上升") == ()


def test_polarity_conflicts_ignores_unrelated_text() -> None:
    assert polarity_conflicts("准确率提升", "本文使用了新方法") == ()


def test_hedge_and_negation_detection() -> None:
    assert has_hedge("该方法可能有效")
    assert not has_hedge("该方法有效")
    assert has_negation("该方法无效")
    assert not has_negation("该方法有效")


# ─────────────────────── 原子化 extract.py ───────────────────────


def test_split_sentences_handles_mixed_punctuation() -> None:
    out = split_sentences("第一句。第二句！第三句？")
    assert out == ["第一句", "第二句", "第三句"]


def test_split_clauses_splits_strong_conjunctions() -> None:
    out = split_clauses("该方法提升了准确率并且降低了延迟")
    assert len(out) == 2


def test_split_clauses_does_not_split_noun_coordination() -> None:
    """「和/与」多用于名词并列，切它会撕碎名词短语、制造假 claim。"""
    out = split_clauses("A 和 B 的方法都有效")
    assert len(out) == 1


def test_extract_splits_conjunction_into_two_claims() -> None:
    ex = RuleBasedExtractor()
    claims = ex.extract("该方法在五个数据集上提升了 5%，并且在低资源场景下降了 12%")
    assert len(claims) == 2
    assert all(c.status is SupportStatus.UNSUPPORTED for c in claims)


def test_extract_assigns_stable_ids() -> None:
    ex = RuleBasedExtractor()
    claims = ex.extract("第一句足够长了。第二句也足够长。", prefix="t")
    assert [c.claim_id for c in claims] == ["t-1", "t-2"]


def test_extract_drops_too_short_fragments() -> None:
    ex = RuleBasedExtractor()
    claims = ex.extract("好的。但是该方法在五个数据集上提升了准确率")
    assert all(len(c.text) >= 6 for c in claims)


def test_extract_lowers_confidence_for_hedged_claims() -> None:
    ex = RuleBasedExtractor()
    hedged = ex.extract("该方法可能提升准确率")
    plain = ex.extract("该方法提升准确率")
    assert hedged[0].confidence < plain[0].confidence


def test_extract_records_stats_for_diagnosis() -> None:
    """能看出「拆得太碎/太粗」，是调拆分策略的前提。"""
    ex = RuleBasedExtractor()
    ex.extract("第一句足够长了。好的。第二句也足够长了。")
    assert ex.last_stats.kept >= 2
    assert ex.last_stats.dropped >= 1


# ─────────────────── 支持程度判断 support.py ───────────────────


@pytest.fixture
def judge() -> RuleBasedJudge:
    return RuleBasedJudge()


def test_no_evidence_is_unsupported_not_fabricated(judge: RuleBasedJudge) -> None:
    """缺证据是信息不足，不是编造 —— 两者对用户的含义不同。"""
    j = judge.judge(claim_of("该方法提升了准确率"))
    assert j.status is SupportStatus.UNSUPPORTED
    assert "没有任何证据" in j.reasons[0] or "没有绑定" in j.reasons[0]


def test_supported_when_coverage_high_and_numbers_match(judge: RuleBasedJudge) -> None:
    j = judge.judge(
        claim_of("准确率提升了 5%", ["实验表明，准确率提升了 5%，效果稳定"])
    )
    assert j.status is SupportStatus.SUPPORTED
    assert j.signals.coverage >= judge.supported_threshold
    assert j.signals.missing_numbers == ()


def test_polarity_conflict_is_fabricated(judge: RuleBasedJudge) -> None:
    """★ 核心用例：结论把方向写反了。

    这类幻觉不需要模型就能确定性地抓到 —— 结论说提升，原文说下降，
    二者逻辑上不相容。
    """
    j = judge.judge(claim_of("该方法提升了准确率", ["该方法导致准确率下降"]))
    assert j.status is SupportStatus.FABRICATED
    assert j.signals.has_polarity_conflict
    assert "极性相反" in j.reasons[0]


def test_missing_number_downgrades_from_supported(judge: RuleBasedJudge) -> None:
    """★ 核心用例：数字对不上就不能算完全支持。

    这里刻意不让它直接判为 FABRICATED —— 证据可能只是省略了数字，
    确定性不足以断言"编造"。降级为部分支持是更诚实的选择。
    """
    j = judge.judge(claim_of("准确率提升了 5%", ["实验表明准确率提升了较多"]))
    assert j.status is SupportStatus.PARTIALLY_SUPPORTED
    assert "5%" in j.signals.missing_numbers
    assert j.signals.has_polarity_conflict is False


def test_low_coverage_is_unsupported(judge: RuleBasedJudge) -> None:
    j = judge.judge(claim_of("该方法在医疗场景同样有效", ["本文仅评估了电商推荐场景"]))
    assert j.status is SupportStatus.UNSUPPORTED


def test_partial_band_gives_partially_supported(judge: RuleBasedJudge) -> None:
    """覆盖率落在中间带时应为部分支持，而不是二选一。

    用「阈值极严 + 证据只部分重叠」构造出中间带。
    （若证据完全包含结论，覆盖率会是 1.0，那本来就该判支持。）
    """
    strict = RuleBasedJudge(supported_threshold=0.99, partial_threshold=0.10)
    j = strict.judge(claim_of("准确率提升了 5%", ["本文还讨论了准确率的定义"]))
    assert strict.partial_threshold <= j.signals.coverage < strict.supported_threshold
    assert j.status is SupportStatus.PARTIALLY_SUPPORTED


def test_judge_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="阈值"):
        RuleBasedJudge(supported_threshold=0.2, partial_threshold=0.8)


def test_signals_are_always_returned_for_audit(judge: RuleBasedJudge) -> None:
    """无论结论如何，都要能解释「为什么判成这样」。"""
    j = judge.judge(claim_of("准确率提升了 5%", ["准确率下降 12%"]))
    assert isinstance(j.signals.coverage, float)
    assert j.signals.claim_numbers == ("5%",)
    assert j.signals.evidence_numbers == ("12%",)


# ─────────────────── 串起来：原子化 → 判断 ───────────────────


def test_end_to_end_extract_then_judge(judge: RuleBasedJudge) -> None:
    """一条含两个断言的句子，其中一个与证据矛盾。

    这演示了原子化的价值：如果不拆，整句会被一起判伪，
    无法指出"是第一句有问题还是第二句"。
    """
    ex = RuleBasedExtractor()
    claims = ex.extract("该方法提升了准确率，并且降低了延迟")
    assert len(claims) == 2

    evidence = "实验显示该方法提升了准确率，但延迟有所上升"
    statuses = []
    for c in claims:
        c.evidence = [span(evidence)]
        statuses.append(judge.judge(c).status)

    # 第一条被支持；第二条（"降低了延迟"）与证据"延迟上升"矛盾
    assert statuses[0] is SupportStatus.SUPPORTED
    assert statuses[1] is SupportStatus.FABRICATED
