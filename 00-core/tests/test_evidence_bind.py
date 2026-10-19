"""P1 证据绑定测试。

最关键的两个测试：

1. **回环测试**：绑定产出的 span 必须能通过 ``verify_span`` 的三重一致性校验。
   这是整条链路的接缝 —— 偏移算错、hash 拿错，都只有这个测试能发现。
2. **矛盾证据仍然被绑定**：这是本模块的核心设计判断。
   若把矛盾的证据丢弃，"结论与原文相矛盾"这个最强幻觉信号就永远
   触发不了，只会退化成"没找到证据"。两者对用户的含义完全不同。
"""

from __future__ import annotations

import pytest

from copilot.evidence.bind import (
    RuleBinder,
    bind_evidence,
    iter_segments,
)
from copilot.evidence.models import (
    Claim,
    Document,
    InMemoryDocStore,
)
from copilot.evidence.support import RuleBasedJudge
from copilot.evidence.models import SupportStatus
from copilot.evidence.verify import SpanProblem, verify_span

# ─────────────────────────── 夹具 ───────────────────────────

DOC_A = (
    "本文提出了一种新的检索方法。"
    "实验表明，该方法在五个数据集上准确率提升了 5%。"
    "然而在低资源场景下，性能下降约 12%，这是本方法的主要局限。"
)
DOC_B = "本文研究了电商推荐场景下的点击率预估问题。与检索方法无关。"


@pytest.fixture
def docs() -> list[Document]:
    return [Document.create("doc-a", DOC_A), Document.create("doc-b", DOC_B)]


# ─────────────────────── 分段与偏移 ───────────────────────


def test_iter_segments_returns_exact_offsets() -> None:
    content = "第一句。第二句。"
    segs = list(iter_segments(content))
    assert len(segs) == 2
    for start, end, text in segs:
        assert content[start:end] == text, "偏移必须能在原文上原样切回来"


def test_iter_segments_trims_whitespace_and_keeps_offsets() -> None:
    content = "  第一句  。  第二句  "
    for start, end, text in iter_segments(content):
        assert content[start:end] == text
        assert text == text.strip()


def test_iter_segments_handles_empty_and_no_boundary() -> None:
    assert list(iter_segments("")) == []
    assert [t for _, _, t in iter_segments("没有句号的整段文字")] == ["没有句号的整段文字"]


def test_segments_never_cross_normalization_boundary(docs: list[Document]) -> None:
    """偏移基于**原文**：即使原文含全角字符，也要能切回来。

    若在 NFKC 归一化后的文本上算偏移，这里必然失败。
    """
    doc = Document.create("d", "指标提升了 ５％（全角）。第二句。")
    for start, end, text in iter_segments(doc.content):
        assert doc.content[start:end] == text


# ─────────────────────── 绑定基本行为 ───────────────────────


def test_binds_matching_sentence(docs: list[Document]) -> None:
    binder = RuleBinder(min_relevance=0.5)
    result = binder.bind("该方法在五个数据集上准确率提升了 5%", docs)
    assert result.bound
    assert "准确率提升了 5%" in result.spans[0].text
    assert result.spans[0].doc_id == "doc-a"


def test_does_not_bind_unrelated_claim(docs: list[Document]) -> None:
    """相关性不足时宁可不绑定 —— 错绑比漏绑危险。"""
    binder = RuleBinder(min_relevance=0.5)
    result = binder.bind("量子纠缠在低温下的退相干机制", docs)
    assert not result.bound
    assert result.rejection_reason is not None
    assert "低于绑定阈值" in result.rejection_reason


def test_no_documents_gives_clear_reason() -> None:
    result = RuleBinder().bind("任意结论", [])
    assert not result.bound
    assert result.rejection_reason == "没有可检索的文档"


def test_respects_max_spans(docs: list[Document]) -> None:
    binder = RuleBinder(min_relevance=0.05, max_spans=1)
    assert len(binder.bind("该方法在五个数据集上准确率提升了 5%", docs).spans) <= 1


def test_accepts_generator_input(docs: list[Document]) -> None:
    """回归测试：实现曾两次遍历 documents，对生成器会静默拿到空序列。"""
    binder = RuleBinder(min_relevance=0.5)
    result = binder.bind("该方法在五个数据集上准确率提升了 5%", (d for d in docs))
    assert result.bound


def test_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="min_relevance"):
        RuleBinder(min_relevance=1.5)
    with pytest.raises(ValueError, match="max_spans"):
        RuleBinder(max_spans=0)


# ─────────────────── 设计判断：矛盾证据仍绑定 ───────────────────


def test_contradicting_evidence_is_still_bound(docs: list[Document]) -> None:
    """★ 核心设计判断：相关但**矛盾**的证据也要绑定。

    结论声称"提升了低资源场景的性能"，原文说"性能下降约 12%"。
    绑定层只回答"是否在讲同一件事"，一致性交给 support 层。
    如果把矛盾证据丢掉，最强幻觉信号就永远触发不了。
    """
    result = RuleBinder(min_relevance=0.3).bind(
        "该方法提升了低资源场景下的性能", docs
    )
    assert result.bound, "相关但矛盾的证据必须被绑定，否则矛盾检测无法触发"
    bound_text = " ".join(s.text for s in result.spans)
    assert "下降" in bound_text


# ─────────────────── 回环：绑定 → 校验 ───────────────────


def test_bound_span_passes_verification(docs: list[Document]) -> None:
    """★ 回环测试：绑定产出的 span 必须通过三重一致性校验。

    这是整条链路的接缝。偏移算错、hash 拿错、文本被改写，
    都只有这个测试能发现 —— 单看任一侧的测试都会漏。
    """
    store = InMemoryDocStore(docs)
    result = RuleBinder(min_relevance=0.5).bind(
        "该方法在五个数据集上准确率提升了 5%", docs
    )
    assert result.bound
    for span in result.spans:
        check = verify_span(span, store)
        assert check.ok, f"绑定的 span 未通过校验：{check.problem.value} — {check.detail}"


def test_tampered_document_is_caught_after_binding(docs: list[Document]) -> None:
    """绑定后文档被替换 —— 校验必须发现（这正是 hash 的用途）。"""
    result = RuleBinder(min_relevance=0.5).bind(
        "该方法在五个数据集上准确率提升了 5%", docs
    )
    tampered = InMemoryDocStore([Document.create("doc-a", "内容被整体替换了。")])
    checks = [verify_span(s, tampered) for s in result.spans]
    assert all(c.problem is SpanProblem.HASH_MISMATCH for c in checks)


# ─────────────── 端到端：绑定 → 支持判断 ───────────────


def test_bind_then_support_detects_contradiction(docs: list[Document]) -> None:
    """★ 全链路：自动绑定 → 支持判断 → 抓到"结论与原文相反"。

    这条链路证明了本模块的设计判断是成立的：
    绑定层**不丢弃**矛盾的证据，因此下游才能报出 FABRICATED，
    而不是含糊地退化成"没找到证据"。
    """
    claim = Claim(claim_id="c1", text="该方法提升了低资源场景下的性能")
    bind_evidence(claim, docs, binder=RuleBinder(min_relevance=0.3))

    assert claim.has_evidence, "相关证据应被自动绑定上"
    judgment = RuleBasedJudge().judge(claim)
    assert judgment.status is SupportStatus.FABRICATED
    assert judgment.signals.has_polarity_conflict


def test_bind_then_support_confirms_supported_claim(docs: list[Document]) -> None:
    """正向对照：确实被支持的结论应判为 SUPPORTED。

    没有这条对照，上面的 FABRICATED 可能只是"规则一律报错"的假象。
    """
    claim = Claim(claim_id="c2", text="该方法在五个数据集上准确率提升了 5%")
    bind_evidence(claim, docs, binder=RuleBinder(min_relevance=0.3))

    judgment = RuleBasedJudge().judge(claim)
    assert judgment.status is SupportStatus.SUPPORTED
