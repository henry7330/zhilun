"""P1 确定性校验的测试。

重点不是覆盖率，而是**证明它拦得住编造**：
论文助手最致命的失败模式是编造一篇不存在的文献，
所以必须有一个测试专门构造这种输入，并断言它被判为 FABRICATED。

另外特意测两类容易混淆的边界：

- "没有证据"（INSUFFICIENT）≠ "证据是假的"（FABRICATED）
- "本地库没有" ≠ "文献不存在"
"""

from __future__ import annotations

import pytest

from copilot.evidence.models import (
    CitationMeta,
    Claim,
    Document,
    EvidenceSpan,
    InMemoryDocStore,
    StaticDoiResolver,
    SupportStatus,
    hash_content,
)
from copilot.evidence.verify import (
    CitationProblem,
    InMemoryCitationLibrary,
    SpanProblem,
    VerdictKind,
    normalize_title,
    verify_citation,
    verify_claim,
    verify_span,
)

# ─────────────────────────── 夹具 ───────────────────────────

DOC_TEXT = "本文提出了一种新的检索方法。实验表明，该方法在五个数据集上准确率提升了 5%。"
REAL_DOI = "10.1234/real.2024.001"
FAKE_DOI = "10.9999/does.not.exist"


@pytest.fixture
def doc() -> Document:
    return Document.create("doc-1", DOC_TEXT)


@pytest.fixture
def store(doc: Document) -> InMemoryDocStore:
    return InMemoryDocStore([doc])


@pytest.fixture
def library() -> InMemoryCitationLibrary:
    return InMemoryCitationLibrary(
        [
            CitationMeta(
                doi=REAL_DOI,
                title="A New Retrieval Method",
                authors=("Zhang, S.", "Li, W."),
                year=2024,
                venue="ACL",
            )
        ]
    )


@pytest.fixture
def resolver() -> StaticDoiResolver:
    return StaticDoiResolver({REAL_DOI})


def make_span(doc: Document, text: str) -> EvidenceSpan:
    """按文本在文档中的真实位置构造一个合法 span。"""
    start = doc.content.index(text)
    return EvidenceSpan(
        doc_id=doc.doc_id,
        source_hash=doc.content_hash,
        char_start=start,
        char_end=start + len(text),
        text=text,
    )


# ─────────────────── EvidenceSpan 自身的不变式 ───────────────────


def test_span_rejects_negative_offset() -> None:
    with pytest.raises(ValueError, match="char_start"):
        EvidenceSpan("d", "h", -1, 3, "abc")


def test_span_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="char_end"):
        EvidenceSpan("d", "h", 5, 2, "")


def test_span_rejects_length_mismatch() -> None:
    """偏移跨度与 text 长度必须一致，否则说明构造时就算错了。"""
    with pytest.raises(ValueError, match="长度"):
        EvidenceSpan("d", "h", 0, 5, "abc")


# ─────────────────────── 证据片段校验 ───────────────────────


def test_verify_span_ok(doc: Document, store: InMemoryDocStore) -> None:
    span = make_span(doc, "准确率提升了 5%")
    result = verify_span(span, store)
    assert result.ok
    assert result.problem is SpanProblem.OK


def test_verify_span_detects_missing_document(
    doc: Document, store: InMemoryDocStore
) -> None:
    span = make_span(doc, "准确率提升了 5%")
    orphan = EvidenceSpan(
        doc_id="doc-不存在",
        source_hash=span.source_hash,
        char_start=span.char_start,
        char_end=span.char_end,
        text=span.text,
    )
    result = verify_span(orphan, store)
    assert result.problem is SpanProblem.DOC_NOT_FOUND


def test_verify_span_detects_tampering(doc: Document) -> None:
    """文档在检索后被替换 —— 只比对文本是发现不了的，必须靠 hash。"""
    span = make_span(doc, "准确率提升了 5%")
    tampered = Document.create("doc-1", "完全不同的内容" * 10)
    store = InMemoryDocStore([tampered])
    result = verify_span(span, store)
    assert result.problem is SpanProblem.HASH_MISMATCH


def test_verify_span_detects_text_mismatch(doc: Document) -> None:
    """hash 对得上，但 offset 记错了 —— 也要能发现。"""
    span = make_span(doc, "准确率提升了 5%")
    shifted = EvidenceSpan(
        doc_id=span.doc_id,
        source_hash=span.source_hash,
        char_start=span.char_start + 1,  # 故意错开一位
        char_end=span.char_end + 1,
        text=span.text,
    )
    result = verify_span(shifted, InMemoryDocStore([doc]))
    assert result.problem is SpanProblem.TEXT_MISMATCH


# ─────────────────────── 引用真实性校验 ───────────────────────


def test_normalize_title_ignores_case_and_punctuation() -> None:
    a = normalize_title("Attention Is All You Need")
    b = normalize_title("attention is all you need.")
    assert a == b


def test_verify_citation_ok(
    library: InMemoryCitationLibrary, resolver: StaticDoiResolver
) -> None:
    result = verify_citation(
        CitationMeta(doi=REAL_DOI, title="A New Retrieval Method", year=2024),
        library,
        resolver,
    )
    assert result.ok
    assert result.matched is not None
    assert result.matched.venue == "ACL"


def test_verify_citation_without_identifier(
    library: InMemoryCitationLibrary, resolver: StaticDoiResolver
) -> None:
    """既无 DOI 也无标题 —— 无法核实，不能当作支持性证据。"""
    result = verify_citation(CitationMeta(authors=("某人",)), library, resolver)
    assert result.problem is CitationProblem.NO_IDENTIFIER


def test_verify_citation_detects_fabricated_doi(
    library: InMemoryCitationLibrary, resolver: StaticDoiResolver
) -> None:
    """★ 核心用例：编造的 DOI 必须被判为无法解析。"""
    result = verify_citation(CitationMeta(doi=FAKE_DOI, title="Fabricated Paper"), library, resolver)
    assert result.problem is CitationProblem.DOI_NOT_RESOLVABLE


def test_verify_citation_not_in_local_library(
    library: InMemoryCitationLibrary,
) -> None:
    """本地库没有 ≠ 文献不存在 —— 这一条与"DOI 解析失败"强度不同。"""
    resolver = StaticDoiResolver({"10.5555/可解析但本地没有"})
    result = verify_citation(
        CitationMeta(doi="10.5555/可解析但本地没有"),
        library,
        resolver,
    )
    assert result.problem is CitationProblem.NOT_IN_LIBRARY


def test_verify_citation_detects_year_mismatch(
    library: InMemoryCitationLibrary, resolver: StaticDoiResolver
) -> None:
    result = verify_citation(
        CitationMeta(doi=REAL_DOI, title="A New Retrieval Method", year=1999),
        library,
        resolver,
    )
    assert result.problem is CitationProblem.YEAR_MISMATCH
    assert "1999" in result.detail and "2024" in result.detail


# ─────────────────────── claim 级汇总 ───────────────────────


def test_claim_with_fabricated_citation_is_fabricated(
    doc: Document,
    store: InMemoryDocStore,
    library: InMemoryCitationLibrary,
    resolver: StaticDoiResolver,
) -> None:
    """★ 核心用例：证据没问题，但引用是编的 —— 整条判为 FABRICATED。"""
    claim = Claim(
        claim_id="c1",
        text="该方法在五个数据集上准确率提升 5%",
        evidence=[make_span(doc, "准确率提升了 5%")],
        citations=[CitationMeta(doi=FAKE_DOI, title="不存在的论文")],
    )
    verdict = verify_claim(claim, store, library, resolver)
    assert verdict.kind is VerdictKind.FABRICATED
    assert verdict.status is SupportStatus.FABRICATED
    assert any("编造" in r for r in verdict.reasons)


def test_claim_with_tampered_evidence_is_fabricated(
    doc: Document,
    store: InMemoryDocStore,
    library: InMemoryCitationLibrary,
    resolver: StaticDoiResolver,
) -> None:
    """证据片段对不上原文 —— 也是编造信号。"""
    bogus_text = "原文里没有"  # 5 个字符
    bad_span = EvidenceSpan(
        doc_id=doc.doc_id,
        source_hash=hash_content("另一个版本的文档"),  # 与真实文档不符
        char_start=0,
        char_end=len(bogus_text),
        text=bogus_text,
    )
    claim = Claim(claim_id="c2", text="某结论", evidence=[bad_span])
    verdict = verify_claim(claim, store, library, resolver)
    assert verdict.kind is VerdictKind.FABRICATED
    assert verdict.status is SupportStatus.FABRICATED


def test_claim_without_any_evidence_is_insufficient_not_fabricated(
    store: InMemoryDocStore,
    library: InMemoryCitationLibrary,
    resolver: StaticDoiResolver,
) -> None:
    """★ 设计要点：「没有证据」不等于「编造」。

    前者应提示补充检索，后者应告警。混为一谈会让系统对普通的信息不足
    过度反应，也会削弱真正编造信号的严重性。
    """
    claim = Claim(claim_id="c3", text="一个尚无依据的猜测")
    verdict = verify_claim(claim, store, library, resolver)
    assert verdict.kind is VerdictKind.INSUFFICIENT
    assert verdict.status is SupportStatus.UNSUPPORTED
    assert verdict.kind is not VerdictKind.FABRICATED


def test_claim_with_valid_evidence_defers_to_semantic_judgment(
    doc: Document,
    store: InMemoryDocStore,
    library: InMemoryCitationLibrary,
    resolver: StaticDoiResolver,
) -> None:
    """确定性检查全过 ≠ 结论成立。

    这里必须返回 NEEDS_SEMANTIC 而不是直接 SUPPORTED——
    "证据存在"和"证据支持该结论"是两件事，后者需要语义判断。
    """
    claim = Claim(
        claim_id="c4",
        text="该方法提升了准确率",
        evidence=[make_span(doc, "准确率提升了 5%")],
        citations=[CitationMeta(doi=REAL_DOI, title="A New Retrieval Method", year=2024)],
    )
    verdict = verify_claim(claim, store, library, resolver)
    assert verdict.kind is VerdictKind.NEEDS_SEMANTIC
    assert verdict.status is None, "确定性层不应替语义层下结论"
