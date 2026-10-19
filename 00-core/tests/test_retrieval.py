"""P2 检索层测试。

最关键的是 **P1↔P2 接缝测试**：切分出的片段必须能通过证据层的三重校验。
检索层与证据层各自单测都过、接缝却断掉，是这类系统最常见的故障。
"""

from __future__ import annotations

import pytest

from copilot.evidence.models import Document, InMemoryDocStore
from copilot.evidence.verify import SpanProblem, verify_span
from copilot.retrieval.base import Chunk, ScoredChunk, best_per_document, rank_documents
from copilot.retrieval.corpus import chunk_document
from copilot.retrieval.metrics import (
    EvalQuery,
    bootstrap_ci,
    evaluate_retriever,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from copilot.retrieval.sparse import Bm25Retriever, CharTrigramRetriever, RrfFusion, rrf

# ─────────────────────────── 夹具 ───────────────────────────

TEXT_A = (
    "本文提出了一种新的检索方法。"
    "实验表明，该方法在五个数据集上准确率提升了 5%。"
    "然而在低资源场景下，性能下降约 12%。"
)
TEXT_B = "本文研究了电商推荐场景下的点击率预估问题。"


@pytest.fixture
def doc_a() -> Document:
    return Document.create("doc-a", TEXT_A)


@pytest.fixture
def doc_b() -> Document:
    return Document.create("doc-b", TEXT_B)


@pytest.fixture
def chunks(doc_a: Document, doc_b: Document) -> list[Chunk]:
    out = chunk_document(doc_a, max_chars=40, overlap_sentences=0)
    out += chunk_document(doc_b, max_chars=40, overlap_sentences=0)
    return out


# ─────────────────────── Chunk 不变式 ───────────────────────


def test_chunk_rejects_offset_length_mismatch() -> None:
    with pytest.raises(ValueError, match="偏移跨度"):
        Chunk("d", "c", 0, 5, "abc", "h")


def test_chunk_to_span_preserves_fields(doc_a: Document) -> None:
    c = chunk_document(doc_a, max_chars=20)[0]
    span = c.to_span()
    assert span.doc_id == c.doc_id
    assert span.char_start == c.char_start
    assert span.text == c.text


# ───────────────────★ P1↔P2 接缝测试 ★───────────────────


def test_chunks_pass_evidence_verification(doc_a: Document, doc_b: Document) -> None:
    """★ 切分出的每个片段都必须通过证据层的三重一致性校验。

    这是两层之间的接缝：偏移算错、hash 不同源，只有这个测试能发现。
    """
    store = InMemoryDocStore([doc_a, doc_b])
    for doc in (doc_a, doc_b):
        for chunk in chunk_document(doc, max_chars=40, overlap_sentences=1):
            check = verify_span(chunk.to_span(), store)
            assert check.ok, f"片段未通过校验：{check.problem.value} — {check.detail}"


def test_chunks_detect_document_replacement(doc_a: Document) -> None:
    """按片段索引后文档被替换 —— 必须能发现（靠 hash 而非文本比对）。"""
    chunks = chunk_document(doc_a, max_chars=40)
    tampered = InMemoryDocStore([Document.create("doc-a", "内容被整体替换。" * 10)])
    problems = {verify_span(c.to_span(), tampered).problem for c in chunks}
    assert problems == {SpanProblem.HASH_MISMATCH}


# ─────────────────────── 切分 ───────────────────────


def test_chunk_offsets_are_exact(doc_a: Document) -> None:
    for c in chunk_document(doc_a, max_chars=40):
        assert doc_a.content[c.char_start : c.char_end] == c.text


def test_chunk_respects_max_chars(doc_a: Document) -> None:
    for c in chunk_document(doc_a, max_chars=25, overlap_sentences=0):
        # 单句可能超过上限，此时允许该句独占一个片段
        assert len(c.text) <= 25 or "。" not in c.text


def test_chunk_overlap_terminates(doc_a: Document) -> None:
    """重叠必须仍然前进，否则会死循环 —— 这是最容易写错的地方。"""
    chunks = chunk_document(doc_a, max_chars=15, overlap_sentences=2)
    assert 0 < len(chunks) <= len(TEXT_A)  # 有界即可证明没有无限循环


def test_chunk_rejects_bad_params(doc_a: Document) -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_document(doc_a, max_chars=0)
    with pytest.raises(ValueError, match="overlap_sentences"):
        chunk_document(doc_a, overlap_sentences=-1)


# ─────────────────────── 聚合 ───────────────────────


def test_best_per_document_keeps_highest_score() -> None:
    c1 = Chunk("d1", "c1", 0, 1, "a", "h")
    c2 = Chunk("d1", "c2", 1, 2, "b", "h")
    c3 = Chunk("d2", "c3", 0, 1, "c", "h")
    out = best_per_document(
        [ScoredChunk(c1, 0.2), ScoredChunk(c2, 0.9), ScoredChunk(c3, 0.5)]
    )
    assert [r.chunk.doc_id for r in out] == ["d1", "d2"]
    assert out[0].score == 0.9


def test_rank_documents_truncates() -> None:
    cs = [Chunk(f"d{i}", f"c{i}", 0, 1, "x", "h") for i in range(5)]
    out = rank_documents([ScoredChunk(c, float(i)) for i, c in enumerate(cs)], k=2)
    assert out == ["d4", "d3"]


# ─────────────────────── 指标 ───────────────────────


def test_recall_at_k_basic() -> None:
    assert recall_at_k(["a", "b", "c"], {"a"}, 1) == 1.0
    assert recall_at_k(["b", "c"], {"a"}, 2) == 0.0
    assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0
    assert recall_at_k(["a"], {"a", "b"}, 1) == 0.5


def test_precision_and_mrr() -> None:
    assert precision_at_k(["a", "x"], {"a"}, 2) == 0.5
    assert mrr_at_k(["x", "a"], {"a"}, 10) == 0.5
    assert mrr_at_k(["x", "y"], {"a"}, 10) == 0.0


def test_ndcg_perfect_and_zero() -> None:
    assert ndcg_at_k(["a"], {"a"}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x"], {"a"}, 10) == 0.0


def test_ndcg_penalises_lower_rank() -> None:
    """同一命中，排第一应优于排第二 —— 否则 nDCG 没有意义。"""
    assert ndcg_at_k(["a", "x"], {"a"}, 10) > ndcg_at_k(["x", "a"], {"a"}, 10)


def test_metrics_handle_empty_gold() -> None:
    assert recall_at_k(["a"], set(), 1) == 0.0
    assert ndcg_at_k(["a"], set(), 1) == 0.0


# ─────────────────────── 统计 ───────────────────────


def test_bootstrap_ci_is_deterministic() -> None:
    """固定种子：同一份数据必须给出同样的区间，否则不可复现。"""
    vals = [0.1, 0.4, 0.7, 0.9, 0.3]
    assert bootstrap_ci(vals) == bootstrap_ci(vals)


def test_bootstrap_ci_brackets_mean_and_narrows_with_n() -> None:
    small = [1.0, 0.0, 1.0, 0.0]
    big = small * 25
    lo_s, hi_s = bootstrap_ci(small)
    lo_b, hi_b = bootstrap_ci(big)
    assert lo_s <= sum(small) / len(small) <= hi_s
    assert (hi_b - lo_b) < (hi_s - lo_s), "样本量越大，区间应越窄"


def test_bootstrap_ci_single_value_degenerates() -> None:
    assert bootstrap_ci([0.5]) == (0.5, 0.5)


# ─────────────────────── 检索器 ───────────────────────


def test_bm25_ranks_relevant_chunk_first(chunks: list[Chunk]) -> None:
    r = Bm25Retriever(chunks)
    hits = r.retrieve("准确率提升了 5%", k=3)
    assert hits, "应至少有一个命中"
    assert "准确率" in hits[0].chunk.text


def test_bm25_empty_query_returns_nothing(chunks: list[Chunk]) -> None:
    assert Bm25Retriever(chunks).retrieve("   ", k=5) == []


def test_bm25_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="chunks"):
        Bm25Retriever([])


def test_bm25_works_on_single_document_corpus() -> None:
    """★ 回归测试：单文档语料下检索曾**静默返回空**。

    根因：``BM25Okapi`` 的 IDF 为 ``log((N - df + 0.5) / (df + 0.5))``，
    在 N=1、df=1 时等于 ``log(1/3) ≈ -1.10`` —— **负数**。
    于是所有分数 ≤ 0，被 ``score <= 0`` 的过滤条件全部丢弃，
    表现为"检索不到任何东西"，且不报任何错。

    改用 ``BM25Plus``（IDF 为 ``log((N+1)/df)``，恒非负）后正常。
    这个用例同时守住两件事：小语料可用 + 不再静默失败。
    """
    doc = Document.create("only", "The retrieval method improved accuracy by 5%.")
    chunks = chunk_document(doc, max_chars=200)
    hits = Bm25Retriever(chunks).retrieve("retrieval accuracy", k=5)
    assert hits, "单文档语料也必须能检索到（曾经静默返回空）"
    assert hits[0].score > 0


def test_trigram_retriever_finds_match(chunks: list[Chunk]) -> None:
    hits = CharTrigramRetriever(chunks).retrieve("点击率预估", k=3)
    assert hits and "点击率" in hits[0].chunk.text


def test_rrf_rewards_agreement() -> None:
    """两个检索器都把 d1 排第一时，d1 的融合分应高于只被一方看好的 d2。"""
    fused = rrf([["d1", "d2"], ["d1", "d3"]])
    assert fused["d1"] > fused["d2"]
    assert fused["d1"] > fused["d3"]


def test_rrf_fusion_returns_document_level(chunks: list[Chunk]) -> None:
    fused = RrfFusion([Bm25Retriever(chunks), CharTrigramRetriever(chunks)])
    hits = fused.retrieve("检索方法", k=5)
    doc_ids = [h.chunk.doc_id for h in hits]
    assert len(doc_ids) == len(set(doc_ids)), "融合应在文档级去重"


# ─────────────────────── 评测执行 ───────────────────────


def test_evaluate_retriever_end_to_end(chunks: list[Chunk]) -> None:
    queries = [
        EvalQuery("q1", "准确率提升了 5%", frozenset({"doc-a"})),
        EvalQuery("q2", "点击率预估", frozenset({"doc-b"})),
    ]
    report = evaluate_retriever(Bm25Retriever(chunks), queries, ks=(1, 2), depth=4)
    assert len(report.queries) == 2
    summary = report.summary()
    assert summary["recall@1"]["mean"] > 0.0
    assert "lo" in summary["recall@1"], "默认应带置信区间"
