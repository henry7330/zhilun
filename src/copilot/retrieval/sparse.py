"""稀疏检索器与融合/重排 —— 全部确定性，不依赖嵌入模型。

本模块刻意只实现**无需模型**的部分，原因有二：

1. 它们是**基线**。没有基线，"接入嵌入模型提升了多少"这个问题无法回答；
2. 它们的可复现性最强（无随机性、无模型版本漂移），适合作为回归门槛。

真正的语义向量召回（dense）是独立的一层，接入后与这里做 RRF 融合即可，
接口已经预留（:class:`~copilot.retrieval.base.Retriever`）。
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence

from rank_bm25 import BM25Plus

from copilot.evidence.text import coverage, tokenize
from copilot.retrieval.base import Chunk, Retriever, ScoredChunk, best_per_document

__all__ = [
    "Bm25Retriever",
    "CharTrigramRetriever",
    "RrfFusion",
    "CoverageReranker",
    "rrf",
]


# ─────────────────────────── BM25 ───────────────────────────


class Bm25Retriever:
    """BM25 检索器。

    用 :func:`~copilot.evidence.text.tokenize` 作分词器：中文取二字组、
    西文取词。这样做的好处是**与证据层的口径一致** ——
    若检索用一个口径、覆盖率评分用另一个口径，会出现"检索到了但判为不支持"
    的诡异现象，且极难排查。

    ⚠️ **用 ``BM25Plus`` 而不是 ``BM25Okapi``**（实测发现的问题）：
    ``BM25Okapi`` 的 IDF 是 ``log((N - df + 0.5) / (df + 0.5))``，
    在**单文档语料**（N=1, df=1）下等于 ``log(1/3) ≈ -1.10``，即**负数**。
    于是所有分数都 ≤ 0，被下方的过滤条件全部丢弃 —— 表现为
    "检索静默返回空"，极难排查。

    ``BM25Plus`` 的 IDF 是 ``log((N + 1) / df)``，恒为非负，
    在小语料下也不会出现这个病态行为。
    """

    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("chunks 不能为空")
        self._chunks = list(chunks)
        corpus = [tokenize(c.text) for c in self._chunks]
        # 空文档会出错；用一个不会命中的占位 token 兜底
        corpus = [t if t else ["\x00empty"] for t in corpus]
        self._bm25 = BM25Plus(corpus, k1=k1, b=b)

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[ScoredChunk] = []
        for i in ranked[:k]:
            score = float(scores[i])
            if score <= 0:  # 0 表示与查询没有任何词面重叠，确实不相关
                continue
            out.append(ScoredChunk(self._chunks[i], score))
        return out


# ───────────────────── 字符 n-gram 余弦 ─────────────────────

_WS = re.compile(r"\s+")


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    t = _WS.sub("", unicodedata.normalize("NFKC", text or "").casefold())
    if not t:
        return []
    if len(t) < n:
        return [t]
    return [t[i : i + n] for i in range(len(t) - n + 1)]


class CharTrigramRetriever:
    """字符 trigram + TF-IDF 余弦。

    与 BM25 的信号**粒度不同**：BM25 用"词/二元组"，这里用滑窗三元组。
    好处是对分词边界、语序、以及少量错字更鲁棒 —— 这也是它能与 BM25
    互补、值得做融合的原因。全部用 numpy 手写，不引入 sklearn。
    """

    name = "char3"

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("chunks 不能为空")
        self._chunks = list(chunks)
        docs = [_char_ngrams(c.text) for c in self._chunks]

        df: Counter[str] = Counter()
        for toks in docs:
            df.update(set(toks))
        n_docs = len(docs)
        # 平滑 IDF，避免除零
        self._idf = {t: math.log((n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._vecs = [self._vectorize(toks) for toks in docs]

    def _vectorize(self, tokens: Iterable[str]) -> dict[str, float]:
        tf = Counter(tokens)
        if not tf:
            return {}
        vec = {t: (1.0 + math.log(c)) * self._idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm else {}

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        qv = self._vectorize(_char_ngrams(query))
        if not qv:
            return []
        scored: list[ScoredChunk] = []
        for chunk, dv in zip(self._chunks, self._vecs, strict=True):
            if not dv:
                continue
            # 两个向量都已 L2 归一化，点积即余弦
            sim = sum(w * dv.get(t, 0.0) for t, w in qv.items())
            if sim > 0:
                scored.append(ScoredChunk(chunk, sim))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]


# ─────────────────────────── 融合 ───────────────────────────


def rrf(rankings: Sequence[Sequence[str]], *, k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion：``score(d) = Σ 1/(k + rank_i(d))``。

    用**排名**而非原始分数融合，是因为不同检索器的分数量纲不可比
    （BM25 是无界正数，余弦在 [0,1]）。直接加权求和需要调归一化，
    而 RRF 只需排名，稳健且无需调参 —— 这是它在实践中流行的原因。

    ``k=60`` 是原论文的推荐值，作用是压低头部排名的绝对优势。
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class RrfFusion:
    """把多个检索器的结果按 RRF 融合（文档级）。"""

    def __init__(self, retrievers: Sequence[Retriever], *, k: int = 60, name: str = "rrf") -> None:
        if not retrievers:
            raise ValueError("至少需要一个检索器")
        self._retrievers = list(retrievers)
        self._k = k
        self.name = name

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        per_retriever: list[list[ScoredChunk]] = []
        for r in self._retrievers:
            per_retriever.append(r.retrieve(query, k=k * 5))

        rankings: list[list[str]] = []
        representative: dict[str, ScoredChunk] = {}
        for results in per_retriever:
            docs = best_per_document(results)
            rankings.append([r.chunk.doc_id for r in docs])
            for r in results:
                representative.setdefault(r.chunk.doc_id, r)

        fused = rrf(rankings, k=self._k)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [ScoredChunk(representative[d].chunk, s) for d, s in ordered if d in representative]


# ─────────────────────────── 重排 ───────────────────────────


class CoverageReranker:
    """用「查询词覆盖率」对候选重排。

    与 BM25 的信号不同：BM25 看重词频与长度归一，覆盖率只看
    **查询里的内容单元有多少出现在片段里**，对"长片段堆砌关键词"
    这种作弊式命中更不敏感。因此它能纠正 BM25 的一部分误排。

    只对前 ``depth`` 个候选重排 —— 重排的价值在于调整头部，
    对长尾排序既无收益也徒增开销。
    """

    def __init__(self, base: Retriever, *, depth: int = 20, name: str | None = None) -> None:
        self._base = base
        self._depth = depth
        self.name = name or f"{base.name}+cov"

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        candidates = self._base.retrieve(query, k=self._depth)
        if not candidates:
            return []

        # 排名分与覆盖率归一到同一量纲后相加：排名分用 RRF 形式，
        # 覆盖率本身就在 [0,1]，无需再放缩。
        ranked = best_per_document(candidates)
        rank_score = rrf([[r.chunk.doc_id for r in ranked]])
        max_rank = max(rank_score.values()) if rank_score else 1.0

        rescored: list[ScoredChunk] = []
        best_chunk_per_doc: dict[str, ScoredChunk] = {}
        for r in candidates:
            cov = coverage(query, r.chunk.text)
            blended = 0.5 * (rank_score.get(r.chunk.doc_id, 0.0) / max_rank) + 0.5 * cov
            rescored.append(ScoredChunk(r.chunk, blended))
            cur = best_chunk_per_doc.get(r.chunk.doc_id)
            if cur is None or r.score > cur.score:
                best_chunk_per_doc[r.chunk.doc_id] = r

        rescored.sort(key=lambda s: s.score, reverse=True)
        return rescored[:k]
