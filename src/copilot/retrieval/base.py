"""检索层的核心类型。

与 P1 的集成点
--------------
:class:`Chunk` 携带 ``doc_id / char_start / char_end / text / source_hash``，
与 :class:`~copilot.evidence.models.EvidenceSpan` **字段完全一致**，
因此可以直接 ``to_span()`` 交给证据校验。

这意味着检索出来的片段天生就是"证据候选" —— 不需要在两层之间做转换，
也不会因为转换时重算 hash 而产生不一致。文档对象统一复用 P1 的
:class:`~copilot.evidence.models.Document`，保证 ``content_hash`` 算法同源。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from copilot.evidence.models import EvidenceSpan

__all__ = ["Chunk", "ScoredChunk", "Retriever", "rank_documents", "best_per_document"]


@dataclass(frozen=True)
class Chunk:
    """一段可检索的文本，带精确偏移与来源 hash。"""

    doc_id: str
    chunk_id: str
    char_start: int
    char_end: int
    text: str
    source_hash: str

    def __post_init__(self) -> None:
        if self.char_end - self.char_start != len(self.text):
            raise ValueError(
                "偏移跨度必须等于文本长度："
                f"{self.char_end - self.char_start} != {len(self.text)}"
            )

    def to_span(self) -> EvidenceSpan:
        """转成证据片段 —— 这是检索层与证据层的接缝。"""
        return EvidenceSpan(
            doc_id=self.doc_id,
            source_hash=self.source_hash,
            char_start=self.char_start,
            char_end=self.char_end,
            text=self.text,
        )


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    """检索器接口。不同实现互换即可做消融对照。"""

    name: str

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]: ...


def best_per_document(results: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """每个文档只保留得分最高的那个片段（max-pool）。

    文档级排序的标准做法：一篇论文只要有**任意一段**高度相关，
    就应该排在前面，而不是看它的平均相关度。
    """
    best: dict[str, ScoredChunk] = {}
    for r in results:
        cur = best.get(r.chunk.doc_id)
        if cur is None or r.score > cur.score:
            best[r.chunk.doc_id] = r
    return sorted(best.values(), key=lambda x: x.score, reverse=True)


def rank_documents(results: Sequence[ScoredChunk], k: int | None = None) -> list[str]:
    """把片段级结果聚合成文档级排名（供 Recall@k / MRR / nDCG 使用）。"""
    docs = [r.chunk.doc_id for r in best_per_document(results)]
    return docs[:k] if k is not None else docs
