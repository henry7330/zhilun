"""流水线节点：每个节点是一个纯函数式的状态变换。

LLM 无关性是这个设计的要点
--------------------------
本模块**一个模型调用都没有**。检索、原子化、证据绑定、校验、支持判断、
记忆门禁全部来自 P1/P2/P3 的确定性实现。

这不是权宜之计，而是刻意验证一个主张：**可验证性的骨架与"谁生成答案"无关。**
将来把 ``claim_source`` 换成 LLM 生成，下游的绑定与校验链路一行都不用改 ——
这正是"防幻觉的很大一部分可以做成确定性"的具体形态。

断言从哪来
----------
节点不假设断言的来源。:class:`PipelineDeps` 里的 ``claim_source`` 是一个可替换的
可调用对象：

- 默认实现：从检索到的片段里**抽取**相关句子（抽取式问答）；
- 生产实现：调用 LLM 生成；
- 测试实现：注入一条**没有依据**的断言，用来验证下游真的能拦住它。

把"来源"抽成接缝，才能既在没有模型时跑通全链路，又在有模型时无缝替换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from copilot.evidence.bind import EvidenceBinder, RuleBinder
from copilot.evidence.extract import RuleBasedExtractor, split_clauses, split_sentences
from copilot.evidence.models import (
    CitationMeta,
    Claim,
    DocStore,
    Document,
    DoiResolver,
    EvidenceSpan,
    SupportStatus,
)
from copilot.evidence.support import RuleBasedJudge, SupportJudge
from copilot.evidence.text import content_tokens
from copilot.evidence.verify import CitationLibrary, VerdictKind, verify_claim
from copilot.memory.engine import MemoryEngine
from copilot.memory.models import MemoryLayer, TrustLevel
from copilot.retrieval.base import Retriever, best_per_document
from copilot.textseg import iter_segments

__all__ = ["PipelineDeps", "PipelineNodes", "ClaimSource", "extractive_claim_source"]


# ─────────────────────── 断言来源（可替换接缝）───────────────────────


class ClaimSource(Protocol):
    """把一个问题和检索结果变成候选断言。"""

    def __call__(
        self, question: str, retrieved: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]: ...


def extractive_claim_source(
    question: str,
    retrieved: Sequence[dict[str, Any]],
    *,
    max_claims: int = 5,
    min_overlap: float = 0.15,
) -> list[dict[str, Any]]:
    """默认实现：从检索片段里抽出与问题相关的句子作为断言。

    这是**抽取式**而非生成式 —— 断言直接来自原文，因此天然可溯源。
    它的价值在于：没有模型也能把整条验证链路跑通、并给出一份
    "有证据支撑的答案"。

    ``min_overlap`` 过滤掉与问题几乎无关的句子，避免产出大量噪声断言
    （断言太碎会让指标虚高，也会淹没真正重要的结论）。
    """
    q_tokens = content_tokens(question)
    if not q_tokens:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for item in retrieved:
        doc_text = item["text"]
        for start, _end, sentence in iter_segments(doc_text):
            if len(sentence) < 8:
                continue
            overlap = len(q_tokens & content_tokens(sentence)) / len(q_tokens)
            if overlap < min_overlap:
                continue
            scored.append(
                (
                    overlap,
                    {
                        "text": sentence,
                        "doc_id": item["doc_id"],
                        "source_hash": item["source_hash"],
                        "char_start": item["char_start"] + start,
                        "char_end": item["char_start"] + start + len(sentence),
                    },
                )
            )

    scored.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    claims: list[dict[str, Any]] = []
    for _score, payload in scored:
        if payload["text"] in seen:
            continue
        seen.add(payload["text"])
        claims.append(payload)
        if len(claims) >= max_claims:
            break
    return claims


# ─────────────────────── 依赖 ───────────────────────


@dataclass
class PipelineDeps:
    """流水线的外部依赖。集中注入，便于测试时替换。"""

    retriever: Retriever
    documents: list[Document]
    doc_store: DocStore
    memory: MemoryEngine
    library: CitationLibrary
    resolver: DoiResolver
    binder: EvidenceBinder = field(default_factory=RuleBinder)
    judge: SupportJudge = field(default_factory=RuleBasedJudge)
    claim_source: Callable[..., list[dict[str, Any]]] = extractive_claim_source
    extractor: RuleBasedExtractor = field(default_factory=RuleBasedExtractor)
    retrieve_k: int = 6
    #: 出现「编造」级结论时，是否要求人工审批后才给出答案
    require_approval_on_fabrication: bool = True


# ─────────────────────── 节点 ───────────────────────


class PipelineNodes:
    """节点集合。实例方法即可直接作为 LangGraph 节点使用。"""

    def __init__(self, deps: PipelineDeps) -> None:
        self.deps = deps

    # ── 规划 ──

    def plan(self, state: dict[str, Any]) -> dict[str, Any]:
        """把问题拆成子问题。

        规则版按句子与连接词切分 —— 简单、可复现、零成本。
        它的作用不是「聪明的规划」，而是让状态里有一个显式的 ``plan`` 字段，
        为将来替换为模型规划留好位置（届时只需换掉这个节点）。
        """
        from copilot.graph.state import SUBQUESTION_LIMIT

        question = state.get("question", "").strip()
        if not question:
            return {"plan": [], "errors": ["问题为空"]}

        sub: list[str] = []
        for part in split_sentences(question) or [question]:
            sub.extend(split_clauses(part) or [part])
        sub = [s for s in sub if len(s) >= 2][:SUBQUESTION_LIMIT]
        return {"plan": sub or [question]}

    # ── 检索 ──

    def retrieve(self, state: dict[str, Any]) -> dict[str, Any]:
        """对每个子问题检索，合并去重后按分数排序。"""
        plan = state.get("plan") or [state.get("question", "")]
        collected: dict[str, Any] = {}

        for sub in plan:
            for hit in self.deps.retriever.retrieve(sub, k=self.deps.retrieve_k):
                chunk = hit.chunk
                prev = collected.get(chunk.chunk_id)
                if prev is None or hit.score > prev["score"]:
                    collected[chunk.chunk_id] = {
                        "doc_id": chunk.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "text": chunk.text,
                        "source_hash": chunk.source_hash,
                        "score": float(hit.score),
                    }

        retrieved = sorted(collected.values(), key=lambda d: d["score"], reverse=True)
        update: dict[str, Any] = {"retrieved": retrieved}
        if not retrieved:
            update["errors"] = ["检索未命中任何片段"]
        return update

    # ── 断言抽取（可替换为 LLM）──

    def extract(self, state: dict[str, Any]) -> dict[str, Any]:
        """把候选断言草稿规范成 claim dict。

        草稿支持两种形态，用 ``evidence`` 字段区分：

        1. **无 evidence 字段**：从片段抽取而来 —— 断言文本**就是**原文，
           于是用扁平字段（doc_id/char_start/...）构造一个 span。
        2. **带 evidence 字段**：断言文本与证据文本**不同** ——
           例如模型把结论写反了（断言说"下降"，原文说"提升"）。

        两种形态必须都能表达，否则就没法测试"结论与证据矛盾"这个最关键的场景。
        注意 span 的 ``text`` 必须与其偏移一致（``EvidenceSpan`` 会强制校验），
        因此证据文本**不能**被断言文本覆盖。
        """
        question = state.get("question", "")
        retrieved = state.get("retrieved", [])
        drafts = self.deps.claim_source(question, retrieved)

        claims: list[dict[str, Any]] = []
        for i, d in enumerate(drafts, start=1):
            explicit = d.get("evidence")
            if explicit:
                evidence = [dict(s) for s in explicit]
            else:
                evidence = [
                    {
                        "doc_id": d["doc_id"],
                        "source_hash": d["source_hash"],
                        "char_start": d["char_start"],
                        "char_end": d["char_end"],
                        "text": d["text"],
                    }
                ]
            claims.append(
                {
                    "claim_id": f"c{i}",
                    "text": d["text"],
                    "evidence": evidence,
                    "citations": list(d.get("citations", [])),
                    "confidence": float(d.get("confidence", 0.8)),
                }
            )
        return {"claims": claims}

    # ── 校验 ──

    def verify(self, state: dict[str, Any]) -> dict[str, Any]:
        """对每条断言做：证据绑定（若缺）→ 确定性校验 → 支持判断。"""
        verdicts: list[dict[str, Any]] = []
        risks: list[str] = []

        for raw in state.get("claims", []):
            claim = _claim_from_dict(raw)

            # 缺证据的断言先尝试自动绑定 —— 这是 P1 的检索式绑定
            if not claim.evidence:
                self.deps.binder.bind(claim.text, self.deps.documents)

            det = verify_claim(
                claim, self.deps.doc_store, self.deps.library, self.deps.resolver
            )
            judgment = self.deps.judge.judge(claim)

            status = (
                det.status if det.kind is not VerdictKind.NEEDS_SEMANTIC else judgment.status
            )
            reasons = list(det.reasons) + list(judgment.reasons)

            verdicts.append(
                {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "status": status.value,
                    "deterministic_kind": det.kind.value,
                    "coverage": judgment.signals.coverage,
                    "reasons": reasons,
                }
            )
            if status is SupportStatus.FABRICATED:
                risks.append(f"{claim.claim_id} 被判为编造/矛盾：{claim.text[:40]}")

        update: dict[str, Any] = {"verdicts": verdicts, "risks": risks}
        if self.deps.require_approval_on_fabrication and risks:
            update["needs_approval"] = True
        return update

    # ── 人工审批 ──

    def approval(self, state: dict[str, Any]) -> dict[str, Any]:
        """高风险结论的人工审批闸门。

        ``interrupt`` 会**暂停整个图**并把 payload 抛给调用方；
        调用方用 ``Command(resume=...)`` 恢复后，``interrupt()`` 返回该值、
        节点从这里继续。这正是 checkpoint 存在的意义 ——
        暂停期间进程可以退出，之后从磁盘恢复。
        """
        from langgraph.types import interrupt

        payload = {
            "question": state.get("question", ""),
            "risks": state.get("risks", []),
            "verdicts": [
                {"claim_id": v["claim_id"], "status": v["status"], "text": v["text"][:60]}
                for v in state.get("verdicts", [])
            ],
            "prompt": "答案中存在被判定为编造/矛盾的结论，是否仍然输出？",
        }
        decision = interrupt(payload)
        if not isinstance(decision, dict):
            decision = {"approved": bool(decision), "operator": "unknown", "note": ""}
        return {"approval": decision}

    # ── 记忆写入 ──

    def write_memory(self, state: dict[str, Any]) -> dict[str, Any]:
        """把结论写进记忆，走 P3 的写入门禁。

        关键：来自检索文献的结论按 ``UNTRUSTED_EXTERNAL`` 写入 ——
        即使它已被证据支撑，也要走门禁流程，而不是直接当可信记忆。
        """
        question = state.get("question", "")
        records: list[dict[str, Any]] = []

        for verdict in state.get("verdicts", []):
            if verdict["status"] not in (
                SupportStatus.SUPPORTED.value,
                SupportStatus.PARTIALLY_SUPPORTED.value,
            ):
                continue
            raw_claim = next(
                (c for c in state.get("claims", []) if c["claim_id"] == verdict["claim_id"]),
                None,
            )
            evidence = _spans_from_dicts((raw_claim or {}).get("evidence", []))
            result = self.deps.memory.observe(
                verdict["text"],
                layer=MemoryLayer.SEMANTIC,
                trust=TrustLevel.UNTRUSTED_EXTERNAL,
                evidence=evidence,
                source_uri=f"session:{state.get('session_id', 'default')}",
            )
            records.append(
                {
                    "content": verdict["text"],
                    "decision": result.gate.decision.value,
                    "reason": result.gate.reason,
                    "stored": result.stored,
                    "quarantined": bool(result.stored and result.item.quarantined),
                }
            )

        if question:
            # 问题本身是用户亲述，作为偏好/情景记忆低风险写入
            self.deps.memory.observe(
                f"用户提问：{question}",
                layer=MemoryLayer.EPISODIC,
                trust=TrustLevel.TRUSTED_USER,
            )
        return {"memories": records}

    # ── 输出 ──

    def answer(self, state: dict[str, Any]) -> dict[str, Any]:
        """组装结构化答案。**不隐藏被拦下的结论。**"""
        verdicts = state.get("verdicts", [])
        approval = state.get("approval")

        supported = [v for v in verdicts if v["status"] == SupportStatus.SUPPORTED.value]
        partial = [
            v for v in verdicts if v["status"] == SupportStatus.PARTIALLY_SUPPORTED.value
        ]
        fabricated = [v for v in verdicts if v["status"] == SupportStatus.FABRICATED.value]
        unsupported = [v for v in verdicts if v["status"] == SupportStatus.UNSUPPORTED.value]

        blocked = bool(fabricated) and not (approval or {}).get("approved", False)

        return {
            "answer": {
                "question": state.get("question", ""),
                "plan": state.get("plan", []),
                "claims": supported + partial,
                "fabricated": fabricated,
                "unsupported": unsupported,
                "blocked": blocked,
                "approval": approval,
                "memory_writes": state.get("memories", []),
                "risks": state.get("risks", []),
                "sources": sorted({r["doc_id"] for r in state.get("retrieved", [])}),
            }
        }

    # ── 路由 ──

    @staticmethod
    def route_after_verify(state: dict[str, Any]) -> str:
        """校验之后：需要审批就走审批，否则直接写记忆。"""
        return "approval" if state.get("needs_approval") else "write_memory"

    @staticmethod
    def route_after_approval(state: dict[str, Any]) -> str:
        """审批之后：批准了才写记忆，否则直接出答案（并标记 blocked）。"""
        approved = bool((state.get("approval") or {}).get("approved"))
        return "write_memory" if approved else "answer"


# ─────────────────────── 转换工具 ───────────────────────


def _spans_from_dicts(items: Sequence[dict[str, Any]]) -> list[EvidenceSpan]:
    return [
        EvidenceSpan(
            doc_id=s["doc_id"],
            source_hash=s["source_hash"],
            char_start=int(s["char_start"]),
            char_end=int(s["char_end"]),
            text=s["text"],
        )
        for s in items
    ]


def _claim_from_dict(raw: dict[str, Any]) -> Claim:
    citations = [
        CitationMeta(
            doi=c.get("doi"),
            title=c.get("title"),
            authors=tuple(c.get("authors", ())),
            year=c.get("year"),
        )
        for c in raw.get("citations", [])
    ]
    return Claim(
        claim_id=raw["claim_id"],
        text=raw["text"],
        evidence=_spans_from_dicts(raw.get("evidence", [])),
        citations=citations,
        confidence=float(raw.get("confidence", 0.0)),
    )
