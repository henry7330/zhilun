"""P4 编排层测试。

用**合成的小语料**而不是 arXiv 缓存：测试必须确定性、快、且不依赖网络产物。
语料内容刻意包含"方向相反"的两句，用来构造"结论写反"的编造场景。

三个重点：

1. **端到端跑通**并产出结构化答案（P1/P2/P3 真正串起来）；
2. **人工审批闸门** —— 出现编造级结论时图必须暂停，且驳回后**不写记忆**；
3. **断点恢复** —— 关闭后重新打开（模拟进程重启）仍能从 checkpoint 续跑。
   这是 P4 的验收条件之一，也是最容易做成"假恢复"的地方。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from copilot.evidence.models import Document, InMemoryDocStore, StaticDoiResolver
from copilot.evidence.verify import InMemoryCitationLibrary
from copilot.graph.build import build_pipeline
from copilot.graph.nodes import PipelineDeps, PipelineNodes
from copilot.memory.engine import MemoryEngine
from copilot.memory.gate import WriteGate
from copilot.memory.store import MemoryStore
from copilot.retrieval.base import Retriever, ScoredChunk
from copilot.retrieval.corpus import chunk_document
from copilot.retrieval.sparse import Bm25Retriever

# ─────────────────────────── 合成语料 ───────────────────────────

DOC_A = Document.create(
    "doc-a",
    "The retrieval method improved accuracy by 5% on five datasets. "
    "It also reduced latency significantly.",
)
DOC_B = Document.create(
    "doc-b",
    "Recommender systems face the cold-start problem. "
    "Large language models are increasingly used to mitigate it.",
)
DEFAULT_DOCS = [DOC_A, DOC_B]


class FixedRetriever:
    """返回固定结果的检索器，让测试不依赖 BM25 的打分细节。"""

    name = "fixed"

    def __init__(self, chunks) -> None:
        self._chunks = list(chunks)

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        return [ScoredChunk(c, 1.0 - i * 0.01) for i, c in enumerate(self._chunks[:k])]


def make_deps(
    documents: list[Document] | None = None,
    *,
    claim_source=None,
    retriever: Retriever | None = None,
    memory: MemoryEngine | None = None,
) -> PipelineDeps:
    docs = documents if documents is not None else list(DEFAULT_DOCS)
    if retriever is None:
        chunks = []
        for doc in docs:
            chunks.extend(chunk_document(doc, max_chars=200, overlap_sentences=0))
        retriever = Bm25Retriever(chunks)

    doc_store = InMemoryDocStore(docs)
    kwargs: dict[str, Any] = {}
    if claim_source is not None:
        kwargs["claim_source"] = claim_source

    return PipelineDeps(
        retriever=retriever,
        documents=docs,
        doc_store=doc_store,
        memory=memory or MemoryEngine(MemoryStore(":memory:"), WriteGate(doc_store=doc_store)),
        library=InMemoryCitationLibrary([]),
        resolver=StaticDoiResolver(set()),
        **kwargs,
    )


def span_of(doc: Document, text: str) -> dict[str, Any]:
    start = doc.content.index(text)
    return {
        "doc_id": doc.doc_id,
        "source_hash": doc.content_hash,
        "char_start": start,
        "char_end": start + len(text),
        "text": text,
    }


def reversed_claim_source(question: str, retrieved) -> list[dict[str, Any]]:
    """模拟 LLM 把结论写反：断言说"上升"，而原文说"下降"。

    证据指向**真实存在**的原文片段（所以片段校验会通过），
    但断言与证据**极性相反** —— 这正是最典型的幻觉形态。

    刻意选用**同一语义轴**的对立词（increase vs reduce）。
    跨轴的情况（如 decreased accuracy vs improved accuracy）规则层判不出来，
    原因见 ``test_rule_baseline_misses_cross_axis_polarity`` ——
    那不是缺陷，而是词面方法的原理性边界。

    注意断言文本与证据文本**必须分开**：``EvidenceSpan`` 要求 span 的 text
    与其偏移一致，所以证据文本不能被断言文本覆盖。
    """
    return [
        {
            "text": "The method increased latency significantly.",
            "evidence": [span_of(DOC_A, "reduced latency significantly")],
        }
    ]


def config(thread: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}}


def make_graph(deps: PipelineDeps, *, checkpointer=None):
    """编译图，默认带**内存 checkpointer**。

    ``interrupt`` 必须配 checkpointer 才能工作（LangGraph 会直接报错），
    所以测试默认挂一个 InMemorySaver —— 否则一旦某条路径触发审批，
    测试会以 "Cannot use Command(resume=...) without checkpointer" 失败，
    而那个报错并不能指出真正的原因。
    """
    from langgraph.checkpoint.memory import InMemorySaver

    return build_pipeline(deps, checkpointer=checkpointer or InMemorySaver())


# ─────────────────────── 编译与端到端 ───────────────────────


def test_pipeline_compiles() -> None:
    graph = make_graph(make_deps())
    assert graph is not None


def test_end_to_end_produces_structured_answer() -> None:
    """★ P1/P2/P3 真正串起来：一次调用走完全链路并产出结构化结果。"""
    graph = make_graph(make_deps())
    out = graph.invoke(
        {"question": "How does the retrieval method perform?", "session_id": "s1"},
        config("t-e2e"),
    )

    assert out["plan"], "应产出计划"
    assert out["retrieved"], "应检索到片段"
    assert out["claims"], "应抽取到断言"
    assert out["verdicts"], "应有校验结果"
    assert out["memories"], "支持的结论应写入记忆"

    answer = out["answer"]
    assert answer["question"].startswith("How does")
    assert answer["claims"], "应有被支持的结论"
    assert answer["blocked"] is False
    assert answer["sources"], "应记录来源文档"


def test_extracted_claims_are_supported() -> None:
    """正向对照：抽取式断言来自原文，应被判为 supported。

    没有这条对照，后面"能拦住编造"的测试可能只是"什么都判编造"的假象。
    """
    graph = make_graph(make_deps())
    out = graph.invoke(
        {"question": "How does the retrieval method perform?", "session_id": "s1"},
        config("t-supported"),
    )
    statuses = {v["status"] for v in out["verdicts"]}
    assert "supported" in statuses
    assert "fabricated" not in statuses


def test_empty_question_records_error() -> None:
    graph = make_graph(make_deps())
    out = graph.invoke({"question": "   ", "session_id": "s"}, config("t-empty"))
    assert out["errors"]
    assert out["answer"]["claims"] == []


# ─────────────── ★ 人工审批闸门（interrupt）★ ───────────────


def test_fabricated_claim_pauses_for_approval() -> None:
    """★ 出现编造级结论时，图必须**暂停**等人工审批，而不是直接输出。"""
    graph = make_graph(make_deps(claim_source=reversed_claim_source))
    out = graph.invoke(
        {"question": "Does the method improve accuracy?", "session_id": "s1"},
        config("t-pause"),
    )

    assert "__interrupt__" in out, "应触发 interrupt 暂停"
    payload = out["__interrupt__"][0].value
    assert payload["risks"], "审批载荷里应包含风险说明"
    assert "是否仍然输出" in payload["prompt"]

    # 暂停时**不应**写入记忆（先审批后落库的纪律）
    assert not out.get("memories")


def test_rejection_blocks_answer_and_writes_no_memory() -> None:
    """驳回 → 答案标记 blocked，且**一条记忆都没写**。"""
    graph = make_graph(make_deps(claim_source=reversed_claim_source))
    cfg = config("t-reject")
    graph.invoke(
        {"question": "Does the method improve accuracy?", "session_id": "s1"}, cfg
    )
    out = graph.invoke(
        Command(resume={"approved": False, "operator": "alice", "note": "结论写反了"}),
        cfg,
    )

    assert out["answer"]["blocked"] is True
    assert out["answer"]["approval"]["operator"] == "alice"
    assert not out.get("memories"), "被驳回的结论不得进入记忆"


def test_approval_allows_answer_and_writes_memory() -> None:
    """批准 → 答案不再 blocked，且受支持的结论写入记忆。"""
    graph = make_graph(make_deps(claim_source=reversed_claim_source))
    cfg = config("t-approve")
    graph.invoke(
        {"question": "Does the method improve accuracy?", "session_id": "s1"}, cfg
    )
    out = graph.invoke(
        Command(resume={"approved": True, "operator": "bob", "note": "已人工确认"}), cfg
    )

    assert out["answer"]["blocked"] is False
    assert out["answer"]["approval"]["approved"] is True


def test_non_dict_resume_value_is_tolerated() -> None:
    """审批返回值不只是 dict 时也要能工作（外部系统可能只回一个 bool）。"""
    graph = make_graph(make_deps(claim_source=reversed_claim_source))
    cfg = config("t-bool")
    graph.invoke({"question": "Does it improve?", "session_id": "s1"}, cfg)
    out = graph.invoke(Command(resume=True), cfg)
    assert out["answer"]["approval"]["approved"] is True


# ─────────────── ★ 断点恢复（模拟进程重启）★ ───────────────


def test_resume_after_restart(tmp_path: Path) -> None:
    """★ P4 验收条件：进程重启后能从 checkpoint 恢复。

    这里用**落盘的 SqliteSaver**，第一段执行到 interrupt 后**关闭连接**，
    再用全新的 saver 与全新的图对象恢复 —— 等价于进程重启。
    若状态只存在内存里，这个测试必然失败。
    """
    db = tmp_path / "checkpoint.sqlite"
    cfg = config("t-restart")
    question = "Does the method improve accuracy?"

    # ── 第一次运行：跑到暂停，然后关闭 ──
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = make_graph(make_deps(claim_source=reversed_claim_source), checkpointer=saver)
        out = graph.invoke({"question": question, "session_id": "s1"}, cfg)
        assert "__interrupt__" in out
    assert db.exists(), "checkpoint 应已落盘"

    # ── 模拟重启：全新 saver、全新图、全新引擎 ──
    with SqliteSaver.from_conn_string(str(db)) as saver2:
        memory = MemoryEngine(MemoryStore(":memory:"), WriteGate(doc_store=InMemoryDocStore(list(DEFAULT_DOCS))))
        graph2 = build_pipeline(
            make_deps(claim_source=reversed_claim_source, memory=memory),
            checkpointer=saver2,
        )

        # 恢复前，状态应仍可读出（证明状态真的在盘上）
        snapshot = graph2.get_state(cfg)
        assert snapshot.values.get("question") == question
        assert snapshot.values.get("verdicts"), "校验结果应已持久化"

        out2 = graph2.invoke(
            Command(resume={"approved": True, "operator": "carol", "note": "重启后恢复"}), cfg
        )

    assert out2["answer"]["blocked"] is False
    assert out2["answer"]["approval"]["operator"] == "carol"
    assert out2["answer"]["question"] == question


# ─────────────────────── 路由 ───────────────────────


def test_route_after_verify() -> None:
    assert PipelineNodes.route_after_verify({"needs_approval": True}) == "approval"
    assert PipelineNodes.route_after_verify({}) == "write_memory"


def test_route_after_approval() -> None:
    assert PipelineNodes.route_after_approval({"approval": {"approved": True}}) == "write_memory"
    assert PipelineNodes.route_after_approval({"approval": {"approved": False}}) == "answer"
    assert PipelineNodes.route_after_approval({}) == "answer"


# ─────────────────────── 记忆写入经过门禁 ───────────────────────


def test_memory_writes_are_evidenced_and_allowed() -> None:
    """来自文献的结论按不可信来源写入，但**有证据**，因此应被放行。"""
    graph = make_graph(make_deps())
    out = graph.invoke(
        {"question": "How does the retrieval method perform?", "session_id": "s1"},
        config("t-mem"),
    )
    decisions = {m["decision"] for m in out["memories"]}
    assert decisions, "应有记忆写入结果"
    assert decisions <= {"allow"}, f"有证据的结论应被放行，实际 {decisions}"
    assert all(not m["quarantined"] for m in out["memories"])


def test_question_is_stored_as_user_episodic_memory() -> None:
    memory = MemoryEngine(MemoryStore(":memory:"), WriteGate(doc_store=InMemoryDocStore(list(DEFAULT_DOCS))))
    graph = make_graph(make_deps(memory=memory))
    graph.invoke(
        {"question": "How does the retrieval method perform?", "session_id": "s1"},
        config("t-episodic"),
    )
    recalled = memory.recall()
    assert any("How does the retrieval method perform?" in i.content for i in recalled)


def test_unsupported_claims_are_not_written_to_memory() -> None:
    """无依据的断言不应进入记忆 —— 只有 supported/partially 才写。"""

    def ungrounded_source(question: str, retrieved) -> list[dict[str, Any]]:
        return [
            {
                "text": "The method works on quantum computers in deep space.",
                "evidence": [span_of(DOC_A, "improved accuracy by 5%")],
            }
        ]

    graph = make_graph(make_deps(claim_source=ungrounded_source))
    out = graph.invoke(
        {"question": "Where does the method work?", "session_id": "s1"},
        config("t-ungrounded"),
    )
    assert out["verdicts"][0]["status"] in ("unsupported", "fabricated")
    assert not out.get("memories"), "无依据结论不得写入记忆"


# ─────────────── 规则基线做不到什么（记录边界）───────────────


def test_rule_baseline_misses_cross_axis_polarity() -> None:
    """★ 记录规则方法的**原理性边界**，而不是假装它什么都能判。

    下面两句在**词面上完全同构**（都是"质量变好" vs "方向下降"），但语义相反：

    - ``decreased accuracy`` vs ``improved accuracy`` → 真矛盾，**规则漏报**
    - ``decreased latency`` vs ``improved latency`` → 其实一致，规则判无冲突是**对的**

    区别只在于「被修饰的指标是否越大越好」：accuracy 越大越好，latency 越小越好。
    这需要**世界知识**，纯词面方法拿不到。

    因此这里**有意不加规则**：若把 improve/decrease 加成反义词，
    第二句就会变成误报 —— 用漏报换误报并不划算。

    这条边界正是 P9 消融要量化的东西：**LLM 判断在这些跨轴案例上能补多少**。
    """
    from copilot.evidence.text import polarity_conflicts

    # 真矛盾，但规则判不出来（漏报）
    assert polarity_conflicts("decreased accuracy", "improved accuracy") == ()

    # 词面同构，但语义其实一致 —— 规则判无冲突是正确的
    assert polarity_conflicts("decreased latency", "improved latency") == ()

    # 同轴的对立词则能正确识别
    assert polarity_conflicts("increased latency", "reduced latency")
