"""组装 LangGraph 流水线。

    用户问题
      ↓
    plan        拆子问题
      ↓
    retrieve    P2 混合检索
      ↓
    extract     抽候选断言（接缝：可换成 LLM 生成）
      ↓
    verify      P1 绑定 + 校验 + 支持判断
      ↓
    <条件路由>   出现「编造」级结论？
      ├─ 是 → approval    ★ interrupt 人工审批（图在此暂停）
      │         └─ 批准 → write_memory
      │         └─ 驳回 → answer（标记 blocked）
      └─ 否 → write_memory   P3 写入门禁
                  ↓
                answer      结构化输出

为什么把 approve 放在"写记忆"之前
----------------------------------
顺序反过来（先写记忆再审批）会造成**不可逆的副作用**：
被驳回的结论已经进了长期记忆，审批就失去了意义。
"先审批、后落库"是这类闸门的基本纪律。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from copilot.evidence.bind import RuleBinder
from copilot.evidence.models import Document, InMemoryDocStore, StaticDoiResolver
from copilot.evidence.support import RuleBasedJudge
from copilot.evidence.verify import InMemoryCitationLibrary
from copilot.graph.nodes import PipelineDeps, PipelineNodes
from copilot.graph.state import PipelineState
from copilot.memory.engine import MemoryEngine
from copilot.memory.gate import WriteGate
from copilot.memory.store import MemoryStore
from copilot.retrieval.base import Retriever
from copilot.retrieval.corpus import chunk_document, documents_from_corpus, load_corpus
from copilot.retrieval.sparse import Bm25Retriever, CharTrigramRetriever, RrfFusion

__all__ = ["build_pipeline", "build_deps_from_corpus", "CompiledPipeline"]


def build_pipeline(
    deps: PipelineDeps,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """组装并编译流水线图。

    ``checkpointer`` 传 None 时图仍可运行，但**无法暂停恢复**
    （``interrupt`` 会失效，LangGraph 会抛错提示需要 checkpointer）。
    """
    nodes = PipelineNodes(deps)
    builder = StateGraph(PipelineState)

    builder.add_node("plan", nodes.plan)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("extract", nodes.extract)
    builder.add_node("verify", nodes.verify)
    builder.add_node("approval", nodes.approval)
    builder.add_node("write_memory", nodes.write_memory)
    builder.add_node("answer", nodes.answer)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "retrieve")
    builder.add_edge("retrieve", "extract")
    builder.add_edge("extract", "verify")
    builder.add_conditional_edges(
        "verify",
        PipelineNodes.route_after_verify,
        {"approval": "approval", "write_memory": "write_memory"},
    )
    builder.add_conditional_edges(
        "approval",
        PipelineNodes.route_after_approval,
        {"write_memory": "write_memory", "answer": "answer"},
    )
    builder.add_edge("write_memory", "answer")
    builder.add_edge("answer", END)

    return builder.compile(checkpointer=checkpointer)


def build_deps_from_corpus(
    *,
    corpus_path: Path | None = None,
    memory: MemoryEngine | None = None,
    retriever: Retriever | None = None,
    max_chars: int = 320,
) -> PipelineDeps:
    """用缓存的 arXiv 语料搭一套依赖，便于端到端跑通。

    检索用 P2 的最优组合（句块级 BM25 ∪ 字符 trigram 融合）——
    这是 P2 消融报告里唯一在配对检验下显著胜出的配置。
    """
    papers, _manifest = load_corpus(corpus_path)
    documents = documents_from_corpus(papers, include_title=True)

    chunks = []
    for doc in documents:
        chunks.extend(chunk_document(doc, max_chars=max_chars, overlap_sentences=1))

    if retriever is None:
        retriever = RrfFusion(
            [Bm25Retriever(chunks), CharTrigramRetriever(chunks)],
            name="bm25+trigram",
        )

    doc_store = InMemoryDocStore(documents)
    engine = memory or MemoryEngine(
        MemoryStore(":memory:"), WriteGate(doc_store=doc_store)
    )

    # 语料里没有 DOI，用一个空文献库 + 不解析任何 DOI 的解析器：
    # 这样"引用真实性"这一环对抽取式流程是中性的，
    # 而伪造引用仍会被拦截（DOI 解析失败）。
    return PipelineDeps(
        retriever=retriever,
        documents=documents,
        doc_store=doc_store,
        memory=engine,
        library=InMemoryCitationLibrary([]),
        resolver=StaticDoiResolver(set()),
        binder=RuleBinder(min_relevance=0.35, max_spans=2),
        judge=RuleBasedJudge(),
    )


class CompiledPipeline:
    """把「图 + 线程 id」包一层，方便反复调用与恢复。"""

    def __init__(self, graph: Any, *, thread_id: str = "default") -> None:
        self.graph = graph
        self.thread_id = thread_id

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}}

    def run(self, question: str, *, session_id: str = "default") -> dict[str, Any]:
        """跑到结束或暂停。返回最终状态（暂停时含 ``__interrupt__``）。"""
        return self.graph.invoke(
            {"question": question, "session_id": session_id}, self.config
        )

    def resume(self, decision: dict[str, Any]) -> dict[str, Any]:
        """用审批决定恢复被 ``interrupt`` 暂停的图。"""
        from langgraph.types import Command

        return self.graph.invoke(Command(resume=decision), self.config)

    def state(self) -> dict[str, Any]:
        snapshot = self.graph.get_state(self.config)
        return dict(snapshot.values) if snapshot else {}

    def is_paused(self) -> bool:
        snapshot = self.graph.get_state(self.config)
        return bool(snapshot and snapshot.next)
