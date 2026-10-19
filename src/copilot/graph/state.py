"""流水线状态定义。

为什么状态里放 dict 而不是我的 dataclass
----------------------------------------
LangGraph 的 checkpointer 需要把状态**序列化**后才能落盘持久化。
自定义 dataclass 需要额外注册序列化器，且一旦类结构变动，
历史 checkpoint 就可能读不回来。

因此状态里统一用 **JSON 原生结构**（str/list/dict/bool），
在节点的入口/出口与类型化模型（`Claim`/`EvidenceSpan`/`MemoryItem`）互转。
代价是节点里多几行转换代码，换来的是 checkpoint 可移植、可人工查看、
且不随内部类重构而失效 —— 对一个强调可复现的项目，这个取舍是值得的。

字段设计说明
------------
- ``plan``/``retrieved``/``claims``/``verdicts``：主数据流，逐节点填充
- ``risks``：**累加**字段（用 reducer），因为风险可能由多个节点各自发现
- ``needs_approval``/``approval``：人工审批闸门（LangGraph ``interrupt``）
- ``errors``：同样累加，便于失败归因
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

__all__ = ["PipelineState", "SUBQUESTION_LIMIT"]


class PipelineState(TypedDict, total=False):
    """端到端流水线的共享状态。"""

    # ── 输入 ──
    question: str
    session_id: str

    # ── 规划 ──
    plan: list[str]
    """把问题拆成的子问题（规则版按连接词与疑问点切分）。"""

    # ── 检索 ──
    retrieved: list[dict[str, Any]]
    """候选片段：``{doc_id, chunk_id, char_start, char_end, text, source_hash, score}``。"""

    # ── 断言与校验 ──
    claims: list[dict[str, Any]]
    """候选断言：``{claim_id, text, evidence: [...], citations: [...], confidence}``。"""

    verdicts: list[dict[str, Any]]
    """逐条校验结果：``{claim_id, status, verdict_kind, reasons, coverage}``。"""

    # ── 风控 ──
    risks: Annotated[list[str], operator.add]
    needs_approval: bool
    approval: dict[str, Any] | None
    """``{approved: bool, operator: str, note: str}``，由人工审批后回填。"""

    # ── 记忆 ──
    memories: list[dict[str, Any]]
    """写入记忆的结果：``{content, decision, reason, stored, quarantined}``。"""

    # ── 输出 ──
    answer: dict[str, Any] | None
    errors: Annotated[list[str], operator.add]


#: 计划里最多保留几个子问题。过多会让后续检索与校验线性膨胀；
#: 过少则覆盖不全。当前取值是「先用一个保守值，等评测数据再调」。
SUBQUESTION_LIMIT = 3
