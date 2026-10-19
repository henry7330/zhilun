"""记忆引擎：把存储与门禁组装成对外接口。

对外只暴露几个动作，刻意保持小：

- ``observe``    写入一条候选记忆（经过门禁）
- ``promote``    把隔离中的记忆提升为可信（需重新通过门禁）
- ``recall``     召回（默认排除隔离与失效）
- ``supersede``  用新事实取代旧事实（双时态标记，不物理删除）
- ``invalidate`` 显式失效并记录理由
- ``audit``      查阅失效审计日志

没有"删除"接口：本层只做**标记**，物理擦除属于合规流程，另行处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from copilot.evidence.models import EvidenceSpan
from copilot.memory.gate import GateResult, WriteDecision, WriteGate
from copilot.memory.models import (
    ForgetReason,
    ForgetRecord,
    MemoryItem,
    MemoryLayer,
    Provenance,
    TrustLevel,
    utcnow,
)
from copilot.memory.store import MemoryStore

__all__ = ["MemoryEngine", "ObserveResult", "EngineStats"]


@dataclass
class ObserveResult:
    """一次写入尝试的结果，保留门禁理由以便解释与审计。"""

    item: MemoryItem
    gate: GateResult
    stored: bool

    @property
    def quarantined(self) -> bool:
        return self.stored and self.item.quarantined

    @property
    def rejected(self) -> bool:
        return not self.stored


@dataclass
class EngineStats:
    total: int = 0
    live: int = 0
    quarantined: int = 0
    invalid: int = 0
    by_layer: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "live": self.live,
            "quarantined": self.quarantined,
            "invalid": self.invalid,
            "by_layer": dict(self.by_layer),
        }


class MemoryEngine:
    """记忆读写门面。"""

    def __init__(
        self,
        store: MemoryStore | None = None,
        gate: WriteGate | None = None,
        *,
        session_id: str = "default",
    ) -> None:
        self.store = store or MemoryStore()
        self.gate = gate or WriteGate()
        self.session_id = session_id

    # ── 写入路径 ────────────────────────────────────────────

    def observe(
        self,
        content: str,
        *,
        layer: MemoryLayer = MemoryLayer.EPISODIC,
        trust: TrustLevel = TrustLevel.TRUSTED_USER,
        evidence: Sequence[EvidenceSpan] | None = None,
        source_uri: str | None = None,
        source_hash: str | None = None,
        turn: int = 0,
        importance: float = 5.0,
        item_id: str | None = None,
    ) -> ObserveResult:
        """写入候选记忆。是否真正落库由门禁决定。"""
        content = content.strip()
        if not content:
            raise ValueError("记忆内容不能为空")

        item = MemoryItem(
            item_id=item_id or self.store.new_id(),
            layer=layer,
            content=content,
            provenance=Provenance(
                trust=trust,
                session_id=self.session_id,
                turn=turn,
                source_uri=source_uri,
                source_hash=source_hash,
            ),
            evidence=list(evidence or []),
            importance=importance,
        )

        result = self.gate.evaluate(item)
        if result.decision is WriteDecision.REJECT:
            return ObserveResult(item=item, gate=result, stored=False)

        if result.decision is WriteDecision.QUARANTINE:
            item.quarantined = True
            item.quarantine_reason = result.reason

        self.store.add(item)
        return ObserveResult(item=item, gate=result, stored=True)

    def promote(self, item_id: str, *, actor: str = "policy") -> tuple[bool, str]:
        """把隔离中的记忆提升为可信。

        **必须重新过门禁**，而不是无条件放行：隔离的原因可能是
        "当时没有证据"，后来补上了证据才应该放行。
        若重新评估仍不通过，保持隔离并返回原因。
        """
        item = self.store.get(item_id)
        if item is None:
            raise KeyError(f"记忆不存在：{item_id}")
        if not item.quarantined:
            return True, "该记忆未被隔离，无需提升"
        if not item.is_live:
            return False, "该记忆已失效，不应提升"

        result = self.gate.evaluate(item)
        if result.decision is not WriteDecision.ALLOW:
            return False, f"重新评估仍未通过：{result.reason}"

        item.quarantined = False
        item.quarantine_reason = None
        self.store.update(item)
        return True, f"已提升（{actor}）：{result.reason}"

    # ── 读取路径 ────────────────────────────────────────────

    def recall(
        self,
        *,
        layers: Sequence[MemoryLayer] | None = None,
        include_quarantined: bool = False,
        limit: int | None = None,
        touch: bool = True,
    ) -> list[MemoryItem]:
        """召回记忆。默认只返回**有效且未被隔离**的条目。"""
        items = self.store.query(
            layers=layers,
            include_invalid=False,
            include_quarantined=include_quarantined,
            limit=limit,
        )
        if touch:
            for it in items:
                self.store.touch(it.item_id)
        return items

    # ── 失效路径 ────────────────────────────────────────────

    def supersede(
        self,
        old_item_id: str,
        new_item_id: str,
        *,
        detail: str = "",
        actor: str = "policy",
        at: datetime | None = None,
    ) -> ForgetRecord:
        """用新事实取代旧事实：旧条目标记失效，但**内容保留**。

        保留内容使系统能回答"上周这个结论是什么、为什么被推翻"，
        这是正确性问题，不是存储洁癖。
        """
        return self.store.mark_invalid(
            old_item_id,
            reason=ForgetReason.SUPERSEDED,
            detail=detail or f"被 {new_item_id} 取代",
            actor=actor,
            superseded_by=new_item_id,
            at=at,
        )

    def invalidate(
        self,
        item_id: str,
        *,
        reason: ForgetReason,
        detail: str,
        actor: str = "user",
    ) -> ForgetRecord:
        return self.store.mark_invalid(
            item_id, reason=reason, detail=detail, actor=actor
        )

    def audit(self, item_id: str | None = None) -> list[ForgetRecord]:
        return self.store.forget_log(item_id)

    # ── 统计 ────────────────────────────────────────────────

    def stats(self) -> EngineStats:
        all_items = self.store.query(include_invalid=True, include_quarantined=True)
        by_layer: dict[str, int] = {}
        for it in all_items:
            by_layer[it.layer.value] = by_layer.get(it.layer.value, 0) + 1
        return EngineStats(
            total=len(all_items),
            live=sum(1 for it in all_items if it.is_live),
            quarantined=sum(1 for it in all_items if it.quarantined),
            invalid=sum(1 for it in all_items if not it.is_live),
            by_layer=by_layer,
        )
