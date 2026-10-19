"""记忆层的数据模型。

与 P1 的关系
------------
记忆条目的 ``evidence`` 字段直接复用 P1 的
:class:`~copilot.evidence.models.EvidenceSpan`。**这不是巧合，而是设计**：
记忆的"可信度"必须能追溯到可验证的原文，因此记忆层与证据层共用同一套
定位结构与 hash 校验，不能各搞一套。

三个关键设计判断
----------------

1. **每条记忆都带 provenance（来源与信任级别）**
   没有来源信息的记忆无法判断可信度。"用户亲口说的"和"从某个网页上读到的"
   在后续所有会话中被同等对待，是记忆投毒能够持久化的根本原因。

2. **不可信内容进隔离区，而不是直接丢弃**
   丢弃会丢失审计线索，也失去了"补充证据后转正"的可能。
   隔离 = 存下来但默认召回不到，等验证通过再提升。

3. **失效用标记（``invalid_at``）而不是物理删除**
   保留审计与回滚能力：能回答"这条为什么不再生效""当时是什么内容"。
   对合规场景（被遗忘权）另需物理擦除流程，本模块只做标记与审计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from copilot.evidence.models import EvidenceSpan

__all__ = [
    "TrustLevel",
    "MemoryLayer",
    "ForgetReason",
    "Provenance",
    "MemoryItem",
    "ForgetRecord",
    "utcnow",
]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrustLevel(str, Enum):
    """记忆来源的信任级别。

    分级的意义在于：**同一句话，来自不同渠道，应享受不同信任**。
    风控里"数据源优先级"是同一个道理，这里是它在 Agent 记忆上的落地。
    """

    TRUSTED_USER = "trusted_user"
    """用户在当前会话中的显式陈述。"""

    DERIVED_USER = "derived_user"
    """从多轮交互中归纳出的用户偏好（比显式陈述弱一档）。"""

    TRUSTED_SYSTEM = "trusted_system"
    """系统/开发者注入。"""

    UNTRUSTED_EXTERNAL = "untrusted_external"
    """检索到的文档、网页、PDF。**不可信**。"""

    UNTRUSTED_TOOL = "untrusted_tool"
    """第三方工具/API 的返回值。**不可信**。"""

    UNTRUSTED_CROSS = "untrusted_cross"
    """其他会话或其他用户的内容。**不可信且不应写入本用户记忆**。"""

    @property
    def is_trusted(self) -> bool:
        return self in (
            TrustLevel.TRUSTED_USER,
            TrustLevel.DERIVED_USER,
            TrustLevel.TRUSTED_SYSTEM,
        )


class MemoryLayer(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PREFERENCE = "preference"


class ForgetReason(str, Enum):
    """失效/遗忘的理由。**没有理由的遗忘无法审计。**"""

    USER_REQUEST = "user_request"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    LOW_VALUE = "low_value"
    SECURITY = "security"


@dataclass(frozen=True)
class Provenance:
    """来源信息。不可变，避免被事后篡改。"""

    trust: TrustLevel
    session_id: str
    turn: int = 0
    source_uri: str | None = None
    source_hash: str | None = None
    captured_at: datetime = field(default_factory=utcnow)

    def to_json(self) -> dict[str, Any]:
        return {
            "trust": self.trust.value,
            "session_id": self.session_id,
            "turn": self.turn,
            "source_uri": self.source_uri,
            "source_hash": self.source_hash,
            "captured_at": self.captured_at.isoformat(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Provenance:
        return cls(
            trust=TrustLevel(data["trust"]),
            session_id=data["session_id"],
            turn=int(data.get("turn", 0)),
            source_uri=data.get("source_uri"),
            source_hash=data.get("source_hash"),
            captured_at=datetime.fromisoformat(data["captured_at"]),
        )


@dataclass
class MemoryItem:
    """一条记忆。

    同时携带「双时态有效性」与「隔离状态」两组字段：
    前者回答"这条现在还成立吗"，后者回答"这条可信吗"。
    两件事互不替代 —— 一条隔离中的记忆也可以是"当前成立"的。
    """

    item_id: str
    layer: MemoryLayer
    content: str
    provenance: Provenance

    # ── 双时态 ──
    valid_at: datetime = field(default_factory=utcnow)
    invalid_at: datetime | None = None
    superseded_by: str | None = None

    # ── 隔离 ──
    quarantined: bool = False
    quarantine_reason: str | None = None

    # ── 证据（复用 P1 的结构，使可信度可回溯）──
    evidence: list[EvidenceSpan] = field(default_factory=list)

    # ── 打分与访问统计（为后续遗忘策略留位）──
    importance: float = 5.0
    access_count: int = 0
    last_access_at: datetime = field(default_factory=utcnow)

    @property
    def is_live(self) -> bool:
        """当前有效（未被标记失效）。"""
        return self.invalid_at is None

    @property
    def is_recallable(self) -> bool:
        """默认能否被召回：既有效、又不在隔离区。"""
        return self.is_live and not self.quarantined

    def to_json(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "layer": self.layer.value,
            "content": self.content,
            "provenance": self.provenance.to_json(),
            "valid_at": self.valid_at.isoformat(),
            "invalid_at": self.invalid_at.isoformat() if self.invalid_at else None,
            "superseded_by": self.superseded_by,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "evidence": [
                {
                    "doc_id": s.doc_id,
                    "source_hash": s.source_hash,
                    "char_start": s.char_start,
                    "char_end": s.char_end,
                    "text": s.text,
                }
                for s in self.evidence
            ],
            "importance": self.importance,
            "access_count": self.access_count,
            "last_access_at": self.last_access_at.isoformat(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MemoryItem:
        return cls(
            item_id=data["item_id"],
            layer=MemoryLayer(data["layer"]),
            content=data["content"],
            provenance=Provenance.from_json(data["provenance"]),
            valid_at=datetime.fromisoformat(data["valid_at"]),
            invalid_at=(
                datetime.fromisoformat(data["invalid_at"]) if data["invalid_at"] else None
            ),
            superseded_by=data.get("superseded_by"),
            quarantined=bool(data.get("quarantined", False)),
            quarantine_reason=data.get("quarantine_reason"),
            evidence=[
                EvidenceSpan(
                    doc_id=s["doc_id"],
                    source_hash=s["source_hash"],
                    char_start=int(s["char_start"]),
                    char_end=int(s["char_end"]),
                    text=s["text"],
                )
                for s in data.get("evidence", [])
            ],
            importance=float(data.get("importance", 5.0)),
            access_count=int(data.get("access_count", 0)),
            last_access_at=datetime.fromisoformat(data["last_access_at"]),
        )


@dataclass(frozen=True)
class ForgetRecord:
    """一次失效/遗忘的审计记录。"""

    item_id: str
    reason: ForgetReason
    detail: str
    actor: str
    at: datetime = field(default_factory=utcnow)
