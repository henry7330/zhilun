"""记忆存储：SQLite 后端，支持双时态与审计日志。

为什么用 SQLite 而不是向量库
----------------------------
本模块要解决的**不是**"如何在海量记忆里快速相似检索"（那是检索层的事），
而是"如何可靠地记住一条记忆的来源、有效性、是否被隔离、何时因何失效"。

这些是**关系型与审计型**需求，SQLite 恰好适配：
- 零部署、单文件、可随项目走；
- 事务保证"标记失效 + 写审计"原子发生，不会出现"失效了但没记理由"；
- 需要相似检索时，记忆文本可以交给 P2 的检索层去索引，两者职责分离。

P2 复用点
---------
``query()`` 返回的 :class:`~copilot.memory.models.MemoryItem` 可直接喂给
检索层建索引（记忆文本即普通文本），无需格式转换。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from copilot.memory.models import (
    ForgetReason,
    ForgetRecord,
    MemoryItem,
    MemoryLayer,
    utcnow,
)

__all__ = ["MemoryStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    item_id           TEXT PRIMARY KEY,
    layer             TEXT NOT NULL,
    content           TEXT NOT NULL,
    provenance_json   TEXT NOT NULL,
    valid_at          TEXT NOT NULL,
    invalid_at        TEXT,
    superseded_by     TEXT,
    quarantined       INTEGER NOT NULL DEFAULT 0,
    quarantine_reason TEXT,
    evidence_json     TEXT NOT NULL DEFAULT '[]',
    importance        REAL NOT NULL DEFAULT 5.0,
    access_count      INTEGER NOT NULL DEFAULT 0,
    last_access_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_layer      ON memories(layer);
CREATE INDEX IF NOT EXISTS idx_memories_invalid    ON memories(invalid_at);
CREATE INDEX IF NOT EXISTS idx_memories_quarantine ON memories(quarantined);

CREATE TABLE IF NOT EXISTS forget_log (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id  TEXT NOT NULL,
    reason   TEXT NOT NULL,
    detail   TEXT NOT NULL,
    actor    TEXT NOT NULL,
    at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_forget_item ON forget_log(item_id);
"""


class MemoryStore:
    """记忆的关系型存储。线程内使用即可，不做跨线程连接共享。"""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── 生命周期 ────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── 写入 ────────────────────────────────────────────────

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    def add(self, item: MemoryItem) -> None:
        data = item.to_json()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO memories
            (item_id, layer, content, provenance_json, valid_at, invalid_at,
             superseded_by, quarantined, quarantine_reason, evidence_json,
             importance, access_count, last_access_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                data["item_id"],
                data["layer"],
                data["content"],
                json.dumps(data["provenance"], ensure_ascii=False),
                data["valid_at"],
                data["invalid_at"],
                data["superseded_by"],
                1 if data["quarantined"] else 0,
                data["quarantine_reason"],
                json.dumps(data["evidence"], ensure_ascii=False),
                data["importance"],
                data["access_count"],
                data["last_access_at"],
            ),
        )
        self._conn.commit()

    def update(self, item: MemoryItem) -> None:
        """更新既有条目。条目不存在时抛错，避免静默丢失写入。"""
        if self.get(item.item_id) is None:
            raise KeyError(f"记忆不存在：{item.item_id}")
        self.add(item)

    # ── 读取 ────────────────────────────────────────────────

    def get(self, item_id: str) -> MemoryItem | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE item_id = ?", (item_id,)
        ).fetchone()
        return self._row_to_item(row) if row else None

    def query(
        self,
        *,
        layers: Sequence[MemoryLayer] | None = None,
        include_invalid: bool = False,
        include_quarantined: bool = False,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        """按作用域查询。

        ``include_invalid=False``（默认）与 ``include_quarantined=False`` 是
        **安全默认值**：失效的记忆不该影响判断，隔离的记忆不该被模型看到。
        需要审计时才显式打开。
        """
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list[Any] = []
        if not include_invalid:
            sql += " AND invalid_at IS NULL"
        if not include_quarantined:
            sql += " AND quarantined = 0"
        if layers:
            sql += f" AND layer IN ({','.join('?' * len(layers))})"
            params.extend(layer.value for layer in layers)
        sql += " ORDER BY last_access_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_item(r) for r in rows]

    def count(self, *, include_invalid: bool = True, include_quarantined: bool = True) -> int:
        sql = "SELECT COUNT(*) AS n FROM memories WHERE 1=1"
        if not include_invalid:
            sql += " AND invalid_at IS NULL"
        if not include_quarantined:
            sql += " AND quarantined = 0"
        return int(self._conn.execute(sql).fetchone()["n"])

    # ── 失效与审计 ──────────────────────────────────────────

    def mark_invalid(
        self,
        item_id: str,
        *,
        reason: ForgetReason,
        detail: str,
        actor: str,
        superseded_by: str | None = None,
        at: datetime | None = None,
    ) -> ForgetRecord:
        """标记失效并**在同一事务里**写审计。

        为什么必须原子：若先标记失效再写日志，中间崩溃就会留下
        "失效了但查不出理由"的状态 —— 而这正是审计最需要答案的时刻。
        """
        moment = at or utcnow()
        item = self.get(item_id)
        if item is None:
            raise KeyError(f"记忆不存在：{item_id}")

        with self._conn:  # 事务：任一步失败则全部回滚
            self._conn.execute(
                "UPDATE memories SET invalid_at = ?, superseded_by = ? WHERE item_id = ?",
                (moment.isoformat(), superseded_by, item_id),
            )
            self._conn.execute(
                "INSERT INTO forget_log (item_id, reason, detail, actor, at) VALUES (?,?,?,?,?)",
                (item_id, reason.value, detail, actor, moment.isoformat()),
            )
        return ForgetRecord(item_id=item_id, reason=reason, detail=detail, actor=actor, at=moment)

    def forget_log(self, item_id: str | None = None) -> list[ForgetRecord]:
        if item_id:
            rows = self._conn.execute(
                "SELECT * FROM forget_log WHERE item_id = ? ORDER BY id", (item_id,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM forget_log ORDER BY id").fetchall()
        return [
            ForgetRecord(
                item_id=r["item_id"],
                reason=ForgetReason(r["reason"]),
                detail=r["detail"],
                actor=r["actor"],
                at=datetime.fromisoformat(r["at"]),
            )
            for r in rows
        ]

    def touch(self, item_id: str) -> None:
        """记录一次访问（为后续的遗忘策略提供统计基础）。"""
        self._conn.execute(
            "UPDATE memories SET access_count = access_count + 1, last_access_at = ? "
            "WHERE item_id = ?",
            (utcnow().isoformat(), item_id),
        )
        self._conn.commit()

    # ── 内部 ────────────────────────────────────────────────

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MemoryItem:
        payload = {
            "item_id": row["item_id"],
            "layer": row["layer"],
            "content": row["content"],
            "provenance": json.loads(row["provenance_json"]),
            "valid_at": row["valid_at"],
            "invalid_at": row["invalid_at"],
            "superseded_by": row["superseded_by"],
            "quarantined": bool(row["quarantined"]),
            "quarantine_reason": row["quarantine_reason"],
            "evidence": json.loads(row["evidence_json"]),
            "importance": row["importance"],
            "access_count": row["access_count"],
            "last_access_at": row["last_access_at"],
        }
        return MemoryItem.from_json(payload)
