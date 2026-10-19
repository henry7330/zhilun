"""B 阶段的数据模型：文献库、引用管理结果。

## 为什么 ``Library`` 里每篇都带 ``content_hash``

``content_hash`` 是**跨阶段契约的核心**。有了它，D 阶段写作时才能确认
"这条证据来自我当时读的那一版文献"，而不是"某篇同名但内容已变的文章"。

这不是洁癖：arXiv 论文会改版，会议版和预印本的图表/数字可能不同。
**引用一条无法确认版本的证据，等于引用了一条不可复现的证据。**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["LibraryItem", "Library", "ReferenceIssue", "ReferenceReport"]


@dataclass(frozen=True)
class LibraryItem:
    """文献库里的一条。"""

    doc_id: str
    title: str
    content_hash: str
    score: float = 0.0
    #: 命中的片段（可用于证据绑定）
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content_hash": self.content_hash,
            "score": round(self.score, 4),
            "snippet": self.snippet,
        }


@dataclass
class Library:
    """B1 的产出。"""

    items: list[LibraryItem] = field(default_factory=list)
    query: str = ""
    #: 检索配置（必须写清楚，否则结论不可复现）
    retriever: str = ""
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "retriever": self.retriever,
            "count": len(self.items),
            "notes": list(self.notes),
            "items": [i.to_dict() for i in self.items],
        }


@dataclass(frozen=True)
class ReferenceIssue:
    """一条参考文献的问题。"""

    code: str
    message: str
    detail: str = ""
    severity: str = "warning"


@dataclass
class ReferenceReport:
    """B3 的产出：去重 + 核实 + 两种格式的成品。"""

    total_in: int = 0
    total_out: int = 0
    duplicates: list[str] = field(default_factory=list)
    issues: list[ReferenceIssue] = field(default_factory=list)
    bibtex: str = ""
    gbt7714: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def removed(self) -> int:
        return self.total_in - self.total_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "removed": self.removed,
            "duplicates": list(self.duplicates),
            "issues": [
                {"code": i.code, "message": i.message, "detail": i.detail, "severity": i.severity}
                for i in self.issues
            ],
            "notes": list(self.notes),
        }
