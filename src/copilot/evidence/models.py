"""证据绑定的数据模型。

设计要点（每条都对应一个具体的失败模式）
----------------------------------------

1. **四态而非布尔**
   `supported / partially_supported / unsupported / fabricated`
   用布尔会丢掉"部分支持"这个真实存在的情况，导致只能二选一地撒谎。
   尤其 `fabricated`（与证据矛盾 / 引用不存在）必须与 `unsupported`
   （只是没找到证据）区分开——前者是编造，后者是信息不足，
   对用户的含义完全不同。

2. **EvidenceSpan 冗余存原文**
   同时存 `(doc_id, source_hash, char_start, char_end, text)`，
   是为了能做**三重一致性校验**：
   文档存在 → 内容 hash 未被篡改 → 偏移处文本确实等于 `text`。
   只存偏移不存原文，就无法发现"检索后被替换"的篡改；
   只存原文不存偏移，就无法回溯到上下文。

3. **引用元数据与证据片段分开**
   引用（DOI/标题/作者/年份）解决"这篇文献是否真实存在"，
   证据片段解决"这个结论是否被原文支持"。
   两件事的校验方式完全不同（一个是查询，一个是文本比对），
   混在一起会写出一团无法测试的逻辑。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

__all__ = [
    "SupportStatus",
    "Document",
    "EvidenceSpan",
    "CitationMeta",
    "Claim",
    "DocStore",
    "DoiResolver",
    "InMemoryDocStore",
    "StaticDoiResolver",
    "hash_content",
]


class SupportStatus(str, Enum):
    """claim 相对证据的校验状态。

    取值语义边界：

    - ``SUPPORTED``：证据明确支持该结论。
    - ``PARTIALLY_SUPPORTED``：证据支持结论的一部分，或支持一个更弱的版本。
    - ``UNSUPPORTED``：找不到任何证据。**信息不足，不是编造。**
    - ``FABRICATED``：与证据矛盾，或引用不存在（DOI 解析失败、
      文献库无匹配、片段不在原文中）。**这是编造，必须拦截。**
    """

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    FABRICATED = "fabricated"


def hash_content(text: str) -> str:
    """返回文档内容的 sha256（十六进制）。

    统一在这里算，避免各处用不同算法导致 hash 不可比。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    """一篇被检索到的文档。

    ``content_hash`` 由 ``create()`` 自动计算，**不要手工赋值**——
    手工赋值会让"篡改检测"失去意义（攻击者也能算出正确 hash）。
    """

    doc_id: str
    content: str
    content_hash: str

    @classmethod
    def create(cls, doc_id: str, content: str) -> Document:
        return cls(doc_id=doc_id, content=content, content_hash=hash_content(content))


@dataclass(frozen=True)
class EvidenceSpan:
    """指向某篇文档中一段原文的定位信息。

    ``char_start`` / ``char_end`` 是 **Python 字符串下标（字符偏移）**，
    不是字节偏移。用字符偏移是为了让 ``text[start:end] == text`` 这一步
    在中文/emoji 上也能成立。
    """

    doc_id: str
    source_hash: str
    char_start: int
    char_end: int
    text: str

    def __post_init__(self) -> None:
        if self.char_start < 0:
            raise ValueError(f"char_start 不能为负：{self.char_start}")
        if self.char_end < self.char_start:
            raise ValueError(
                f"char_end({self.char_end}) 不能小于 char_start({self.char_start})"
            )
        if self.char_end - self.char_start != len(self.text):
            raise ValueError(
                "char_end - char_start 必须等于 text 长度："
                f"偏移跨度 {self.char_end - self.char_start}，text 长度 {len(self.text)}"
            )


@dataclass(frozen=True)
class CitationMeta:
    """一条引用的元数据。字段全部可选，因为缺失本身是要被检查的信号。"""

    doi: str | None = None
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None


@dataclass
class Claim:
    """一条待验证的原子断言。

    "原子"指：可以独立判断真伪的最小单位。
    "A 与 B 均有效，且 A 优于 B" 应拆成三条 claim，
    否则无法定位到底哪一部分没有证据。
    """

    claim_id: str
    text: str
    status: SupportStatus = SupportStatus.UNSUPPORTED
    evidence: list[EvidenceSpan] = field(default_factory=list)
    citations: list[CitationMeta] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)


# ─────────────────────────────────────────────────────────────
# 接缝（seam）：把「需要外部资源」的部分抽成可注入的协议，
# 这样确定性校验的测试不需要网络，也不需要真实文献库。
# ─────────────────────────────────────────────────────────────


class DocStore(Protocol):
    """按 doc_id 取回文档。"""

    def get(self, doc_id: str) -> Document | None: ...


class DoiResolver(Protocol):
    """判断一个 DOI 是否真实可解析。

    真实实现会发 HTTP 请求；测试里用 ``StaticDoiResolver`` 注入固定集合。
    """

    def resolves(self, doi: str) -> bool: ...


class InMemoryDocStore:
    """内存版 DocStore。用于测试与本地小规模验证。"""

    def __init__(self, documents: list[Document] | None = None) -> None:
        self._docs: dict[str, Document] = {}
        for doc in documents or []:
            self.add(doc)

    def add(self, doc: Document) -> None:
        self._docs[doc.doc_id] = doc

    def get(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def __len__(self) -> int:
        return len(self._docs)


class StaticDoiResolver:
    """固定集合的 DOI 解析器。

    刻意**不提供** "always true" 的默认实现：那会让测试变成同义反复，
    校验逻辑对不对根本测不出来。
    """

    def __init__(self, known: set[str] | None = None) -> None:
        self._known = {d.lower() for d in (known or set())}

    def resolves(self, doi: str) -> bool:
        return bool(doi) and doi.lower() in self._known
