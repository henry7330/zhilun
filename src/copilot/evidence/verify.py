"""确定性校验：能靠规则判定真伪的部分。

为什么要把「确定性」单独拎出来
------------------------------
防幻觉有两类检查，性质完全不同：

| 类型 | 例子 | 判定方式 |
|---|---|---|
| **确定性** | DOI 能否解析、片段是否真在原文、hash 是否一致 | 规则，**可 100% 判定，不需要模型** |
| 语义性 | 这段证据是否"支持"这个结论 | 需要模型判断，有不确定性 |

把它们混在一起是常见的架构错误：会让人误以为"防幻觉"整体都不可靠。
实际上**"引用是否真实存在"完全可以做成确定性的**，而
论文助手最致命的失败模式（编造一篇不存在的文献）恰好落在这一类。

所以本模块负责确定性部分，并**显式返回"需要语义判断"**这个状态，
把边界画清楚，而不是用规则硬猜语义。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from copilot.evidence.models import (
    CitationMeta,
    Claim,
    DocStore,
    DoiResolver,
    EvidenceSpan,
    SupportStatus,
)

__all__ = [
    "SpanProblem",
    "SpanVerification",
    "CitationProblem",
    "CitationVerification",
    "VerdictKind",
    "DeterministicVerdict",
    "CitationLibrary",
    "InMemoryCitationLibrary",
    "normalize_title",
    "verify_span",
    "verify_citation",
    "verify_claim",
]


# ─────────────────────────────────────────────────────────────
# 证据片段校验
# ─────────────────────────────────────────────────────────────


class SpanProblem(str, Enum):
    OK = "ok"
    DOC_NOT_FOUND = "doc_not_found"
    HASH_MISMATCH = "hash_mismatch"
    TEXT_MISMATCH = "text_mismatch"


@dataclass(frozen=True)
class SpanVerification:
    """一次片段校验的结果。``detail`` 用于人工排查，不要给模型看。"""

    span: EvidenceSpan
    problem: SpanProblem
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.problem is SpanProblem.OK


def verify_span(span: EvidenceSpan, doc_store: DocStore) -> SpanVerification:
    """三重一致性校验：文档存在 → hash 一致 → 偏移处文本等于记录文本。

    任何一步失败都说明这条证据不可信，进而该 claim 不能被判为 supported。

    这里**故意逐层短路**并把失败原因返回，而不是只返回 True/False——
    否则线上出问题时无法区分"文档没检索到"和"文档被换过了"。
    """
    doc = doc_store.get(span.doc_id)
    if doc is None:
        return SpanVerification(
            span, SpanProblem.DOC_NOT_FOUND, f"doc_id={span.doc_id!r} 不在文档库中"
        )

    if doc.content_hash != span.source_hash:
        return SpanVerification(
            span,
            SpanProblem.HASH_MISMATCH,
            f"记录 hash={span.source_hash[:12]}… 与实际 hash={doc.content_hash[:12]}… 不符"
            "（文档在检索后被修改或替换）",
        )

    actual = doc.content[span.char_start : span.char_end]
    if actual != span.text:
        return SpanVerification(
            span,
            SpanProblem.TEXT_MISMATCH,
            f"偏移 [{span.char_start}:{span.char_end}] 处的实际文本与记录不符；"
            f"实际={actual[:60]!r} 记录={span.text[:60]!r}",
        )

    return SpanVerification(span, SpanProblem.OK)


# ─────────────────────────────────────────────────────────────
# 引用真实性校验
# ─────────────────────────────────────────────────────────────


class CitationProblem(str, Enum):
    OK = "ok"
    NO_IDENTIFIER = "no_identifier"
    DOI_NOT_RESOLVABLE = "doi_not_resolvable"
    NOT_IN_LIBRARY = "not_in_library"
    YEAR_MISMATCH = "year_mismatch"


@dataclass(frozen=True)
class CitationVerification:
    citation: CitationMeta
    problem: CitationProblem
    matched: CitationMeta | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.problem is CitationProblem.OK


class CitationLibrary(Protocol):
    """真实文献库：按 DOI 或标题查回记录，查不到返回 None。"""

    def lookup(self, citation: CitationMeta) -> CitationMeta | None: ...


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """标题归一化：大小写、标点、多余空白都不应影响匹配。

    不做归一化会导致大量假阳性——同一篇文献在不同来源里
    大小写与标点写法经常不一致（"Attention Is All You Need" vs
    "Attention is all you need."）。
    """
    text = title.casefold()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


class InMemoryCitationLibrary:
    """内存文献库，按 DOI 与归一化标题双索引。"""

    def __init__(self, records: list[CitationMeta] | None = None) -> None:
        self._by_doi: dict[str, CitationMeta] = {}
        self._by_title: dict[str, CitationMeta] = {}
        for record in records or []:
            self.add(record)

    def add(self, record: CitationMeta) -> None:
        if record.doi:
            self._by_doi[record.doi.casefold()] = record
        if record.title:
            self._by_title[normalize_title(record.title)] = record

    def lookup(self, citation: CitationMeta) -> CitationMeta | None:
        # DOI 是强标识，优先用它
        if citation.doi:
            hit = self._by_doi.get(citation.doi.casefold())
            if hit is not None:
                return hit
        if citation.title:
            return self._by_title.get(normalize_title(citation.title))
        return None

    def __len__(self) -> int:
        return len(self._by_doi) or len(self._by_title)


def verify_citation(
    citation: CitationMeta,
    library: CitationLibrary,
    resolver: DoiResolver,
) -> CitationVerification:
    """校验一条引用是否真实存在。

    判定顺序（从强到弱）：

    1. 既无 DOI 也无标题 → ``NO_IDENTIFIER``：**无法核实**，
       这类引用不允许作为支持性证据。
    2. 有 DOI 但解析不了 → ``DOI_NOT_RESOLVABLE``：编造信号。
    3. 文献库里查不到 → ``NOT_IN_LIBRARY``。
    4. 查到了但年份对不上 → ``YEAR_MISMATCH``。

    ⚠️ 注意第 3 条的语义：**"不在本地库里" 不等于 "不存在"**。
    本地库只是本次检索到的文献集合，所以这一步的强度弱于第 2 条。
    真实系统里应结合解析器一起判断；把这个区分写出来，
    是为了避免把"我们没检索到"错报成"这篇是编的"。
    """
    if not citation.doi and not citation.title:
        return CitationVerification(
            citation,
            CitationProblem.NO_IDENTIFIER,
            detail="既无 DOI 也无标题，无法核实真实性",
        )

    if citation.doi and not resolver.resolves(citation.doi):
        return CitationVerification(
            citation,
            CitationProblem.DOI_NOT_RESOLVABLE,
            detail=f"DOI {citation.doi!r} 无法解析",
        )

    matched = library.lookup(citation)
    if matched is None:
        return CitationVerification(
            citation,
            CitationProblem.NOT_IN_LIBRARY,
            detail="本地文献库中无匹配记录（注意：这不等于文献不存在）",
        )

    if citation.year is not None and matched.year is not None and citation.year != matched.year:
        return CitationVerification(
            citation,
            CitationProblem.YEAR_MISMATCH,
            matched=matched,
            detail=f"年份不符：声称 {citation.year}，实际 {matched.year}",
        )

    return CitationVerification(citation, CitationProblem.OK, matched=matched)


# ─────────────────────────────────────────────────────────────
# 汇总：确定性部分能给出什么结论
# ─────────────────────────────────────────────────────────────


class VerdictKind(str, Enum):
    """确定性校验的三种结论。第三种是"我判不了"，必须显式表达。"""

    FABRICATED = "fabricated"
    """确定是编造：引用解析不了，或证据片段与原文不符。"""

    INSUFFICIENT = "insufficient"
    """没有可用证据。**是信息不足，不是编造**——对应 UNSUPPORTED。"""

    NEEDS_SEMANTIC = "needs_semantic"
    """确定性检查全部通过，但"证据是否支持该结论"需要语义判断。"""


@dataclass
class DeterministicVerdict:
    kind: VerdictKind
    status: SupportStatus | None = None
    """当 kind 为 FABRICATED / INSUFFICIENT 时给出的最终状态；
    NEEDS_SEMANTIC 时为 None，表示还需后续判断。"""

    reasons: list[str] = field(default_factory=list)


def verify_claim(
    claim: Claim,
    doc_store: DocStore,
    library: CitationLibrary,
    resolver: DoiResolver,
) -> DeterministicVerdict:
    """对一个 claim 做全部确定性检查。

    规则（顺序即优先级）：

    1. 任一引用被判定为编造（无法解析 / 年份不符）→ ``FABRICATED``
    2. 任一证据片段校验失败（文档不存在 / hash 不符 / 文本不符）→ ``FABRICATED``
    3. 既无证据也无引用 → ``INSUFFICIENT``（缺信息，不是编造）
    4. 其余 → ``NEEDS_SEMANTIC``，交给语义层判断支持程度

    注意第 3 条与第 1、2 条的差别：**"没有证据"和"证据是假的"
    对用户的含义完全不同**，前者要提示补检索，后者要告警。
    """
    reasons: list[str] = []

    for citation in claim.citations:
        result = verify_citation(citation, library, resolver)
        if result.problem in (
            CitationProblem.DOI_NOT_RESOLVABLE,
            CitationProblem.YEAR_MISMATCH,
        ):
            reasons.append(f"引用被判定为编造：{result.detail}")
        elif result.problem is CitationProblem.NO_IDENTIFIER:
            reasons.append(f"引用缺少标识、无法核实：{result.detail}")

    for span in claim.evidence:
        result = verify_span(span, doc_store)
        if not result.ok:
            reasons.append(f"证据片段校验失败（{result.problem.value}）：{result.detail}")

    fatal = any("编造" in r or "校验失败" in r for r in reasons)
    if fatal:
        return DeterministicVerdict(VerdictKind.FABRICATED, SupportStatus.FABRICATED, reasons)

    if not claim.evidence and not claim.citations:
        return DeterministicVerdict(
            VerdictKind.INSUFFICIENT,
            SupportStatus.UNSUPPORTED,
            reasons + ["该 claim 既无证据片段也无引用"],
        )

    return DeterministicVerdict(VerdictKind.NEEDS_SEMANTIC, None, reasons)
