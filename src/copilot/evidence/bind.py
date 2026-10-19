"""证据绑定：从检索到的文档里，自动找出支撑某个 claim 的原文片段。

这一层的产物是 :class:`~copilot.evidence.models.EvidenceSpan` ——
带**精确字符偏移**与**来源 hash** 的原文定位。它是整条验证链的地基：
没有它，`verify.py` 的三重一致性校验无从谈起。

核心设计判断：把两个问题分开
----------------------------
容易被混为一谈的两件事：

1. **相关性**：这段原文是不是在讲同一件事？（用词汇覆盖率判断）
2. **一致性**：它支持还是反对这个结论？（用极性/数字判断）

本模块**只回答第 1 个问题**，一致性交给 :mod:`copilot.evidence.support`。

为什么这样分：如果绑定阶段就把"矛盾的证据"丢掉，那么
"结论与检索到的原文相矛盾"这一**最强的幻觉信号**就永远不会触发，
只会退化成"没找到证据"（UNSUPPORTED）。而这两者对用户的含义
天差地别 —— 前者是可靠性事故，后者只是信息不足。

偏移计算的坑
------------
**绝不能**在 NFKC 归一化后的文本上计算偏移。归一化会改变字符串长度
（全角→半角、连字展开等），偏移一旦错位，`verify_span` 的文本比对必然失败。
因此：分段与偏移基于**原文**，归一化只在评分时临时使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

from copilot.evidence.models import Claim, Document, EvidenceSpan
from copilot.evidence.text import coverage, numbers_in
from copilot.textseg import iter_segments, trimmed_segment  # noqa: F401  (向后兼容再导出)

__all__ = [
    "Segment",
    "BindingResult",
    "EvidenceBinder",
    "RuleBinder",
    "iter_segments",
    "bind_evidence",
]


@dataclass(frozen=True)
class Segment:
    """一个带评分的候选片段。保留评分是为了能解释"为什么选了它"。"""

    doc_id: str
    char_start: int
    char_end: int
    text: str
    relevance: float
    matched_numbers: int = 0


@dataclass
class BindingResult:
    """绑定过程与结果的完整快照（可审计）。"""

    claim_text: str
    spans: list[EvidenceSpan] = field(default_factory=list)
    candidates: list[Segment] = field(default_factory=list)
    considered: int = 0
    best_relevance: float = 0.0

    @property
    def bound(self) -> bool:
        return bool(self.spans)

    @property
    def rejection_reason(self) -> str | None:
        if self.bound:
            return None
        if self.considered == 0:
            return "没有可检索的文档"
        return (
            f"最高相关性仅 {self.best_relevance:.2f}，低于绑定阈值；"
            "按「宁可少绑定，不可绑错」处理，留空让下游判为信息不足"
        )


class EvidenceBinder(Protocol):
    """证据绑定接口。便于把规则方案与检索/模型方案互换对照。"""

    def bind(self, claim_text: str, documents: Iterable[Document]) -> BindingResult: ...


class RuleBinder:
    """规则式证据绑定：句级切分 + 词汇覆盖率排序。

    参数取舍：

    - ``min_relevance`` 调高 → 绑定更保守，漏绑变多但**错绑变少**。
      错绑比漏绑危险：它会让"矛盾"检测基于错误的片段触发。
    - ``max_spans`` 调高 → 覆盖跨句证据，但也引入更多噪声片段。
    """

    def __init__(self, *, min_relevance: float = 0.5, max_spans: int = 2) -> None:
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError(f"min_relevance 必须在 [0,1]，当前 {min_relevance}")
        if max_spans < 1:
            raise ValueError(f"max_spans 必须 >= 1，当前 {max_spans}")
        self.min_relevance = min_relevance
        self.max_spans = max_spans

    # ── 内部 ────────────────────────────────────────────────

    def _score(self, claim_text: str, seg_text: str) -> tuple[float, int]:
        """返回 ``(相关性, 命中数字个数)``。

        相关性**只看词汇覆盖**，不看极性 —— 极性是"是否一致"的问题，
        由下游 :mod:`support` 负责。混在一起会让矛盾的证据被提前丢弃。
        """
        rel = coverage(claim_text, seg_text)
        claim_nums = set(numbers_in(claim_text))
        matched = len(claim_nums & set(numbers_in(seg_text))) if claim_nums else 0
        return rel, matched

    # ── 对外 ────────────────────────────────────────────────

    def bind(self, claim_text: str, documents: Iterable[Document]) -> BindingResult:
        result = BindingResult(claim_text=claim_text)

        # 先实体化：后面既要遍历评分，又要按 doc_id 回查 content_hash。
        # 若直接迭代生成器，第二次遍历会得到空序列（曾经的实现缺陷）。
        docs = list(documents)

        candidates: list[Segment] = []
        for doc in docs:
            for start, end, text in iter_segments(doc.content):
                rel, matched = self._score(claim_text, text)
                candidates.append(
                    Segment(
                        doc_id=doc.doc_id,
                        char_start=start,
                        char_end=end,
                        text=text,
                        relevance=rel,
                        matched_numbers=matched,
                    )
                )
                result.considered += 1

        if not candidates:
            return result

        # 先按相关性，再用命中数字数做同分排序 —— 数字一致是更强的证据信号
        candidates.sort(key=lambda s: (s.relevance, s.matched_numbers), reverse=True)
        result.candidates = candidates
        result.best_relevance = candidates[0].relevance

        chosen = [c for c in candidates if c.relevance >= self.min_relevance]
        chosen = chosen[: self.max_spans]
        if not chosen:
            return result

        by_id = {d.doc_id: d for d in docs}
        for seg in chosen:
            doc = by_id[seg.doc_id]
            result.spans.append(
                EvidenceSpan(
                    doc_id=seg.doc_id,
                    source_hash=doc.content_hash,
                    char_start=seg.char_start,
                    char_end=seg.char_end,
                    text=seg.text,
                )
            )
        return result


def bind_evidence(
    claim: Claim,
    documents: Iterable[Document],
    *,
    binder: EvidenceBinder | None = None,
) -> BindingResult:
    """**就地**把证据绑定到 ``claim`` 上，并返回绑定过程的结果。

    返回 :class:`BindingResult` 而不是 claim，是为了让调用方能拿到审计信息
    （候选评分、未绑定的原因）—— 这些在排查"为什么这条没绑上"时是必需的。

    绑定**只增不减**：若 claim 已有证据，则保留原有的，追加新找到的。
    """
    active = binder or RuleBinder()
    docs = list(documents)  # 需要多次遍历（评分 + 回查 hash），先实体化
    result = active.bind(claim.text, docs)
    if result.spans:
        claim.evidence.extend(result.spans)
    return result
