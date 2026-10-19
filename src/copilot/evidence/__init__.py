"""E2 · 证据层（引用真实性）的公开入口。

这一层是**基础能力**，被检索、记忆、编排、E4/E5 保真校验共同复用
（因此代码放在 ``e2-citation/`` 但属于"能力层"，见 ``00-core/README.md``）。

四态支持判定
------------
======================  ============================================
``supported``           证据明确支持
``partially_supported`` 部分支持 / 数字缺失
``unsupported``         **找不到证据**（信息不足，不是编造）
``fabricated``          **与证据矛盾 / 引用不存在**（要告警）
======================  ============================================

**为什么 ``fabricated`` 必须与 ``unsupported`` 分开**：
前者要告警，后者只提示补检索。合成一态会让系统对普通信息不足过度反应，
同时削弱真正编造信号的严重性。

典型用法（完全确定性，不调用模型）::

    from copilot.evidence import (
        Document, InMemoryDocStore, StaticDoiResolver,
        InMemoryCitationLibrary, verify_claim,
    )

    verdict = verify_claim(claim, doc_store, library, resolver)
    if verdict.kind is VerdictKind.FABRICATED:
        ...

> 本文件原先为空 —— 即 E2 只有子模块、**没有对外接口**。
> 补齐后编排层（``copilot.graph.quality``）可以直接
> ``from copilot.evidence import verify_claim``，不必知道内部文件划分。
"""

from copilot.evidence.bind import (
    BindingResult,
    EvidenceBinder,
    RuleBinder,
    Segment,
    bind_evidence,
)
from copilot.evidence.extract import (
    ClaimExtractor,
    ExtractionStats,
    RuleBasedExtractor,
    split_clauses,
    split_sentences,
)
from copilot.evidence.models import (
    CitationMeta,
    Claim,
    DocStore,
    Document,
    DoiResolver,
    EvidenceSpan,
    InMemoryDocStore,
    StaticDoiResolver,
    SupportStatus,
    hash_content,
)
from copilot.evidence.support import (
    RuleBasedJudge,
    SupportJudge,
    SupportJudgment,
    SupportSignals,
)
from copilot.evidence.text import (
    content_tokens,
    coverage,
    has_hedge,
    has_negation,
    normalize_number,
    numbers_in,
    polarity_conflicts,
    tokenize,
)
from copilot.evidence.verify import (
    CitationLibrary,
    CitationProblem,
    CitationVerification,
    DeterministicVerdict,
    InMemoryCitationLibrary,
    SpanProblem,
    SpanVerification,
    VerdictKind,
    normalize_title,
    verify_citation,
    verify_claim,
    verify_span,
)

__all__ = [
    # 数据模型
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
    # 文本特征
    "tokenize",
    "content_tokens",
    "coverage",
    "normalize_number",
    "numbers_in",
    "has_hedge",
    "has_negation",
    "polarity_conflicts",
    # 原子化
    "ClaimExtractor",
    "ExtractionStats",
    "RuleBasedExtractor",
    "split_sentences",
    "split_clauses",
    # 绑定（只判相关性，不判一致性）
    "Segment",
    "BindingResult",
    "EvidenceBinder",
    "RuleBinder",
    "bind_evidence",
    # 校验
    "SpanProblem",
    "SpanVerification",
    "verify_span",
    "CitationProblem",
    "CitationVerification",
    "CitationLibrary",
    "InMemoryCitationLibrary",
    "normalize_title",
    "verify_citation",
    "VerdictKind",
    "DeterministicVerdict",
    "verify_claim",
    # 支持判断（规则基线）
    "SupportSignals",
    "SupportJudgment",
    "SupportJudge",
    "RuleBasedJudge",
]
