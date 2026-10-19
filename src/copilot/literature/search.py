"""B1 文献检索 与 B3 引用管理 —— 两件都能**完全确定性**完成的事。

## B1：为什么默认用「BM25 ∪ 字符 trigram」

这是本项目 `reports/retrieval_ablation.md` 里**唯一在配对检验下显著胜出**
的配置（MRR +0.041，p(BH)=0.007）。

同一份报告里还测出 **dense 融合显著有害**（MRR −0.091，p(BH)=0.010）——
静态嵌入弱于 BM25，而 RRF 等权融合会把强检索器拖下水。

所以这里**不启用 dense**，不是偏好，是有数据支撑的决定。

## B3：为什么去重不能按字符串相等

同一篇文献在不同来源里的写法千差万别：

```
张三. 一种方法[J]. 计算机学报, 2020.
张三 . 一种方法 [J]. 计算机学报 , 2020.
```

按字符串比等于"全都不重复"。正确做法是**按标识去重**：

1. 有 DOI → 按 DOI（最可靠）；
2. 没有 DOI → 按**规范化标题**（去空白、去标点、小写）。

**为什么 DOI 优先**：同名论文真实存在（不同年份、不同团队撞名），
只有 DOI 能唯一确定一篇文献。

## B3：复用而不是重写

| 能力 | 现成实现 |
|---|---|
| BibTeX / GB-T 7714 生成 | `copilot.bib`（共享层，F1 也在用） |
| DOI / 文献库核实 | `copilot.evidence.verify_citation`（E2） |

**同一件事只能有一份实现。** 引用格式在本项目里有两处会用到
（B3 管理与 F1 导出），共用 `copilot.bib` 就是这条纪律的体现。
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from copilot.bib import BibEntry, render_bibtex_file, render_gbt7714_list
from copilot.literature.models import Library, LibraryItem, ReferenceIssue, ReferenceReport

__all__ = [
    "search_literature",
    "manage_references",
    "normalize_title",
    "dedupe_key",
    "DEFAULT_RETRIEVER",
]

DEFAULT_RETRIEVER = "bm25+trigram"

# 标题规范化与去重键**已上移到共享层** `copilot.bib`：
# D 阶段的「文献引用」节点也要用同一套判定，
# 而阶段之间不该互相 import。这里只做转发，保证"只有一份实现"。
from copilot.bib import dedupe_key, normalize_title  # noqa: E402,F401


# ─────────────────────────── B1 ───────────────────────────


def search_literature(
    keywords: Sequence[str],
    corpus: Sequence[Any],
    *,
    k: int = 20,
    max_chars: int = 320,
) -> Library:
    """多路检索（BM25 ∪ 字符 trigram），返回带 hash 的文献库。

    ``corpus``：``Document`` 序列（见 ``copilot.evidence.Document``）。
    **不做引文图扩展** —— 那需要外部数据源，与"零外部依赖、可复现"冲突。
    """
    from copilot.retrieval.base import rank_documents
    from copilot.retrieval.sparse import Bm25Retriever, CharTrigramRetriever
    from copilot.textseg import iter_segments  # noqa: F401  (保证依赖可见)
    from copilot.retrieval.corpus import chunk_document

    query = " ".join(k.strip() for k in keywords if k and k.strip())
    if not query:
        return Library(query="", retriever=DEFAULT_RETRIEVER,
                       notes=["没有关键词，检索未执行"])

    chunks: list[Any] = []
    for doc in corpus:
        chunks.extend(chunk_document(doc, max_chars=max_chars, overlap_sentences=1))
    if not chunks:
        return Library(query=query, retriever=DEFAULT_RETRIEVER,
                       notes=["语料为空，检索未执行"])

    from copilot.retrieval.base import RrfFusion

    retriever = RrfFusion(
        [Bm25Retriever(chunks), CharTrigramRetriever(chunks)],
        name=DEFAULT_RETRIEVER,
    )
    results = retriever.retrieve(query, k=max(k * 4, k))
    ranked = rank_documents(results, k=k)

    by_id = {d.doc_id: d for d in corpus}
    scores: dict[str, float] = {}
    snippets: dict[str, str] = {}
    for r in results:
        doc_id = getattr(r, "doc_id", "")
        if doc_id and doc_id not in scores:
            scores[doc_id] = float(getattr(r, "score", 0.0))
            snippets[doc_id] = (getattr(r, "text", "") or "")[:160]

    items = [
        LibraryItem(
            doc_id=doc_id,
            title=getattr(by_id.get(doc_id), "title", "") or doc_id,
            content_hash=getattr(by_id.get(doc_id), "content_hash", "") or "",
            score=scores.get(doc_id, 0.0),
            snippet=snippets.get(doc_id, ""),
        )
        for doc_id in ranked
        if doc_id in by_id
    ]

    return Library(
        items=items,
        query=query,
        retriever=DEFAULT_RETRIEVER,
        notes=[
            f"检索 {len(corpus)} 篇语料，返回 {len(items)} 篇",
            "未启用 dense（实测显著有害，见 reports/retrieval_ablation.md）",
        ],
    )


# ─────────────────────────── B3 ───────────────────────────


def manage_references(
    entries: Sequence[BibEntry],
    *,
    library: Any = None,
    resolver: Any = None,
) -> ReferenceReport:
    """去重 → 核实（可选）→ 生成两种格式。

    ``library`` / ``resolver`` 不给时**跳过核实并说明** ——
    静默跳过会让人以为"引用都查过了、没问题"，那是最危险的情况。
    """
    report = ReferenceReport(total_in=len(entries))

    # ── 去重（保持原顺序，保留第一次出现的那条）──
    seen: dict[str, int] = {}
    unique: list[BibEntry] = []
    for e in entries:
        key = dedupe_key(e)
        if key in seen:
            report.duplicates.append(e.title or e.key)
            continue
        seen[key] = len(unique)
        unique.append(e)

    report.total_out = len(unique)

    # ── 核实（可选）──
    if library is None or resolver is None:
        report.notes.append(
            "未提供文献库 / DOI 解析器 —— **跳过引用核实**（这不等于「引用都没问题」）"
        )
    else:
        from copilot.evidence import CitationMeta, CitationProblem, verify_citation

        for e in unique:
            meta = CitationMeta(
                doi=e.doi,
                title=e.title or None,
                authors=tuple(e.authors),
                year=e.year,
                venue=e.venue or None,
            )
            try:
                res = verify_citation(meta, library, resolver)
            except Exception as exc:  # noqa: BLE001 - 单条失败不中断整体
                report.issues.append(
                    ReferenceIssue(
                        code="REF-VERIFY-ERROR",
                        message=f"核实过程出错：{type(exc).__name__}",
                        detail=str(exc)[:120],
                        severity="warning",
                    )
                )
                continue
            problem = getattr(res, "problem", CitationProblem.OK)
            if problem is CitationProblem.OK:
                continue
            report.issues.append(
                ReferenceIssue(
                    code=f"REF-{problem.value.upper()}",
                    message=f"引用核实不通过：{problem.value}",
                    detail=(e.title or e.key)[:80],
                    severity="error" if problem is CitationProblem.DOI_NOT_RESOLVABLE else "warning",
                )
            )

    # ── 生成（复用共享层的实现）──
    report.bibtex = render_bibtex_file(unique)
    report.gbt7714 = render_gbt7714_list(unique)

    if report.duplicates:
        report.notes.append(
            f"按 DOI / 规范化标题去重，移除 {len(report.duplicates)} 条重复"
        )
    return report
