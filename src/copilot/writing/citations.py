"""D4 文献引用：正文引用的文献**必须来自 B 阶段的文献库**。

## 为什么这一关比事后核查更有效

旧设计里"引用真实性"是事后核查（E2：查 DOI 能不能解析、片段在不在原文）。
那是**补漏**，成本高、还不一定查得全。

现在的做法是**从源头约束**：

```
B 阶段（文献检索）已经核实过真实性 → 文献库里的每一篇都是可信的
        ↓  把文献库作为输入传给 D 阶段
D4 只做一件事：**正文引用的每一篇，必须能在文献库里找到**
        ↓
        找不到 = 引用来路不明 = 直接报错
```

**这一关是纯确定性的**（按 DOI / 规范化标题比对），不需要模型，也花不了钱。

## 为什么是三态而不是两态

| 情况 | 判定 | 为什么 |
|---|---|---|
| 正文引了，库里也有 | ✅ 通过 | 正常 |
| 正文引了，**库里没有** | ❌ 错误 | 来路不明 —— 这正是最容易出事的地方 |
| 库里有的，正文没引 | ℹ️ 提示 | 不算错，但可能是"忘了引"或"文献浪费" |

第三种只提示不报错：参考文献表里放几篇综述做背景是正常写法。

## 与 E2 的关系

E2（引用真实性）已随链路简化移除。它的职能被拆成两半：
- **源头**：B3 的 `verify_citation`（DOI / 文献库核实）
- **约束**：本节点（引用必须来自那个已核实的库）

两道关合起来，比原来的单点事后核查更严，也更省。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from copilot.bib import BibEntry, from_reference, normalize_title
from copilot.quality.format.docmodel import DocModel
from copilot.quality.rewrite.protected import CITATION_MARK

__all__ = ["CitationIssue", "CitationAudit", "audit_citations", "DOI_IN_RAW"]

#: 从参考文献原文里捞 DOI（`Reference` 模型本身没有 doi 字段）
DOI_IN_RAW = re.compile(r"10\.\d{4,9}/[^\s,;，；]+")


def _keys(entry: BibEntry) -> set[str]:
    """一篇文献的**全部身份标识**（DOI 与标题都算）。

    为什么要"全部"而不是"一个"：
    库里的条目有 DOI，而正文的参考文献常常只有标题 ——
    单键匹配会大面积报"库里没有"，那是误报。
    """
    keys: set[str] = set()
    doi = (getattr(entry, "doi", None) or "").strip().lower().rstrip(".")
    if doi:
        keys.add(f"doi:{doi}")
    title = normalize_title(getattr(entry, "title", "") or "")
    if title:
        keys.add(f"title:{title}")
    return keys


@dataclass(frozen=True)
class CitationIssue:
    kind: str
    """``not_in_library`` / ``orphan_mark`` / ``unused_entry``。"""

    detail: str
    severity: str = "error"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "severity": self.severity}


@dataclass
class CitationAudit:
    """一次引用来源核对的结论。"""

    references: int = 0
    matched: int = 0
    issues: list[CitationIssue] = field(default_factory=list)
    #: 正文引用标注里出现的编号（去重升序）
    marks: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def unmatched(self) -> list[CitationIssue]:
        return [i for i in self.issues if i.kind == "not_in_library"]

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def summary(self) -> str:
        return (
            f"参考文献 {self.references} 条，库内匹配 {self.matched} 条"
            + (f"，{len(self.unmatched)} 条来路不明" if self.unmatched else "")
            + ("" if self.ok else "（有错误）")
        )

    def to_dict(self) -> dict:
        return {
            "references": self.references,
            "matched": self.matched,
            "ok": self.ok,
            "marks": list(self.marks),
            "issues": [i.to_dict() for i in self.issues],
            "notes": list(self.notes),
        }


def audit_citations(
    doc: DocModel,
    library_entries: Sequence[BibEntry],
    *,
    require_all: bool = True,
    report_unused: bool = True,
) -> CitationAudit:
    """核对"正文引用 ⊆ 文献库"。

    ``library_entries``：B 阶段产出的文献库（``copilot.bib.BibEntry`` 序列）。
    ``require_all``：为 ``True`` 时，库外的引用算错误；为 ``False`` 时只提示。
    """
    audit = CitationAudit(notes=[f"文献库 {len(library_entries)} 篇"])

    # ★ 先记下"正文到底有几条参考文献" —— 即使没给文献库也要报出来。
    #   否则报告会写「参考文献 0 条」，而实际有 5 条，用户会以为文档里没引用。
    audit.references = len(doc.references)

    if not library_entries:
        audit.issues.append(
            CitationIssue(
                kind="not_in_library",
                detail="没有提供文献库 —— **无法核对引用来源**（这不等于引用都没问题）",
                severity="warning",
            )
        )
        return audit

    lib_keys: dict[str, BibEntry] = {}
    for e in library_entries:
        for k in _keys(e):
            lib_keys.setdefault(k, e)

    # ── ① 每条参考文献都要能在库里找到 ──
    used: set[int] = set()
    for ref in doc.references:
        probe = from_reference(ref)
        m = DOI_IN_RAW.search(ref.raw or "")
        if m:
            probe.doi = m.group(0).rstrip(".;，。；")

        keys = _keys(probe)
        hit = next((lib_keys[k] for k in keys if k in lib_keys), None)
        if hit is None:
            title = (ref.title or ref.raw or "")[:48]
            audit.issues.append(
                CitationIssue(
                    kind="not_in_library",
                    detail=f"第 [{ref.index}] 条不在文献库：{title}",
                    severity="error" if require_all else "warning",
                )
            )
            continue
        audit.matched += 1
        used.add(id(hit))

    # ── ② 正文引用标注不能超出参考文献条数 ──
    body = doc.body_text
    numbers = sorted(
        {
            int(n)
            for grp in CITATION_MARK.findall(body or "")
            for n in re.findall(r"\d+", grp if isinstance(grp, str) else "")
        }
    )
    audit.marks = numbers
    total = len(doc.references)
    if total:
        orphan = [n for n in numbers if n > total]
        for n in orphan:
            audit.issues.append(
                CitationIssue(
                    kind="orphan_mark",
                    detail=f"正文引用了 [{n}]，但参考文献只有 {total} 条",
                    severity="error",
                )
            )

    # ── ③ 库里没被引用的（只提示）──
    if report_unused:
        unused = [e for e in library_entries if id(e) not in used]
        if unused:
            names = "；".join((e.title or e.key)[:28] for e in unused[:5])
            more = f" 等 {len(unused)} 篇" if len(unused) > 5 else ""
            audit.issues.append(
                CitationIssue(
                    kind="unused_entry",
                    detail=f"文献库里这些没被引用：{names}{more}",
                    severity="info",
                )
            )

    return audit
