"""B 阶段 · Agent「文献」—— 一个可以独立调用的服务。

## 这个 Agent 做什么

从「我要研究 X」到「我掌握了 X 的文献全貌，并能写出有据可查的综述」。

```
B1 文献检索 ──▶ Library（带 content_hash，可追溯版本）
B3 引用管理 ──▶ ReferenceReport（去重 + 核实 + BibTeX / GB-T 7714 成品）
B2 综述写作 ──▶ SurveyPlan（分组 + 写作要点 + 需并列的分歧）
```

**顺序是 B1 → B3 → B2**：先把文献与引用理干净，再谈写综述。
文献都没核实过就写综述，等于把不确定的东西写进正文。

## 怎么用（独立调用，不需要先跑 A 阶段）

```python
from copilot.literature import run_literature

report = run_literature(
    keywords=["检索增强", "RAG"],
    corpus=documents,          # 可选：有语料就检索
    entries=bib_entries,       # 可选：有参考文献就管理
)
print(report.summary_line())
print(report.to_markdown())
```

三个节点都可以独立跳过：只给 ``entries`` 就只跑 B3 + B2；
什么都不给也能跑（全部跳过并说明原因，报告会是 ✅ 但只有说明性的 note）。

> 为什么"什么都不给"也算通过：这个阶段没有"必须检查的东西"，
> 只有"能做什么"。**但 note 会明确写出"没做检索/没管理引用"** ——
> 静默跳过才是危险的。

## 阶段之间零依赖

B 阶段**不 import** A/C/D/E/F 任何阶段的服务代码。
它复用的两样东西都在共享层：

| 复用什么 | 在哪 |
|---|---|
| 检索层（BM25 + trigram + RRF） | `copilot.retrieval` |
| 参考文献模型与两种格式生成 | `copilot.bib` |
| 引用核实（DOI / 文献库） | `copilot.evidence`（可选传入） |
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from copilot.bib import BibEntry
from copilot.contracts import Finding
from copilot.quality.format.report import Severity
from copilot.stage import StageNode, ok_outcome, run_stage

from copilot.literature.models import Library, ReferenceReport
from copilot.literature.search import manage_references, search_literature
from copilot.literature.survey import SurveyPlan, plan_survey

__all__ = [
    "run_literature",
    "build_literature_nodes",
    "search_literature",
    "manage_references",
    "plan_survey",
    "Library",
    "ReferenceReport",
    "SurveyPlan",
]

STAGE_KEY = "B"
CODE_SEARCH = "B1-SEARCH"
CODE_REF = "B3-REF"
CODE_SURVEY = "B2-SURVEY"


def build_literature_nodes(
    *,
    keywords: Sequence[str] = (),
    corpus: Sequence[Any] = (),
    entries: Sequence[BibEntry] = (),
    library: Any = None,
    resolver: Any = None,
    k: int = 20,
) -> tuple[StageNode, ...]:
    """装配三个节点。"""

    # ── B1 文献检索 ──

    def b1_search(state: dict[str, Any]):
        if not keywords or not corpus:
            missing = "关键词" if not keywords else "语料"
            return ok_outcome(
                notes=[f"没有{missing} —— **跳过检索**（不等于「没有相关文献」）"],
                stats={"retrieved": 0},
            )

        lib = search_literature(keywords, corpus, k=k)
        findings: list[Finding] = []

        if not lib.items:
            findings.append(
                Finding(
                    node="B1",
                    code=CODE_SEARCH,
                    severity=Severity.WARNING,
                    message="检索没有命中任何文献",
                    fix="换关键词，或扩充语料 —— 没检索到不等于不存在相关研究",
                )
            )

        for item in lib.items:
            if not item.content_hash:
                findings.append(
                    Finding(
                        node="B1",
                        code=CODE_SEARCH,
                        severity=Severity.ERROR,
                        message=f"文献「{item.title}」没有 content_hash",
                        fix="补上内容 hash —— 没有它就无法确认后续引用的是同一版本",
                    )
                )

        return ok_outcome(
            findings=findings,
            stats={"retrieved": len(lib.items), "retriever": lib.retriever},
            notes=list(lib.notes),
            patch={"library": lib},
        )

    # ── B3 引用管理 ──

    def b3_references(state: dict[str, Any]):
        if not entries:
            return ok_outcome(
                notes=["没有参考文献 —— **跳过引用管理**"],
                stats={"total_in": 0},
            )

        report = manage_references(entries, library=library, resolver=resolver)
        findings: list[Finding] = []

        for dup in report.duplicates:
            findings.append(
                Finding(
                    node="B3",
                    code=CODE_REF,
                    severity=Severity.INFO,
                    message=f"重复文献已合并：{dup[:60]}",
                    fix="无需处理 —— 已按 DOI / 规范化标题去重",
                )
            )

        for issue in report.issues:
            findings.append(
                Finding(
                    node="B3",
                    code=issue.code,
                    severity=Severity.ERROR if issue.severity == "error" else Severity.WARNING,
                    message=issue.message,
                    fix="核对 DOI / 年份 / 标题，或补全可识别的标识",
                    payload={"detail": issue.detail},
                )
            )

        return ok_outcome(
            findings=findings,
            stats={
                "total_in": report.total_in,
                "total_out": report.total_out,
                "removed": report.removed,
            },
            notes=list(report.notes),
            patch={"reference_report": report},
        )

    # ── B2 综述写作 ──

    def b2_survey(state: dict[str, Any]):
        if not entries:
            return ok_outcome(
                notes=["没有参考文献 —— **跳过综述分组**"],
            )

        plan = plan_survey(entries)
        findings: list[Finding] = [
            Finding(
                node="B2",
                code=CODE_SURVEY,
                severity=Severity.INFO,
                message=(
                    f"综述骨架：{len(plan.groups)} 组 / {plan.total} 篇；"
                    f"{len(plan.conflicts)} 处需要并列呈现的分歧"
                ),
                fix="按骨架填充正文；每句话都要能落到具体文献（证据绑定）",
            )
        ]

        return ok_outcome(
            findings=findings,
            stats={"groups": len(plan.groups), "papers": plan.total},
            notes=list(plan.notes),
            patch={"survey_plan": plan},
        )

    return (
        StageNode(code="B1", name="文献检索", run=b1_search,
                  description="BM25 ∪ trigram + RRF（复用检索层）"),
        StageNode(code="B3", name="引用管理", run=b3_references,
                  description="去重 + 核实 + 两种格式成品（复用 copilot.bib / evidence）"),
        StageNode(code="B2", name="综述写作", run=b2_survey,
                  description="分组 + 写作要点 + 需并列的分歧（缺省给骨架）"),
    )


def run_literature(
    *,
    keywords: Sequence[str] = (),
    corpus: Sequence[Any] = (),
    entries: Sequence[BibEntry] = (),
    library: Any = None,
    resolver: Any = None,
    k: int = 20,
    label: str = "<inline>",
    only: Sequence[str] | None = None,
    on_node: Callable[[str, Any], None] | None = None,
):
    """**B 阶段的对外入口。** 独立调用，不需要先跑 A 阶段。

    三个输入都可选，缺哪个就跳过对应节点并说明。
    """
    nodes = build_literature_nodes(
        keywords=keywords, corpus=corpus, entries=entries,
        library=library, resolver=resolver, k=k,
    )
    return run_stage(
        stage=STAGE_KEY,
        label=label,
        nodes=nodes,
        state={"keywords": list(keywords), "corpus": list(corpus), "entries": list(entries)},
        only=only,
        on_node=on_node,
    )
