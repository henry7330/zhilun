"""D 阶段 · Agent「写作」—— 一个可以独立调用的服务。

## 这个 Agent 做什么

从「我有研究内容和数据」到「一份结构完整、图表规范的论文草稿」。

```
D1 大纲生成 ──▶ Outline（模板骨架 + 要点分配 + 篇幅分配）
D3 图表规范 ──▶ FigureReport（编号/引用问题 + 具体修复建议）
D2 分章写作 ──▶ 正文骨架（有模型时逐节生成）
```

## 顺序为什么是 D1 → D3 → D2

**先定结构，再查图表，最后写字。**

- 大纲没定就写正文 → 写完发现章节不对，全废；
- 图表放在正文之后查 → 正文改了图表位置，引用又乱了。

D3 放在 D2 之前，是为了在"还没写正文"时先把图表编号固定下来。
（代码里的节点顺序按这个来，而不是按编号 D1/D2/D3。）

## 怎么用（独立调用）

```python
from copilot.writing import run_writing

report = run_writing(
    title="基于图神经网络的推荐方法研究",
    draft=open("thesis.md", encoding="utf-8").read(),   # 可选：有草稿就查图表
)
print(report.summary_line())
```

**不传 ``draft`` 时跳过 D3**，只出大纲。

## 阶段之间零依赖

D 阶段不 import C 阶段。要接上，把 ``ResearchDesign.to_dict()`` 的字典传给
``design=`` 即可 —— **用数据对接，不用代码耦合**。
这样任何一个阶段都能单独用，也能单独测。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from copilot.bib import BibEntry
from copilot.contracts import Finding
from copilot.quality.format import parse_markdown
from copilot.quality.format.report import Severity
from copilot.stage import StageNode, ok_outcome, run_stage
from copilot.writing.citations import CitationAudit, audit_citations
from copilot.writing.figures import analyze_figures
from copilot.writing.models import FigureReport, Outline
from copilot.writing.outline import build_outline

__all__ = [
    "run_writing",
    "build_writing_nodes",
    "build_outline",
    "analyze_figures",
    "audit_citations",
    "CitationAudit",
    "Outline",
    "FigureReport",
]

STAGE_KEY = "D"
CODE_OUTLINE = "D1-OUTLINE"
CODE_SECTION = "D2-SECTION"
CODE_FIGURE = "D3-FIGURE"
CODE_CITATION = "D4-CITATION"

#: 生成器签名：(章节标题, 要点, 目标字数) → 正文
SectionGenerator = Callable[[str, list[str], int], str]


def build_writing_nodes(
    *,
    design: Mapping[str, Any] | None = None,
    generator: SectionGenerator | None = None,
    required_sections: list[str] | None = None,
    entries: Sequence[BibEntry] = (),
) -> tuple[StageNode, ...]:
    """装配四个节点。**顺序是 D1 → D3 → D2 → D4**。

    ``entries`` 是 B 阶段的文献库。D4 只认它 ——
    **正文引用的每一篇都必须能在这里找到**，找不到就是错误。
    """

    # ── D1 大纲 ──

    def d1_outline(state: dict[str, Any]):
        outline = build_outline(
            title=state["title"],
            design=design,
            required_sections=required_sections,
        )
        findings: list[Finding] = []

        for p in outline.validate():
            findings.append(
                Finding(
                    node="D1",
                    code=CODE_OUTLINE,
                    severity=Severity.WARNING if outline.skeleton else Severity.ERROR,
                    message=f"大纲问题：{p}",
                    fix="先补研究设计（C 阶段）或手动填要点 —— 要点为空的章节写出来必然是空的",
                )
            )

        if outline.skeleton:
            findings.append(
                Finding(
                    node="D1",
                    code=CODE_OUTLINE,
                    severity=Severity.INFO,
                    message="当前是**大纲骨架**（没有研究设计，要点为空）",
                    fix="把 ResearchDesign.to_dict() 传给 design= 即可自动分配要点",
                )
            )

        return ok_outcome(
            findings=findings,
            stats={
                "sections": len(outline.flat()),
                "total_words": outline.total_words,
                "skeleton": outline.skeleton,
            },
            notes=[f"目标总字数 {outline.total_words}"],
            patch={"outline": outline},
        )

    # ── D3 图表规范 ──

    def d3_figures(state: dict[str, Any]):
        draft = (state.get("draft") or "").strip()
        if not draft:
            return ok_outcome(notes=["没有草稿 —— 跳过图表检查"])

        doc = parse_markdown(draft, path=state.get("label", "<inline>"))
        report = analyze_figures(doc)

        findings: list[Finding] = []
        for fix in report.fixes:
            findings.append(
                Finding(
                    node="D3",
                    code=CODE_FIGURE,
                    severity=Severity.WARNING,
                    message=fix.describe(),
                    location=f"第 {fix.line + 1} 行",
                    fix="按建议修改（**D3 只给建议，不自动改文件**）",
                    payload=fix.to_dict(),
                )
            )
        for cap in report.unreferenced:
            findings.append(
                Finding(
                    node="D3",
                    code=CODE_FIGURE,
                    severity=Severity.INFO,
                    message=f"{cap.label} {cap.number} 未被正文引用",
                    location=f"第 {cap.line + 1} 行",
                    fix="在正文里补一句引用，或删掉这个图表",
                )
            )

        return ok_outcome(
            findings=findings,
            stats={
                "captions": len(report.captions),
                "references": len(report.references),
                "fixes": len(report.fixes),
                "unreferenced": len(report.unreferenced),
            },
            notes=[f"扫描 {len(report.captions)} 个题注 / {len(report.references)} 处引用"],
            patch={"figure_report": report},
        )

    # ── D2 分章写作 ──

    def d2_sections(state: dict[str, Any]):
        outline: Outline | None = state.get("outline")
        if outline is None:
            return ok_outcome(notes=["没有大纲 —— 跳过写作"])

        findings: list[Finding] = []
        written = 0

        for _level, section in outline.flat():
            if generator is None:
                # 无生成器：只出骨架，且**明确说明这是骨架**
                findings.append(
                    Finding(
                        node="D2",
                        code=CODE_SECTION,
                        severity=Severity.INFO,
                        message=f"「{section.title}」待写（{len(section.points)} 个要点 / 目标 {section.target_words} 字）",
                        fix="接入生成器（generator）或手动撰写",
                    )
                )
                continue

            try:
                text = generator(section.title, list(section.points), section.target_words)
            except Exception as exc:  # noqa: BLE001 - 单节失败不中断其他章节
                findings.append(
                    Finding(
                        node="D2",
                        code=CODE_SECTION,
                        severity=Severity.WARNING,
                        message=f"「{section.title}」生成失败：{type(exc).__name__}",
                        fix="重试或手动撰写",
                    )
                )
                continue

            written += 1
            if not text or len(text) < section.target_words * 0.5:
                findings.append(
                    Finding(
                        node="D2",
                        code=CODE_SECTION,
                        severity=Severity.WARNING,
                        message=(
                            f"「{section.title}」生成内容偏短"
                            f"（{len(text or '')} 字 / 目标 {section.target_words} 字）"
                        ),
                        fix="补充论述或降低该章目标字数",
                    )
                )

        return ok_outcome(
            findings=findings,
            stats={"sections": len(outline.flat()), "written": written},
            notes=["未接入生成器，只出骨架" if generator is None else f"生成 {written} 节"],
        )

    # ── D4 文献引用 ──

    def d4_citations(state: dict[str, Any]):
        from copilot.quality.format import parse_markdown

        text = state.get("draft") or ""
        if not text.strip():
            return ok_outcome(notes=["没有正文 —— **跳过引用来源核对**"])

        doc = parse_markdown(text, path=state.get("label", "<inline>"))
        audit = audit_citations(doc, entries or ())

        findings = [
            Finding(
                node="D4",
                code=CODE_CITATION,
                severity={
                    "error": Severity.ERROR,
                    "warning": Severity.WARNING,
                    "info": Severity.INFO,
                }[i.severity],
                message=i.detail,
                fix=(
                    "只能引用 B 阶段文献库里的文献 —— 补进文献库，或删掉这条引用"
                    if i.kind == "not_in_library"
                    else ""
                ),
            )
            for i in audit.issues
        ]
        return ok_outcome(
            findings=findings,
            stats={
                "references": audit.references,
                "matched": audit.matched,
                "library": len(entries or ()),
            },
            notes=[audit.summary()],
            patch={"citation_audit": audit},
        )

    return (
        StageNode(code="D1", name="大纲生成", run=d1_outline,
                  description="模板骨架 + 要点分配 + 篇幅分配"),
        StageNode(code="D3", name="图表规范", run=d3_figures,
                  description="编号/引用分析 + 具体修复建议"),
        StageNode(code="D2", name="分章写作", run=d2_sections,
                  requires_model=generator is not None,
                  description="逐节生成 + 证据绑定（缺省给骨架）"),
        StageNode(code="D4", name="文献引用", run=d4_citations,
                  description="正文引用必须来自 B 阶段的文献库"),
    )


def run_writing(
    *,
    title: str,
    draft: str = "",
    design: Mapping[str, Any] | None = None,
    generator: SectionGenerator | None = None,
    required_sections: list[str] | None = None,
    entries: Sequence[BibEntry] = (),
    label: str = "<inline>",
    only: Sequence[str] | None = None,
    on_node: Callable[[str, Any], None] | None = None,
):
    """**D 阶段的对外入口。** 独立调用，不需要先跑 A/B/C。

    - 给了 ``draft`` 才跑 D3（图表检查）与 D4（引用来源核对）；
    - 给了 ``design`` 才分配要点；
    - 给了 ``entries``（B 阶段文献库）才能核对引用来源。
    """
    nodes = build_writing_nodes(
        design=design, generator=generator,
        required_sections=required_sections, entries=entries,
    )
    return run_stage(
        stage=STAGE_KEY,
        label=label or title[:48],
        nodes=nodes,
        state={"title": title, "draft": draft, "label": label},
        only=only,
        on_node=on_node,
    )
