"""E 阶段 · Agent「格式规整」—— 一个可以独立调用的服务。

## 这个 Agent 做什么

把一份 Markdown 草稿按**学校/期刊的格式要求**规整成可交付的稿子。

```
E1 格式规整   ──▶ 按用户给的格式要求改文件（LLM 一段式）+ diff
E2 文本一致性 ──▶ 术语一致 / 关键词覆盖 / 章节篇幅 / 图表引用
E3 导出       ──▶ BibTeX / GB-T 7714 / LaTeX + **回读一致性校验**
```

## 关于 E1 的「一段式」

按你的选择：**LLM 直接读格式要求并输出改好的文件**，不先产出结构化规范。

代价是改完没有天然判据。所以本节点内置两样东西来兜底：

1. **改前快照 + diff**：告诉你哪些行变了、变成什么；
2. **回读校验**：改完用**确定性规则**再扫一遍，看问题数是否真的减少。

**它不是"完全不管"，而是"事后能查"。**

## 关于「用户没提到就不改」

严格执行：`E1` 只按 `requirement` 里出现的项动手，其余不碰。
但会在报告里列出**未覆盖项**（常见格式项里你没要求的），避免"规整过了 = 格式没问题"的错觉。

## 关于「不做引用真实性核查」

按你的要求，本阶段**不核查引用是否真实**。
那件事在 B 阶段（文献检索）做过了 —— B3 的 `verify_citation`。
D 阶段的「文献引用」节点又限制了引用只能来自 B 的文献库。两道关，不需要第三道。

## 怎么用（独立调用）

```python
from copilot.format_stage import run_format

report = run_format(
    draft=open("thesis.md", encoding="utf-8").read(),
    requirement=open("school_format.txt", encoding="utf-8").read(),
    formatter=my_llm_formatter,        # (draft, requirement) -> 新 draft
)
print(report.summary_line())
```
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from copilot.contracts import Finding
from copilot.quality.format.report import Severity
from copilot.stage import StageNode, ok_outcome, run_stage

__all__ = ["run_format", "build_format_nodes", "FormatFormatter"]

STAGE_KEY = "E"
CODE_NORMALIZE = "E1-NORMALIZE"
CODE_CONSISTENCY = "E2-CONSISTENCY"
CODE_EXPORT = "E3-EXPORT"

#: 格式规整器签名：(draft, requirement) -> 规整后的 draft
FormatFormatter = Callable[[str, str], str]

#: 常见格式项 —— 用来提示"你哪些没要求"。
#:
#: 这不是要全查，而是要**如实告诉用户没查什么**。
COMMON_FORMAT_ITEMS: tuple[str, ...] = (
    "标题层级与编号",
    "正文字体/字号/行距",
    "摘要与关键词格式",
    "图表编号与题注",
    "参考文献著录格式",
    "页眉页脚与页码",
    "公式编号",
    "章节分页",
    "附录格式",
    "致谢格式",
)


def _diff_summary(before: str, after: str, *, limit: int = 12) -> list[str]:
    """逐行对比，返回"变了哪些行"的摘要。

    只报**改了内容的行**，不报纯换行差异 —— 后者噪声太大。
    """
    a = before.splitlines()
    b = after.splitlines()
    if a == b:
        return []
    out: list[str] = [f"行数 {len(a)} → {len(b)}"]
    import difflib

    changed = 0
    for line in difflib.unified_diff(a, b, lineterm="", n=0):
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith(("+", "-")):
            changed += 1
            if changed <= limit:
                out.append(f"{line[0]} {line[1:][:70]}")
    if changed > limit:
        out.append(f"…另有 {changed - limit} 处改动")
    return out


def build_format_nodes(
    *,
    draft: str,
    requirement: str = "",
    formatter: FormatFormatter | None = None,
    template: Any = None,
    anonymity_required: bool = False,
) -> tuple[StageNode, ...]:
    """装配三个节点：**规整 → 一致性 → 导出**。"""

    # ── E1 格式规整 ──

    def e1_normalize(state: dict[str, Any]):
        from copilot.delivery.prcheck import check_submission

        # ① 内容卫生检查（占位符/敏感信息/匿名性）—— 无论是否接入 LLM 都要跑
        hygiene = check_submission(draft, anonymity_required=anonymity_required)
        findings: list[Finding] = [
            Finding(
                node="E1",
                code=i.code,
                severity=i.severity,
                message=i.message,
                location=i.location,
                fix=i.fix,
            )
            for i in hygiene.issues
            if i.code.startswith("PRE")
        ]

        result_text = draft
        diff: list[str] = []

        # ② 有 LLM 且给了格式要求 → 一段式改文件
        if formatter is not None and (requirement or "").strip():
            try:
                result_text = formatter(draft, requirement)
            except Exception as exc:  # noqa: BLE001
                findings.append(
                    Finding(
                        node="E1",
                        code=CODE_NORMALIZE,
                        severity=Severity.ERROR,
                        message=f"格式规整调用失败：{type(exc).__name__}",
                        fix="检查模型配置后重试；本次未改动文件",
                    )
                )
                result_text = draft
            else:
                diff = _diff_summary(draft, result_text)
                # ③ 回读校验：改完再用确定性规则扫一遍
                after = check_submission(result_text, anonymity_required=anonymity_required)
                before_n = len([i for i in hygiene.issues if i.code.startswith("PRE")])
                after_n = len([i for i in after.issues if i.code.startswith("PRE")])
                findings.append(
                    Finding(
                        node="E1",
                        code=CODE_NORMALIZE,
                        severity=Severity.INFO,
                        message=(
                            f"已按格式要求规整：{len(diff)} 处差异；"
                            f"内容卫生问题 {before_n} → {after_n}"
                        ),
                        fix="核对下面的 diff —— 尤其留意非格式类的改动",
                        payload={"diff": diff},
                    )
                )
                if after_n >= before_n and before_n > 0:
                    findings.append(
                        Finding(
                            node="E1",
                            code=CODE_NORMALIZE,
                            severity=Severity.WARNING,
                            message="规整后内容卫生问题没有减少",
                            fix="可能 LLM 没按要求改，或要求里没涉及这些项",
                        )
                    )
        else:
            findings.append(
                Finding(
                    node="E1",
                    code=CODE_NORMALIZE,
                    severity=Severity.INFO,
                    message=(
                        "未做版式规整"
                        + ("（未提供格式要求）" if not (requirement or "").strip() else "（未接入 LLM）")
                    ),
                    fix="提供学校格式要求文本并接入 formatter，即可一段式改文件",
                )
            )

        # ④ 未覆盖项清单 —— 防"规整过了 = 格式没问题"的错觉
        covered = [
            item for item in COMMON_FORMAT_ITEMS
            if any(k in (requirement or "") for k in (item[:2], item))
        ] if requirement else []
        uncovered = [i for i in COMMON_FORMAT_ITEMS if i not in covered]
        if uncovered:
            findings.append(
                Finding(
                    node="E1",
                    code=CODE_NORMALIZE,
                    severity=Severity.INFO,
                    message="以下常见格式项你的要求里没提到，**本次未处理**：" + "、".join(uncovered),
                    fix="需要的话把它们补进格式要求再跑一次",
                )
            )

        return ok_outcome(
            findings=findings,
            stats={
                "changed": len(diff),
                "formatter": "llm" if formatter is not None else "none",
                "uncovered_items": len(uncovered),
            },
            notes=[
                f"内容卫生问题 {len([i for i in hygiene.issues if i.code.startswith('PRE')])} 条",
                "**只改格式，不改内容** —— diff 里的非格式改动应当被拒绝",
            ],
            patch={"normalized": result_text, "diff": diff},
        )

    # ── E2 文本一致性 ──

    def e2_consistency(state: dict[str, Any]):
        from copilot.quality.format import parse_markdown
        from copilot.quality.review import review_document

        text = state.get("normalized") or draft
        doc = parse_markdown(text, path=state.get("label", "<inline>"))
        report = review_document(doc, template)

        findings = [
            Finding(
                node="E2",
                code=i.code,
                severity=i.severity,
                message=i.message,
                location=i.location,
                fix=i.fix,
            )
            for i in report.issues
        ]
        return ok_outcome(
            findings=findings,
            stats={"issues": len(report.issues)},
            notes=["术语一致性 / 关键词覆盖 / 章节篇幅 / 图表引用"],
            patch={"consistency_report": report},
        )

    # ── E3 导出 ──

    def e3_export(state: dict[str, Any]):
        from copilot.delivery.export import export_paper
        from copilot.quality.format import parse_markdown

        text = state.get("normalized") or draft
        doc = parse_markdown(text, path=state.get("label", "<inline>"))
        result = export_paper(doc)

        findings: list[Finding] = []
        if not result.ok:
            for check in (result.latex_check, result.bib_check):
                if check is None or check.ok:
                    continue
                for d in check.diffs:
                    findings.append(
                        Finding(
                            node="E3",
                            code=CODE_EXPORT,
                            severity=Severity.ERROR,
                            message=f"导出一致性校验未通过：{d}",
                            fix="产物与原稿不一致 —— 先查导出链路，再交付",
                        )
                    )

        return ok_outcome(
            findings=findings,
            stats={
                "entries": len(result.entries),
                "tex_chars": len(result.tex),
                "ok": result.ok,
            },
            notes=[f"BibTeX {len(result.entries)} 条 / LaTeX {len(result.tex)} 字符"],
            patch={"export_result": result},
        )

    return (
        StageNode(code="E1", name="格式规整", run=e1_normalize,
                  requires_model=formatter is not None,
                  description="按格式要求改文件（LLM 一段式）+ diff + 未覆盖项"),
        StageNode(code="E2", name="文本一致性", run=e2_consistency,
                  description="术语 / 关键词 / 篇幅 / 图表引用"),
        StageNode(code="E3", name="导出", run=e3_export,
                  description="BibTeX / GB-T 7714 / LaTeX + 回读校验"),
    )


def run_format(
    *,
    draft: str,
    requirement: str = "",
    formatter: FormatFormatter | None = None,
    template: Any = None,
    anonymity_required: bool = False,
    label: str = "<inline>",
    only: Sequence[str] | None = None,
    on_node: Callable[[str, Any], None] | None = None,
):
    """**E 阶段的对外入口。** 独立调用，不需要先跑 A–D。"""
    if not (draft or "").strip():
        raise ValueError("draft 为空，没有可规整的内容")

    nodes = build_format_nodes(
        draft=draft, requirement=requirement, formatter=formatter,
        template=template, anonymity_required=anonymity_required,
    )
    return run_stage(
        stage=STAGE_KEY,
        label=label,
        nodes=nodes,
        state={"draft": draft, "requirement": requirement, "label": label},
        only=only,
        on_node=on_node,
    )
