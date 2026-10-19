"""F1 · 导出一致性校验 —— **把产物读回来，与原稿对比**。

为什么必须"回读"
----------------
**"导出后丢东西"是最常见的交付事故**：漏了一章、少了一个图、
参考文献少了几条 —— 而且**肉眼很难发现**（谁会去数 chapter 数量？）。

**为什么不能在生成时自检**：
生成时自检只能验证"我按逻辑写了 N 个 ``\\section``"，
而回读验证的是"**文件里真的有 N 个 ``\\section``**"。

后者才能查出写入截断、编码问题、模板吞内容、转义把内容吃掉这类问题。

> 这与本项目的核心主张一致：**可验证的意思是"从产物反查"，不是"我相信我写对了"。**

校验什么
--------
========================  ==========================================
校验项                     怎么比
========================  ==========================================
章节数                     原稿节点数 vs ``\\section`` + ``\\subsection`` + ...
每章标题                   原稿标题（去编号、去转义）vs ``\\section{...}`` 里的文本
图表数                     原稿题注数 vs ``\\begin{figure}`` + ``\\begin{table}``
参考文献数                 原稿条目数 vs ``@article`` 等条目数
========================  ==========================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from copilot.quality.format.docmodel import DocModel

__all__ = [
    "ExportDiff",
    "ExportVerification",
    "verify_latex",
    "verify_bibtex",
    "SECTION_COMMANDS",
]

#: 会被计入"章节数"的 LaTeX 命令
SECTION_COMMANDS: tuple[str, ...] = (
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
)

_SECTION_RE = re.compile(
    r"\\(" + "|".join(SECTION_COMMANDS) + r")\*?\{([^}]*)\}"
)
_FIGURE_RE = re.compile(r"\\begin\{figure\}")
_TABLE_RE = re.compile(r"\\begin\{table\}")
_BIB_ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,]+),")


@dataclass
class ExportDiff:
    """一项差异。"""

    item: str
    expected: object
    actual: object
    note: str = ""

    def __str__(self) -> str:
        return f"{self.item}: 期望 {self.expected}，实际 {self.actual}" + (
            f"（{self.note}）" if self.note else ""
        )


@dataclass
class ExportVerification:
    """一次导出校验的结果。"""

    ok: bool
    diffs: list[ExportDiff] = field(default_factory=list)
    #: 实际统计到的数字，便于报告展示与人工复核
    stats: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        if self.ok:
            return "导出校验通过：" + "，".join(f"{k}={v}" for k, v in self.stats.items())
        return "导出校验未通过：" + "；".join(str(d) for d in self.diffs)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stats": dict(self.stats),
            "diffs": [
                {"item": d.item, "expected": d.expected, "actual": d.actual, "note": d.note}
                for d in self.diffs
            ],
        }


def _strip_number(title: str, number: str | None) -> str:
    if not number:
        return title
    return title[len(number):].lstrip(" .、,，:：") if title.startswith(number) else title


def verify_latex(
    tex: str, doc: DocModel, *, document_title: str | None = None
) -> ExportVerification:
    """校验 LaTeX 骨架与原稿结构是否一致。

    ``document_title``：被取作文档名的那个标题（渲染成 ``\\title{}``，
    不会成为 ``\\section``）。**不传会导致"章节标题丢失"的误报** ——
    校验器必须知道哪些标题本来就不该出现在正文里。
    """
    diffs: list[ExportDiff] = []

    # ── 章节数 ──
    sections_found = _SECTION_RE.findall(tex)
    expected_sections = len(doc.flat_sections())
    stats = {
        "sections": len(sections_found),
        "figures": len(_FIGURE_RE.findall(tex)),
        "tables": len(_TABLE_RE.findall(tex)),
    }

    # 被取作文档名的那个标题不属于正文结构。
    #
    # ★ 必须同时要求它**有子章节**：文档用扁平 `#` 标题时，第一个标题
    # 只是第一章（如「摘要」），它下面没有子章节 —— 此时它照样会被渲染成
    # `\section`，不能被减掉。少了这个条件会少算一章，报出假错。
    title_section_skipped = (
        bool(document_title)
        and bool(doc.sections)
        and doc.sections[0].title.strip() == document_title.strip()
        and bool(doc.sections[0].children)
    )
    if title_section_skipped:
        expected_sections -= 1

    if len(sections_found) != expected_sections:
        diffs.append(
            ExportDiff(
                "章节数",
                expected_sections,
                len(sections_found),
                "产物里的 \\section 系列命令数量与原稿章节数不一致，可能有章节被吞掉",
            )
        )

    # ── 图表数 ──
    expected_figures = len([c for c in doc.captions if c.kind == "figure"])
    expected_tables = len([c for c in doc.captions if c.kind == "table"])

    if stats["figures"] != expected_figures:
        diffs.append(
            ExportDiff("图数量", expected_figures, stats["figures"], "figure 环境数不符")
        )
    if stats["tables"] != expected_tables:
        diffs.append(
            ExportDiff("表数量", expected_tables, stats["tables"], "table 环境数不符")
        )

    # ── 标题内容 ──
    # 转义会改变文本（& → \&），所以比对前要把产物侧还原回来
    produced_titles = {_unescape(t).strip() for _cmd, t in sections_found}
    sections = doc.flat_sections()
    check_from = 1 if title_section_skipped else 0
    for section in sections[check_from:]:
        expected = _strip_number(section.title, section.number).strip()
        if not expected:
            continue
        if expected not in produced_titles:
            diffs.append(
                ExportDiff("章节标题", expected, "（产物中未找到）", "标题在导出后丢失或被改写")
            )

    return ExportVerification(ok=not diffs, diffs=diffs, stats=stats)


def _unescape(text: str) -> str:
    """把 LaTeX 转义还原，用于文本比对。"""
    back = {
        r"\&": "&",
        r"\%": "%",
        r"\$": "$",
        r"\#": "#",
        r"\_": "_",
        r"\{": "{",
        r"\}": "}",
        r"\textasciitilde{}": "~",
        r"\textasciicircum{}": "^",
        r"\textbackslash{}": "\\",
    }
    out = text
    # 先还原长的（\textasciitilde{} 等），再还原两字符的，避免前缀冲突
    for safe in sorted(back, key=len, reverse=True):
        out = out.replace(safe, back[safe])
    return out


def verify_bibtex(bib_text: str, doc: DocModel) -> ExportVerification:
    """校验 ``.bib`` 产物与原文参考文献条数是否一致。"""
    diffs: list[ExportDiff] = []
    entries = _BIB_ENTRY_RE.findall(bib_text)
    # @string / @comment / @preamble 不算条目
    entries = [e for e in entries if e[0].lower() not in ("string", "comment", "preamble")]

    expected = len(doc.references)
    stats = {"bib_entries": len(entries)}

    if len(entries) != expected:
        diffs.append(
            ExportDiff(
                "参考文献条数",
                expected,
                len(entries),
                "条目数不一致，可能有文献在导出时丢失",
            )
        )

    keys = [k.strip() for _t, k in entries]
    if len(keys) != len(set(keys)):
        duplicated = sorted({k for k in keys if keys.count(k) > 1})
        diffs.append(
            ExportDiff(
                "引用键唯一性",
                "无重复",
                f"重复：{', '.join(duplicated)}",
                "重复的键会让 \\cite 指向错误文献",
            )
        )

    return ExportVerification(ok=not diffs, diffs=diffs, stats=stats)
