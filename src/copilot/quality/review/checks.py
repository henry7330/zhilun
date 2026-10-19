"""E6 · 审稿检查规则。

与 E1 的关系：**复用而非重写**
------------------------------
E1 已经把文档解析成了结构化模型（章节树、参考文献、图表题注、交叉引用），
E6 直接用它，不再自己写一遍解析。

**判据**：同一件事只能有一份实现。两份解析器迟早会不一致，
而那种不一致极难发现 —— 这与 E3/E4 共用引用判定是同一个理由。

与 E1 的分工
------------
======================  ==========================  ==========================
                        E1（格式视角）                E6（内容视角）
======================  ==========================  ==========================
关注的                  "符不符合模板"                "写得对不对、一致不一致"
例                      缺「结论」章节                摘要里说的术语正文里没有
例                      参考文献格式不对              同一概念用了两种叫法
======================  ==========================  ==========================

**两者共用同一套 Issue 词汇**，所以输出可以合并成一份质检报告。

E6 只做**可判定**的部分
-----------------------
结构缺失、术语不一致、关键词缺失、图表引用断裂 —— 这些都能规则判定。
"论证链是否断裂"需要跨句语义理解，属模型范畴，本文件**不假装能做**。
"""

from __future__ import annotations

import re
from collections import defaultdict

from copilot.quality.format.docmodel import DocModel, Section
from copilot.quality.format.report import Issue, Severity
from copilot.quality.format.rules.figures import check_figures
from copilot.quality.format.template import Template

__all__ = [
    "check_terms",
    "check_keywords",
    "check_section_length",
    "check_figure_refs",
    "REVIEW_RULES",
    "SYNONYM_GROUPS",
    "KEYWORD_LINE",
]

RULE = "review"

#: 同义术语组：**同一个概念**在中文论文里常见的不同叫法。
#:
#: ⚠️ 刻意只收"大多数学科内视为同义"的组，不收有争议的
#: （例如「准确率 / 精确率」在信息检索里是**两个不同指标**，不能混为一组）。
#: 检测到多种叫法只提示"确认是否统一"，**不判定对错** ——
#: 作者可能是有意区分，机器不该替他决定。
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("召回率", "查全率"),
    ("精确率", "查准率"),
    ("准确率", "正确率"),
    ("鲁棒性", "健壮性"),
    ("数据集", "数据集合"),
    ("特征提取", "特征抽取"),
    ("超参数", "超参"),
    ("端到端", "端到端式"),
)

#: 摘要里的关键词行
KEYWORD_LINE = re.compile(r"^\s*(?:\*\*)?\s*关键词\s*(?:\*\*)?\s*[:：]\s*(.+)$", re.MULTILINE)

#: 英文关键词行
KEYWORD_LINE_EN = re.compile(r"^\s*(?:\*\*)?\s*keywords?\s*(?:\*\*)?\s*[:：]\s*(.+)$", re.MULTILINE | re.IGNORECASE)

#: 章节"过短"的默认阈值（正文字数）。低于此值的章节通常论述不充分。
#:
#: 可配置：不同学校/不同篇幅要求的论文差异很大。
SHORT_SECTION_CHARS = 120

#: **不参与篇幅检查**的章节。
#:
#: 这几类章节本来就短：参考文献是条目列表、致谢是礼节性文字、
#: 摘要有独立的字数上限。对它们套用"论述不充分"的判断只会制造噪声。
_LENGTH_EXEMPT_TITLES = frozenset(
    {
        "参考文献",
        "references",
        "bibliography",
        "致谢",
        "acknowledgements",
        "acknowledgments",
        "摘要",
        "abstract",
        "附录",
        "appendix",
    }
)


def _split_keywords(raw: str) -> list[str]:
    return [k.strip(" .。；;，,") for k in re.split(r"[；;，,、]", raw) if k.strip(" .。；;，,")]


def check_terms(doc: DocModel) -> list[Issue]:
    """REV001 术语一致性。

    **完全可判定，而且人工几乎不可能全文核对** —— 一个人写到后面
    很自然会把"召回率"写成"查全率"，自己读一遍也发现不了。
    """
    issues: list[Issue] = []
    body = doc.body_text

    for group in SYNONYM_GROUPS:
        used = [t for t in group if t in body]
        if len(used) >= 2:
            first_line = next(
                (
                    i
                    for i, line in enumerate(doc.lines)
                    if any(t in line for t in used)
                ),
                0,
            )
            issues.append(
                Issue(
                    code="REV001",
                    severity=Severity.WARNING,
                    rule=RULE,
                    line=first_line,
                    message=(
                        f"同一概念出现了多种叫法：{'、'.join(used)}"
                    ),
                    fix=(
                        f"确认是否需要统一（如果是有意区分，请在各处明确定义差异）；"
                        f"建议全文统一为一种，或在首次出现时说明二者关系"
                    ),
                )
            )
    return issues


def check_keywords(doc: DocModel) -> list[Issue]:
    """REV002 摘要关键词未在正文出现。

    摘要里承诺的关键词，正文里一次都没出现 —— 通常意味着：
    摘要与正文脱节，或者关键词选得不合适（这两个都是审稿人会指出的问题）。
    """
    issues: list[Issue] = []

    raw = None
    for pattern in (KEYWORD_LINE, KEYWORD_LINE_EN):
        m = pattern.search(doc.body_text)
        if m:
            raw = m.group(1)
            break
    if not raw:
        return issues

    keywords = _split_keywords(raw)
    if not keywords:
        return issues

    # 正文 = 去掉摘要部分
    abstract_titles = {"摘要", "abstract"}
    body_without_abstract = "\n".join(
        line
        for i, line in enumerate(doc.lines)
        if not any(
            s.line <= i < s.span[1]
            for s in doc.flat_sections()
            if s.title.strip().lower() in abstract_titles
        )
    )

    missing = [k for k in keywords if k and k not in body_without_abstract]
    if missing:
        issues.append(
            Issue(
                code="REV002",
                severity=Severity.WARNING,
                rule=RULE,
                line=0,
                message=f"摘要把这些列为关键词，但正文中从未出现：{'、'.join(missing)}",
                fix="要么在正文中实际讨论它们，要么从关键词里去掉（关键词应当反映正文内容）",
            )
        )
    return issues


#: 常见章节名 —— 用来判断"哪一层才是章"。
#:
#: 为什么需要它：论文有两种写法，**光看层级分不出来**：
#:
#: ::
#:
#:     # 论文题目          ← 顶层是"题目"          # 一                ← 顶层是"章"
#:     ## 摘要            ← 第二层才是"章"        ## 一点一            ← 第二层是"小节"
#:     ## 结论                                     ## 一点二
#:
#: 两种结构下"顶层只有 1 个节点"。所以不能用节点数量判断，
#: 而要看**标题像不像章节名**。
COMMON_SECTION_NAMES: frozenset[str] = frozenset(
    {
        "摘要", "abstract", "绪论", "引言", "前言", "概述",
        "文献综述", "相关工作", "研究现状", "背景",
        "研究方法", "方法", "方法设计", "实验", "实验与分析", "实验设计",
        "结果", "结果与分析", "讨论", "分析与讨论",
        "结论", "总结", "总结与展望", "结语",
        "参考文献", "references", "致谢", "acknowledgements",
        "附录", "appendix",
    }
)


def _resolve_chapter_level(doc: DocModel, template: Template | None) -> list[Section]:
    """判断应该检查哪一层的章节。

    规则（按优先级）：

    1. 顶层有**多个**节点 → 顶层就是章（最常见，也最没有歧义）；
    2. 顶层只有 1 个节点：
       - 它的标题**不是**已知章节名，而子节点里有 → 它是文档题目，**下沉一层**；
       - 否则 → 顶层就是章（"章 + 小节"的结构）。

    ★ 这个 bug 是建评测集才发现的：早期实现直接取 ``doc.sections``，
    于是 ``# 题目`` + ``## 章节`` 这种写法下**篇幅检查被完全静默跳过** ——
    不报错、不提示，看起来像"篇幅都合格"。
    """
    sections = doc.sections
    if len(sections) != 1 or not sections[0].children:
        return sections

    top = sections[0]
    children = top.children

    required = {t.strip().lower() for t in (template.required_sections if template else ())}
    top_title = top.title.strip().lower()

    def is_chapter_name(title: str) -> bool:
        return title in COMMON_SECTION_NAMES or title in required

    # 顶层本身就是章节名 → 顶层是章（例如 `# 摘要` + `## 1.1 …`）
    if is_chapter_name(top_title):
        return sections

    # 子节点有章节名，而顶层没有 → 顶层是文档题目，真正的章在下一层
    if any(is_chapter_name(c.title.strip().lower()) for c in children):
        return list(children)

    return sections


def check_section_length(
    doc: DocModel,
    *,
    min_chars: int = SHORT_SECTION_CHARS,
    template: Template | None = None,
) -> list[Issue]:
    """REV003 章节过短。

    **只报"章"这一级**：小节的篇幅本来就短，对小节设阈值只会制造噪声。

    哪一层算"章"由 :func:`_resolve_chapter_level` 判断；
    参考文献/致谢/摘要等章节按性质豁免（见 ``_LENGTH_EXEMPT_TITLES``）。
    """
    issues: list[Issue] = []
    for s in _resolve_chapter_level(doc, template):
        if any(t in s.title.strip().lower() for t in _LENGTH_EXEMPT_TITLES):
            continue
        start, end = s.span
        chars = sum(
            len(line.strip())
            for line in doc.lines[start + 1 : end]
            if line.strip() and not line.lstrip().startswith("#")
        )
        if 0 < chars < min_chars:
            issues.append(
                Issue(
                    code="REV003",
                    severity=Severity.INFO,
                    rule=RULE,
                    line=s.line,
                    message=f"章节「{s.title}」正文仅 {chars} 字，可能论述不充分",
                    fix="补充内容；若该章本来只是过渡，可考虑并入相邻章节",
                )
            )
    return issues


def check_figure_refs(doc: DocModel, template: Template) -> list[Issue]:
    """图表引用一致性 —— **直接复用 E1 的规则**。

    同一能力不该实现两遍：E1 是"格式视角"（编号规范不规范），
    E6 是"内容视角"（引用有没有断），但**判定逻辑完全相同**。
    """
    return check_figures(doc, template)


#: 默认规则集。签名与 E1 一致，便于合并成一份报告。
REVIEW_RULES = (
    check_terms,
    check_keywords,
    check_section_length,
)
