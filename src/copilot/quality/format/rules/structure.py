"""子部分 ③ 结构校验规则。

检查什么：必需章节是否齐全、顺序是否正确、层级是否过深、字数是否达标、章节是否为空。

为什么这一块最该自动化
----------------------
「有没有『结论』这一章」是**客观事实**，不需要任何智能判断。
而它恰恰是论文被打回的最常见原因之一。

凡是"完全可判定"的检查，都不应该交给模型 ——
规则版更快、免费、可复现、可解释（能说清为什么报错），
而模型可能今天是这个答案、明天是那个答案。
"""

from __future__ import annotations

import re

from copilot.quality.format.docmodel import DocModel
from copilot.quality.format.report import Issue, Severity
from copilot.quality.format.template import Template

__all__ = ["check_structure", "normalize_heading"]

RULE = "structure"

#: 章节标题里的编号前缀。
#:
#: ⚠️ 「第X章」这一类**必须覆盖**（实测踩过）：中文论文大量使用
#: 「第一章 绪论」而不是「1 绪论」。漏掉它会让**所有用「第X章」的论文
#: 全部误报为"章节缺失"** —— 一个模板匹配问题会伪装成 5 条内容问题。
_NUM_PREFIX = re.compile(
    r"^(?:"
    r"第\s*[一二三四五六七八九十百千\d]+\s*[章节篇]|"   # 第一章 / 第 2 节 / 第三篇
    r"第\s*[一二三四五六七八九十百千\d]+\s*部分|"        # 第一部分
    r"\d+(?:\.\d+)*|"                                   # 1 / 1.2 / 1.2.3
    r"[一二三四五六七八九十]+|"                          # 一、
    r"[（(]\s*\d+\s*[)）]"                               # （2）
    r")[.、,，:：\s]*"
)


def normalize_heading(title: str) -> str:
    """归一化章节标题，用于与模板比对。

    「2.1 文献综述」与「文献综述」必须视为同一章节 —— 否则模板里写
    「文献综述」就永远匹配不上带编号的标题，全部误报为"缺失"。
    这是此类校验最常见的假阳性来源。
    """
    t = _NUM_PREFIX.sub("", title.strip())
    return t.strip(" .:：·-—").lower()


def check_structure(doc: DocModel, template: Template) -> list[Issue]:
    issues: list[Issue] = []
    flat = doc.flat_sections()
    titles = [normalize_heading(s.title) for s in flat]

    # ── SEC001 必需章节缺失 ──
    missing: list[str] = []
    for required in template.required_sections:
        if normalize_heading(required) not in titles:
            missing.append(required)
            issues.append(
                Issue(
                    code="SEC001",
                    severity=Severity.ERROR,
                    rule=RULE,
                    message=f"缺少必需章节「{required}」",
                    fix=f"补充「{required}」章节；若学校要求不同，请修改模板配置",
                )
            )

    # ── SEC002 章节顺序不符 ──
    # 只在章节齐全时判断：有缺失时顺序比较没有意义，只会产生连锁误报
    if template.section_order_strict and not missing:
        positions = [titles.index(normalize_heading(r)) for r in template.required_sections]
        for i in range(len(positions) - 1):
            if positions[i] > positions[i + 1]:
                issues.append(
                    Issue(
                        code="SEC002",
                        severity=Severity.ERROR,
                        rule=RULE,
                        line=flat[positions[i]].line,
                        message=(
                            f"章节顺序不符：「{template.required_sections[i]}」"
                            f"出现在「{template.required_sections[i + 1]}」之后"
                        ),
                        fix="按模板要求的顺序调整章节；若顺序无要求，可在模板中关闭严格顺序",
                    )
                )
                break  # 只报第一处，避免同一根因刷屏

    # ── SEC003 层级过深 ──
    for s in flat:
        if s.level > template.max_heading_level:
            issues.append(
                Issue(
                    code="SEC003",
                    severity=Severity.WARNING,
                    rule=RULE,
                    line=s.line,
                    message=(
                        f"标题层级过深（{s.level} 级）：{s.title}"
                        f"；模板上限为 {template.max_heading_level} 级"
                    ),
                    fix="合并或上提该小节；过深层级在目录里会显得零碎",
                )
            )

    # ── SEC004 正文字数不足 ──
    if template.min_body_words is not None:
        words = doc.body_words
        if words < template.min_body_words:
            issues.append(
                Issue(
                    code="SEC004",
                    severity=Severity.WARNING,
                    rule=RULE,
                    message=(
                        f"正文字数不足：{words} 字，要求至少 {template.min_body_words} 字"
                    ),
                    fix="补充论述；注意字数统计已排除参考文献部分",
                )
            )

    # ── SEC005 空章节（**只看叶子章节**）──
    # 有子章节的章节不参与本检查：它的内容在子章节里，
    # 「某一章直接进入 2.1 / 2.2，自己没有引言段」是完全常见的合法写法，
    # 报它只会制造噪声 —— 而噪声会让用户不再相信报告里的其它条目。
    for s in flat:
        if s.children:
            continue
        content_lines = doc.lines[s.line + 1 : min(s.span[1], len(doc.lines))]
        if not any(ln.strip() for ln in content_lines):
            issues.append(
                Issue(
                    code="SEC005",
                    severity=Severity.INFO,
                    rule=RULE,
                    line=s.line,
                    message=f"章节「{s.title}」没有正文内容",
                    fix="补充内容，或删除该空章节",
                )
            )

    return issues
