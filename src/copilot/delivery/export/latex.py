"""F1 · LaTeX 骨架生成。

只生成骨架，不生成正文
----------------------
正文是用户的，导出工具不该碰。

骨架保证的是**结构与原文一致**：章节层级、图表位置、参考文献挂载点。
正文部分留 ``% TODO: 正文`` 占位，由用户填。

为什么需要它
------------
手搭 LaTeX 骨架是纯重复劳动，而且**容易与原文结构不一致** ——
改了章节忘了改 ``\\section`` 层级，是常见错误。
"""

from __future__ import annotations

from typing import Sequence

from copilot.delivery.export.bib import escape_latex
from copilot.quality.format.docmodel import Caption, DocModel, Section

__all__ = ["to_latex_skeleton", "resolve_title", "DEFAULT_PREAMBLE"]


def resolve_title(doc: DocModel, title: str | None = None) -> str:
    """决定文档标题：显式传入 > 第一个**带子章节**的一级标题 > 占位名。

    单独抽出来，是因为 ``verify.py`` 必须知道**哪个标题被取作了文档名** ——
    否则它会把文档名也算作"应该出现 ``\\section``"的章节，从而误报。

    ★ 为什么要求"带子章节"：中文论文常见的写法是扁平标题

    ::

        # 摘要        ← 这是一章，不是书名
        # 方法
        # 结论

    如果只看"第一个标题"，就会把「摘要」当成文档名，进而跳到它的子章节 ——
    而它根本没有子章节，结果**后面所有章节全被丢掉**（实测过：
    3 个章节一个都没渲染出来）。
    """
    if title:
        return title
    if doc.sections and doc.sections[0].title.strip() and doc.sections[0].children:
        return doc.sections[0].title
    return "（待填论文标题）"

#: 默认导言区。用 ``ctexart`` 以支持中文；
#: 需要的宏包只放**骨架必须用到**的几个（graphicx 用于图表，amsmath 用于公式）。
DEFAULT_PREAMBLE = r"""\documentclass[12pt]{ctexart}

\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}
\usepackage{geometry}
\geometry{a4paper, margin=2.5cm}

\title{TITLE_PLACEHOLDER}
\author{%
% TODO: 作者
}
\date{\today}

"""

#: 标题层级 → LaTeX 命令
_LEVEL_CMD = {1: "section", 2: "subsection", 3: "subsubsection", 4: "paragraph"}


def _direct_captions(section: Section, doc: DocModel) -> list[Caption]:
    """本节的**直接**图表（不含子章节里的）。

    为什么按"直接"划分：否则一个章下的所有图都会被算进第一个子节，
    生成出来的顺序会乱。
    """
    child_ranges = [(c.line, c.span[1]) for c in section.children]
    start, end = section.span
    out: list[Caption] = []
    for cap in doc.captions:
        if not (start <= cap.line < end):
            continue
        if any(a <= cap.line < b for a, b in child_ranges):
            continue
        out.append(cap)
    return sorted(out, key=lambda c: c.line)


def _render_caption(cap: Caption) -> str:
    """图表占位。图片本身留空，由用户提供。"""
    if cap.kind == "table":
        return (
            "\\begin{table}[htbp]\n"
            "  \\centering\n"
            "  % TODO: 表格内容\n"
            f"  \\caption{{{escape_latex(cap.text or '（缺题注）')}}}\n"
            "\\end{table}\n"
        )
    return (
        "\\begin{figure}[htbp]\n"
        "  \\centering\n"
        "  % TODO: \\includegraphics[width=0.8\\textwidth]{figures/xxx.png}\n"
        f"  \\caption{{{escape_latex(cap.text or '（缺题注）')}}}\n"
        "\\end{figure}\n"
    )


def _render_section(section: Section, doc: DocModel, level_offset: int) -> list[str]:
    """渲染一个章节。

    ``level_offset`` 用来处理「一级标题被取作文档名」的情况：
    此时正文里的 ``## 章节`` 应上移一级渲染成 ``\\section``，
    否则会变成悬空的 ``\\subsection``（没有父节）。
    """
    effective = max(1, section.level + level_offset)
    cmd = _LEVEL_CMD.get(effective, "paragraph")

    # 去掉标题里的编号，交给 LaTeX 自动编号（避免「1.1 1.1 小节」双重编号）
    title = section.title
    if section.number:
        title = title[len(section.number):].lstrip(" .、,，:：")
    lines = [f"\\{cmd}{{{escape_latex(title)}}}", ""]

    for cap in _direct_captions(section, doc):
        lines.append(_render_caption(cap))

    if not section.children:
        lines += ["% TODO: 正文", ""]

    for child in section.children:
        lines += _render_section(child, doc, level_offset)

    return lines


def to_latex_skeleton(
    doc: DocModel,
    *,
    title: str | None = None,
    preamble: str = DEFAULT_PREAMBLE,
    bibliography: str = "refs",
) -> str:
    """把文档结构转成 LaTeX 骨架。

    ``title`` 为空时取文档里第一个一级标题。
    """
    resolved_title = resolve_title(doc, title)

    lines: list[str] = [preamble.replace("TITLE_PLACEHOLDER", escape_latex(resolved_title))]
    lines.append("\\begin{document}\n\\maketitle\n")

    # 若第一个标题被当成文档名，则从它的子章节开始渲染，并把层级上移一级。
    #
    # ★ 必须要求它**确有子章节**：扁平 `#` 标题的文档里，第一个标题是第一章
    # （如「摘要」）而不是书名，把它当书名会把后面所有章节都丢掉。
    roots: Sequence[Section] = doc.sections
    level_offset = 0
    if roots and roots[0].title == resolved_title and roots[0].children:
        roots = list(roots[0].children)
        level_offset = -1

    for section in roots:
        lines += _render_section(section, doc, level_offset)

    lines.append(f"\\bibliographystyle{{plain}}\n\\bibliography{{{bibliography}}}\n")
    lines.append("\\end{document}\n")
    return "\n".join(lines)
