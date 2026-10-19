"""回读校验的回归测试 —— **扁平标题会把整篇丢光**。

## 这个 bug 值得单独锁住

导出器有一条启发式：*"如果第一个标题被取作文档名，就从它的子章节开始渲染"*。
它假定第一个标题是个**容器**（`# 论文题目` + `## 各章`）。

但中文论文极常见的写法是**扁平标题**：

```
# 摘要
# 方法
# 结论
```

此时「第一个标题」是第一章，不是书名，它**没有子章节**。
旧实现直接取 `roots[0].children`（空列表），于是 **`\\section` 一个都没渲染**，
3 个章节全部丢失 —— 而导出函数自己还返回"成功"。

是**回读校验**（`verify_latex`）抓到的：原稿 3 章 vs 产物 0 章。
这正好证明了"回读校验"存在的价值 —— 生成时自检说"我按逻辑写了 N 个 section"，
只有回读才知道文件里到底有没有。
"""

from __future__ import annotations

import pytest

from copilot.delivery.export import export_paper
from copilot.delivery.export.latex import resolve_title
from copilot.quality.format import parse_markdown

FLAT = """# 摘要

本文研究长尾推荐问题。

# 方法

本文设计了两阶段策略。

# 结论

方法有效。
"""

NESTED = """# 基于图网络的推荐方法

## 摘要

本文研究长尾推荐问题。

## 方法

本文设计了两阶段策略。

## 结论

方法有效。
"""


def _sections(tex: str) -> int:
    return tex.count("\\section{")


def test_flat_headings_keep_every_chapter() -> None:
    """★ 回归：扁平标题下 3 个章节必须全部渲染出来。"""
    result = export_paper(parse_markdown(FLAT))
    assert _sections(result.tex) == 3
    for title in ("摘要", "方法", "结论"):
        assert f"\\section{{{title}}}" in result.tex


def test_flat_headings_pass_readback_check() -> None:
    """★ 回归：产物与原稿章节数一致，回读校验应当通过。"""
    result = export_paper(parse_markdown(FLAT))
    assert result.ok, result.latex_check.diffs


def test_nested_headings_still_work() -> None:
    """容器式写法（`# 题目` + `## 各章`）不能被这次修复弄坏。"""
    result = export_paper(parse_markdown(NESTED))
    assert result.ok, result.latex_check.diffs
    assert _sections(result.tex) == 3


def test_flat_document_does_not_use_chapter_as_title() -> None:
    """★ 第一章「摘要」不该被当作文档名（否则它会从正文里消失）。"""
    doc = parse_markdown(FLAT)
    assert resolve_title(doc) != "摘要"
    assert "待填" in resolve_title(doc)


def test_nested_document_does_use_container_as_title() -> None:
    doc = parse_markdown(NESTED)
    assert resolve_title(doc) == "基于图网络的推荐方法"


def test_explicit_title_wins() -> None:
    doc = parse_markdown(FLAT)
    assert resolve_title(doc, "我的题目") == "我的题目"


def test_single_section_document_is_exported() -> None:
    result = export_paper(parse_markdown("# 方法\n\n正文。\n"))
    assert _sections(result.tex) == 1
    assert result.ok


def test_empty_document_does_not_crash() -> None:
    result = export_paper(parse_markdown(""))
    assert _sections(result.tex) == 0
    assert result.ok


@pytest.mark.parametrize("draft", [FLAT, NESTED])
def test_bibliography_survives_both_layouts(draft: str) -> None:
    result = export_paper(parse_markdown(draft))
    assert "\\bibliography{" in result.tex
    assert "\\end{document}" in result.tex
