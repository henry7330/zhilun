"""F1 · 文档导出测试。

核心验收：
1. **LaTeX 转义正确** —— 标题含 `&` `%` `_` 时生成的 `.bib` 可被 LaTeX 接受；
2. **导出无丢失** —— 回读校验能发现人为删掉一章 / 一个图；
3. 引用键唯一（重复键会让 `\\cite` 指错文献）；
4. 两种格式（BibTeX / GB/T 7714）都能生成。
"""

from __future__ import annotations

import pytest

from copilot.delivery.export import (
    BibEntry,
    escape_latex,
    export_paper,
    from_reference,
    render_bibtex_file,
    render_gbt7714_list,
    to_bibtex,
    to_gbt7714,
    verify_bibtex,
    verify_latex,
)
from copilot.quality.format import parse_markdown

# ─────────────────────────── 夹具 ───────────────────────────

THESIS = """# 基于图神经网络的推荐方法研究

## 摘要

本文研究了一种基于图神经网络的推荐方法。

## 绪论

检索增强在近年受到关注，整体框架 见图 1。

**图 1** 研究框架

## 方法

我们采用两阶段流程。

**表 1** 参数设置

## 结论

本文的方法取得了预期效果。

## 参考文献

[1] 张三, 李四. 一种基于图神经网络的方法[J]. 计算机学报, 2020, 43(2): 1-10.
[2] 王五. 深度学习[M]. 北京: 科学出版社, 2021.
[3] 赵六. 检索增强综述[J]. 软件学报, 2019, 30(5): 100-120.
"""


@pytest.fixture
def doc():
    return parse_markdown(THESIS)


# ─────────────────── ★ LaTeX 转义 ───────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("A & B", r"A \& B"),
        ("50% 提升", r"50\% 提升"),
        ("a_b", r"a\_b"),
        ("$x$", r"\$x\$"),
        ("#1", r"\#1"),
        ("{brace}", r"\{brace\}"),
    ],
)
def test_escape_latex_special_chars(raw: str, expected: str) -> None:
    assert escape_latex(raw) == expected


def test_escape_latex_handles_backslash_first() -> None:
    """反斜杠必须**最先**处理，否则 ``\\&`` 会被二次转义。

    这是顺序敏感的一个真实陷阱：如果先替换 ``&`` → ``\\&``，
    再替换 ``\\`` → ``\\textbackslash{}``，
    就会得到 ``\\textbackslash{}&`` —— 编译直接失败。
    """
    assert escape_latex("\\") == r"\textbackslash{}"
    # 已经是转义形态的文本再转义一次，不应产生 \\& 这种畸形
    once = escape_latex("A & B")
    assert "\\textbackslash{}&" not in once


def test_bibtex_escapes_title(doc) -> None:
    """★ 标题含 `&` 时生成的 BibTeX 必须已转义。"""
    entry = BibEntry(
        key="test2020",
        entry_type="article",
        authors=["张三"],
        title="R&D of A&B",
        venue="Journal of X & Y",
        year=2020,
    )
    out = to_bibtex(entry)
    assert r"\&" in out
    assert " A&B " not in out and "R&D" not in out


# ─────────────────── 文献模型 ───────────────────


def test_from_reference_maps_fields(doc) -> None:
    entry = from_reference(doc.references[0])
    assert entry.entry_type == "article"
    assert entry.authors == ["张三", "李四"]
    assert entry.year == 2020
    assert entry.cn_type == "J"
    assert entry.key


def test_from_reference_maps_book(doc) -> None:
    entry = from_reference(doc.references[1])
    assert entry.entry_type == "book"
    assert entry.cn_type == "M"


def test_reference_keys_are_unique(doc) -> None:
    """★ 重复的键会让 ``\\cite`` 指向错误文献。"""
    used: set[str] = set()
    entries = [from_reference(r, used_keys=used) for r in doc.references]
    keys = [e.key for e in entries]
    assert len(keys) == len(set(keys))


def test_key_collision_gets_suffix(doc) -> None:
    """同姓同年同题名首词时，第二个键要自动加后缀。"""
    used: set[str] = set()
    a = from_reference(doc.references[0], used_keys=used)
    # 构造一条必然同键的记录
    dup = from_reference(doc.references[0], used_keys=used)
    assert a.key != dup.key
    assert dup.key.endswith("-2")


# ─────────────────── 两种输出格式 ───────────────────


def test_gbt7714_article_format() -> None:
    entry = BibEntry(
        key="k", entry_type="article", authors=["张三", "李四"],
        title="一种方法", venue="计算机学报", year=2020,
        volume="43", number="2", pages="1-10", cn_type="J",
    )
    out = to_gbt7714(entry)
    assert "张三, 李四" in out
    assert "[J]" in out
    assert "2020" in out
    assert "43(2)" in out
    assert "1-10" in out


def test_gbt7714_uses_etal_for_many_authors() -> None:
    entry = BibEntry(
        key="k", authors=["甲", "乙", "丙", "丁"],
        title="题名", venue="刊名", year=2020, cn_type="J",
    )
    out = to_gbt7714(entry)
    assert "等" in out
    assert "丁" not in out


def test_gbt7714_does_not_invent_missing_fields() -> None:
    """缺卷期页时不补造 —— 宁可简略，也不编造。"""
    entry = BibEntry(key="k", authors=["张三"], title="题名", venue="刊名", year=2020)
    out = to_gbt7714(entry)
    assert "None" not in out
    assert "()" not in out


def test_render_bibtex_file_has_all_entries(doc) -> None:
    used: set[str] = set()
    entries = [from_reference(r, used_keys=used) for r in doc.references]
    text = render_bibtex_file(entries)
    assert text.count("@") == len(entries)


def test_render_gbt7714_list_is_numbered(doc) -> None:
    entries = [from_reference(r) for r in doc.references]
    text = render_gbt7714_list(entries)
    assert text.startswith("[1]")
    assert f"[{len(entries)}]" in text


# ─────────────── ★ 回读校验：导出无丢失 ───────────────


def test_export_passes_verification(doc) -> None:
    result = export_paper(doc)
    assert result.ok, result.summary()
    assert "\\section" in result.tex
    assert len(result.entries) == 3


def test_verification_catches_missing_section(doc) -> None:
    """★ 人为删掉一章 → 校验必须发现。

    这正是"回读反查"的价值：肉眼看 .tex 根本不会去数 section 数量。
    """
    result = export_paper(doc)
    broken = result.tex.replace("\\section{方法}", "")
    check = verify_latex(broken, doc)
    assert not check.ok
    assert any(d.item == "章节数" for d in check.diffs)


def test_verification_catches_missing_figure(doc) -> None:
    result = export_paper(doc)
    broken = result.tex.replace("\\begin{figure}", "").replace("\\end{figure}", "")
    check = verify_latex(broken, doc)
    assert not check.ok
    assert any(d.item == "图数量" for d in check.diffs)


def test_verification_catches_missing_title(doc) -> None:
    result = export_paper(doc)
    broken = result.tex.replace("\\section{方法}", "\\section{另外一个标题}")
    check = verify_latex(broken, doc)
    assert not check.ok
    assert any(d.item == "章节标题" for d in check.diffs)


def test_bib_verification_catches_missing_entry(doc) -> None:
    result = export_paper(doc)
    blocks = result.bibtex.split("\n\n")
    broken = "\n\n".join(blocks[:-1])
    check = verify_bibtex(broken, doc)
    assert not check.ok
    assert any(d.item == "参考文献条数" for d in check.diffs)


def test_bib_verification_catches_duplicate_keys(doc) -> None:
    duplicated = "@article{k1,\n  title = {A},\n}\n\n@article{k1,\n  title = {B},\n}\n"
    check = verify_bibtex(duplicated, doc)
    assert not check.ok
    assert any(d.item == "引用键唯一性" for d in check.diffs)


# ─────────────────── 结构细节 ───────────────────


def test_latex_uses_document_title_from_first_heading(doc) -> None:
    result = export_paper(doc)
    assert "基于图神经网络的推荐方法研究" in result.tex
    # 文档标题不该再作为 section 出现
    assert "\\section{基于图神经网络的推荐方法研究}" not in result.tex


def test_latex_strips_manual_numbering(doc) -> None:
    """标题里的「1.1」要去掉，交给 LaTeX 自动编号，否则会双重编号。"""
    md = "# 论文\n\n## 1.1 研究背景\n\n内容在此。\n"
    d = parse_markdown(md)
    result = export_paper(d)
    assert "\\section{研究背景}" in result.tex
    assert "1.1 研究背景" not in result.tex


def test_latex_includes_bibliography_hook(doc) -> None:
    result = export_paper(doc)
    assert "\\bibliography{refs}" in result.tex


def test_export_result_summary_is_readable(doc) -> None:
    summary = export_paper(doc).summary()
    assert "LaTeX 骨架" in summary
    assert "校验" in summary
    assert "通过" in summary
