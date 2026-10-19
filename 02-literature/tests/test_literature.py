"""B 阶段（文献）测试。

核心验收：
1. B3 去重**按 DOI / 规范化标题**，不按字符串相等；
2. 缺文献库时**跳过核实并明说**，不假装"引用没问题"；
3. B2 的篇数是**去重的**（多维分组会重叠，不能相加）；
4. 三个节点都能独立跳过，且跳过要留下说明。
"""

from __future__ import annotations

from copilot.bib import BibEntry
from copilot.literature import manage_references, plan_survey, run_literature
from copilot.literature.search import dedupe_key, normalize_title


def _e(key: str, title: str, *, doi: str | None = None, venue: str = "计算机学报",
       year: int = 2020) -> BibEntry:
    return BibEntry(key=key, authors=["张三"], title=title, venue=venue,
                    year=year, doi=doi, cn_type="J")


# ─────────────────── 去重 ───────────────────


def test_normalize_title_ignores_punctuation_and_case() -> None:
    assert normalize_title("Retrieval Augmented, Generation!") == normalize_title(
        "retrieval augmented generation"
    )


def test_dedupe_key_prefers_doi() -> None:
    a = _e("a", "完全不同的标题啊", doi="10.1/ABC")
    b = _e("b", "另一个标题", doi="10.1/abc")
    assert dedupe_key(a) == dedupe_key(b), "DOI 大小写不同应视为同一篇"


def test_dedupe_falls_back_to_title() -> None:
    a = _e("a", "检索增强 方法研究")
    b = _e("b", "检索增强方法研究")
    assert dedupe_key(a) == dedupe_key(b)


def test_manage_references_removes_duplicates() -> None:
    entries = [
        _e("a", "检索增强方法研究", doi="10.1/x"),
        _e("b", "检索增强 方法研究", doi="10.1/x"),   # 同 DOI
        _e("c", "图神经网络综述", doi="10.1/y"),
    ]
    report = manage_references(entries)
    assert report.total_in == 3
    assert report.total_out == 2
    assert report.removed == 1


def test_manage_references_generates_both_formats() -> None:
    report = manage_references([_e("a", "一种方法")])
    assert "@article" in report.bibtex
    assert "[J]" in report.gbt7714


def test_missing_library_is_reported_not_silent() -> None:
    """★ 静默跳过会让人以为「引用都查过了、没问题」—— 那是最危险的情况。"""
    report = manage_references([_e("a", "一种方法")])
    assert any("跳过引用核实" in n for n in report.notes)
    assert any("不等于" in n for n in report.notes)


# ─────────────────── 综述分组 ───────────────────


def test_plan_survey_counts_unique_papers() -> None:
    """★ 多维分组会重叠，篇数不能把各组相加。

    实测踩过：4 篇文献被算成 8 篇。
    """
    entries = [
        _e("a", "检索增强", venue="计算机学报", year=2020),
        _e("b", "图神经网络", venue="软件学报", year=2022),
        _e("c", "长尾推荐", venue="软件学报", year=2016),
        _e("d", "冷启动", venue="计算机学报", year=2019),
    ]
    plan = plan_survey(entries)
    assert plan.total == 4, "篇数必须是去重后的 4，而不是各组之和"


def test_plan_survey_marks_conflicts() -> None:
    plan = plan_survey([_e("a", "x"), _e("b", "y", venue="软件学报")])
    assert plan.conflicts, "综述骨架必须提示需要并列呈现的分歧"


def test_plan_survey_empty_corpus() -> None:
    plan = plan_survey([])
    assert plan.groups == []
    assert any("没有文献" in n for n in plan.notes)


# ─────────────────── 阶段入口 ───────────────────


def test_run_literature_with_only_entries() -> None:
    report = run_literature(entries=[_e("a", "一种方法")], label="refs")
    assert report.stage == "B"
    assert {r.node for r in report.results} == {"B1", "B3", "B2"}
    assert report.by_node()["B1"].notes, "没语料时 B1 必须说明跳过了"


def test_run_literature_with_nothing_still_explains() -> None:
    report = run_literature()
    assert report.stage == "B"
    for node in ("B1", "B3", "B2"):
        assert report.by_node()[node].notes, node


def test_search_without_corpus_skips_with_reason() -> None:
    report = run_literature(keywords=["检索"], corpus=())
    assert any("跳过检索" in n for n in report.by_node()["B1"].notes)
