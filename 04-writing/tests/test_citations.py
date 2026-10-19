"""D4 文献引用测试 —— **引用必须来自 B 阶段的文献库**。

核心验收：
1. 库里有的引用 → 通过；
2. 库里没有的引用 → **错误**（这正是最容易出事的地方）；
3. 正文 `[n]` 超出参考文献条数 → 错误；
4. 没给文献库时**如实说"无法核对"**，不能默认通过。
"""

from __future__ import annotations

from copilot.bib import BibEntry
from copilot.quality.format import parse_markdown
from copilot.writing import audit_citations, run_writing

DRAFT_OK = """# 方法

本文采用两阶段检索策略[1]。

# 参考文献

[1] 张三. 一种方法[J]. 计算机学报, 2020.
"""

DRAFT_FOREIGN = """# 方法

本文采用两阶段检索策略[1]。

# 参考文献

[1] 李四. 另一篇来路不明的文章[J]. 某期刊, 2019.
"""

LIBRARY = [
    BibEntry(key="zhang2020", authors=["张三"], title="一种方法",
             venue="计算机学报", year=2020, doi="10.1/x", cn_type="J"),
]


def _doc(text: str):
    return parse_markdown(text)


# ─────────────────── 判定 ───────────────────


def test_reference_in_library_passes() -> None:
    audit = audit_citations(_doc(DRAFT_OK), LIBRARY)
    assert audit.references == 1
    assert audit.matched == 1
    assert audit.ok


def test_reference_outside_library_is_an_error() -> None:
    """★ 来路不明的引用必须报错 —— 这正是 D4 存在的理由。"""
    audit = audit_citations(_doc(DRAFT_FOREIGN), LIBRARY)
    assert not audit.ok
    assert audit.unmatched
    assert "不在文献库" in audit.unmatched[0].detail


def test_missing_library_is_reported_not_passed() -> None:
    """★ 没给文献库 ≠ 引用都没问题。"""
    audit = audit_citations(_doc(DRAFT_OK), ())
    assert any("无法核对引用来源" in i.detail for i in audit.issues)
    # 是 warning 而不是 error：不是"引用错了"，是"没法核对"
    assert all(i.severity != "error" for i in audit.issues)


def test_orphan_mark_is_an_error() -> None:
    """正文引用了 [5]，但参考文献只有 1 条。"""
    draft = DRAFT_OK.replace("策略[1]。", "策略[1]，如文献[5]所述。")
    audit = audit_citations(_doc(draft), LIBRARY)
    assert any(i.kind == "orphan_mark" for i in audit.issues)
    assert not audit.ok


def test_unused_library_entry_is_only_info() -> None:
    lib = LIBRARY + [
        BibEntry(key="wang2021", authors=["王五"], title="没人引用的综述",
                 venue="软件学报", year=2021, cn_type="J")
    ]
    audit = audit_citations(_doc(DRAFT_OK), lib)
    unused = [i for i in audit.issues if i.kind == "unused_entry"]
    assert unused
    assert unused[0].severity == "info"
    assert audit.ok, "没被引用不算错"


def test_matching_also_works_by_doi() -> None:
    """库里条目和正文标题写法不同，但 DOI 相同 —— 也要能匹配上。"""
    lib = [BibEntry(key="zhang2020", authors=["张三"], title="一种方法",
                    venue="计算机学报", year=2020, doi="10.1234/xyz", cn_type="J")]
    draft = """# 方法

见文献[1]。

# 参考文献

[1] 张三. 一种稍微不同的标题写法[J]. 计算机学报, 2020. doi:10.1234/xyz
"""
    audit = audit_citations(_doc(draft), lib)
    assert audit.matched == 1


def test_audit_serializes() -> None:
    import json

    payload = json.loads(json.dumps(audit_citations(_doc(DRAFT_OK), LIBRARY).to_dict(),
                                    ensure_ascii=False))
    assert payload["ok"] is True


# ─────────────────── 接入阶段 ───────────────────


def test_stage_runs_d4_with_library() -> None:
    report = run_writing(title="t", draft=DRAFT_OK, entries=LIBRARY)
    d4 = report.by_node()["D4"]
    assert d4.ok
    assert d4.stats["library"] == 1
    assert d4.stats["matched"] == 1


def test_stage_flags_foreign_reference() -> None:
    report = run_writing(title="t", draft=DRAFT_FOREIGN, entries=LIBRARY)
    d4 = report.by_node()["D4"]
    assert any("不在文献库" in f.message for f in d4.findings)
    # 节点本身执行成功（没崩），但报告因 ERROR 判为不可通过 —— 这两件事必须分开
    assert d4.ok
    assert not report.can_submit


def test_stage_d4_skips_without_draft() -> None:
    report = run_writing(title="t")
    d4 = report.by_node()["D4"]
    assert any("跳过" in n for n in d4.notes)


def test_d4_is_last_node() -> None:
    order: list[str] = []
    run_writing(title="t", draft=DRAFT_OK, entries=LIBRARY,
                on_node=lambda c, _r: order.append(c))
    assert order[-1] == "D4"
