"""文本重叠度量 —— 用于**验证改写是否真的降低了重复**。

这里测的不是"函数返回了个数"，而是**度量本身能不能区分开我们关心的情况**：

- 改都没改 → 两个度量都应当很高
- 只换几个词 → 重叠仍高
- 真改写 → 两个度量都应当低下来
- 只把长句拆成短句 → **连续重叠降了，但 n-gram 包含率仍然高**
  （这正是需要两个度量的原因）
"""

from __future__ import annotations

from copilot.overlap import (
    best_match_region,
    longest_common_run,
    longest_common_substring,
    ngram_containment,
    overlap_ratio,
    overlap_report,
)

OLD = "本文提出了一种基于图神经网络的推荐方法，并在三个公开数据集上取得了显著效果。"


# ─────────────────────── 最长公共子串 ───────────────────────


def test_longest_common_substring_basic() -> None:
    assert longest_common_substring("abcdef", "xxcdefyy") == (2, 2, 4)


def test_longest_common_substring_no_overlap() -> None:
    assert longest_common_substring("abc", "xyz")[2] == 0


def test_longest_common_substring_empty() -> None:
    assert longest_common_substring("", "abc") == (0, 0, 0)
    assert longest_common_substring("abc", "") == (0, 0, 0)


def test_longest_common_substring_prefers_contiguous() -> None:
    """要的是**连续**重叠长度，不是"匹配块"总长。

    这是刻意不用 ``difflib.SequenceMatcher`` 的原因：
    它会把断裂的匹配算作一个长块，从而**高估"还剩多少没改"**。
    """
    assert longest_common_substring("ABCDEF", "ABCXXXDEF")[2] == 3


def test_longest_common_run_normalizes_both_sides() -> None:
    """两侧都必须规范化，否则全角/半角差异会让重叠在每个标点处断裂。"""
    a = "你好，世界。"
    b = "你好,世界."
    assert longest_common_run(a, b) == len("你好,世界.")


# ─────────────────────── 两个度量 ───────────────────────


def test_identical_text_has_full_overlap() -> None:
    report = overlap_report(OLD, OLD)
    assert report["overlap_ratio"] == 1.0
    assert report["ngram_containment"] == 1.0


def test_real_rewrite_lowers_both_metrics() -> None:
    """★ 真改写：两个度量都应当低下来。"""
    rewritten = (
        "我们设计了一个双塔结构，用对比学习把图文映射到同一个表示空间，"
        "在若干标准测试集上验证了它的有效性。"
    )
    report = overlap_report(rewritten, OLD)
    assert report["ngram_containment"] < 0.2, report
    assert report["overlap_ratio"] < 0.4, report


def test_word_swap_only_still_has_high_overlap() -> None:
    """只换几个词不算改写 —— 度量必须能识破。"""
    swapped = "本文提出了一种基于图神经网络的推荐方法，并在两个公开数据集上取得了显著效果。"
    report = overlap_report(swapped, OLD)
    assert report["ngram_containment"] > 0.5, report
    assert report["overlap_ratio"] > 0.5, report


def test_splitting_into_short_sentences_shows_why_two_metrics_needed() -> None:
    """★ 两个度量的存在理由：拆句能压低"连续重叠"，但压不下 n-gram 包含率。

    把长句拆成短句后，最长连续重叠变短了；可词组仍然来自原句，
    所以 n-gram 包含率依然明显高于**真改写**。
    只看一个指标就会误判为"改写成功"。
    """
    split = (
        "本文提出了一种方法。该方法基于图神经网络。"
        "它用于推荐任务。并在三个公开数据集上取得了显著效果。"
    )
    real_rewrite = (
        "我们设计了一个双塔结构，用对比学习把图文映射到同一个表示空间，"
        "在若干标准测试集上验证了它的有效性。"
    )

    split_containment = ngram_containment(split, OLD)
    rewrite_containment = ngram_containment(real_rewrite, OLD)

    assert longest_common_run(split, OLD) < len(OLD) / 2, "拆句应压短连续重叠"
    assert split_containment > rewrite_containment * 2, (
        f"拆句的 n-gram 包含率（{split_containment:.2f}）应显著高于"
        f"真改写（{rewrite_containment:.2f}）—— 这正是需要第二个度量的理由"
    )


def test_ngram_containment_handles_text_shorter_than_n() -> None:
    """★ 新文本短于 n 时的特判。

    不特判的话切不出 n-gram，会一律返回 0.0 —— 那等于说"完全没抄"，
    而一个 4 字的短语完全可能整句来自原文。
    """
    old = "本文提出了一种基于图神经网络的推荐方法，并在三个公开数据集上取得了显著效果。"
    assert ngram_containment("推荐方法", old) == 1.0, "整串来自原文却报了 0"
    assert ngram_containment("量子计算", old) == 0.0


def test_empty_inputs_return_zero() -> None:
    assert ngram_containment("", OLD) == 0.0
    assert overlap_ratio("", OLD) == 0.0


# ─────────────────────── 最佳匹配区域 ───────────────────────


def test_best_match_region_locates_needle() -> None:
    haystack = "开头一些无关的内容。本文提出了一种基于图神经网络的推荐方法。后面还有一些内容。"
    needle = "基于图神经网络的推荐方法"
    found = best_match_region(needle, haystack, min_run=6)
    assert found is not None
    start, end, run = found
    assert haystack[start:end] == needle
    assert run == len(needle)


def test_best_match_region_returns_none_when_too_weak() -> None:
    """★ 匹配太弱时返回 None —— **宁可报"定位失败"，也不要硬指一个位置**。

    外部报告里的片段可能来自**旧版本**的文档（用户之后又改过）。
    硬指一个位置比说"没定位到"更糟：用户会去改错误的地方。
    """
    assert best_match_region("完全不相干的一句话", "另一段毫不相关的内容", min_run=8) is None


def test_best_match_region_tolerates_formatting_differences() -> None:
    """全角/半角、空白的差异不应导致定位失败。"""
    haystack = "前言。本文提出了一种基于图神经网络的推荐方法。"
    needle = "本文提出了一种基于图神经网络的推荐方法"   # 规范化后一致
    assert best_match_region(needle, haystack, min_run=6) is not None
