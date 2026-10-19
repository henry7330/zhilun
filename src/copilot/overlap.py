"""文本重叠度量 —— 用于**验证改写是否真的降低了重复**。

为什么这个工具必须在 00-core
-----------------------------
它服务两件事，分属不同模块：

1. **E3 报告解析**：外部报告里的片段要定位回用户文档，
   需要"这段文字和文档里哪一段最像"；
2. **E4 降重**：改写之后要回答一个硬问题 ——
   **「改写到底有没有降低与原句的重叠？」**

第 2 点是本模块存在的核心理由：**降重不能靠"感觉变了个说法"**。
必须有一个可测量的判据，否则无法判断改写是否有效，
也无法在报告里给出"重叠从 82% 降到 19%"这样的依据。

两个互补的度量
--------------
======================  ============================================
度量                     回答的问题
======================  ============================================
**最长连续重叠**          改写后还剩多长**一字不差**的连续片段？
**n-gram 包含率**        新句子里有多大比例的词组来自原句？
======================  ============================================

为什么要两个：
- 只连续重叠 → 把句子打散成短句就能骗过它（连续段变短，但词组还在抄）；
- 只 n-gram → 无法反映"整段照搬"的严重程度。

**两个都低**才说明真的改写了。这也是判定"改写是否合格"的客观标准。
"""

from __future__ import annotations

from typing import Sequence

from copilot.textnorm import normalize

__all__ = [
    "longest_common_substring",
    "longest_common_run",
    "overlap_ratio",
    "ngram_containment",
    "overlap_report",
    "DEFAULT_NGRAM",
]

#: n-gram 长度。与「通用度」判定用的是同一个量级，见各模块文档。
DEFAULT_NGRAM = 5


def longest_common_substring(a: str, b: str) -> tuple[int, int, int]:
    """返回 ``(a_start, b_start, length)`` —— 两串的最长公共子串。

    用**滚动数组**做动态规划：只保留上一行，空间 O(len(b))。
    字符级比较（中文按字，西文按字符），因为查重关心的是逐字重合。

    为什么不用 difflib：``SequenceMatcher`` 求的是最长**匹配块**，
    允许中间断裂，用于"整体相似度"合适；但我们要的是
    「**连续重叠了多少个字**」，断裂的匹配会让"还剩多少没改"被高估。
    """
    if not a or not b:
        return (0, 0, 0)

    prev = [0] * (len(b) + 1)
    best_len = 0
    best_a = 0
    best_b = 0
    ai_chars = a

    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ch = ai_chars[i - 1]
        for j in range(1, len(b) + 1):
            if ch == b[j - 1]:
                run = prev[j - 1] + 1
                cur[j] = run
                if run > best_len:
                    best_len = run
                    best_a = i - run
                    best_b = j - run
        prev = cur

    return (best_a, best_b, best_len)


def longest_common_run(a: str, b: str) -> int:
    """两串最长连续重叠的**长度**。"""
    return longest_common_substring(normalize(a), normalize(b))[2]


def overlap_ratio(a: str, b: str) -> float:
    """最长连续重叠 / 较短文本的长度，``[0, 1]``。

    除以**较短**的那个：改写后的句子通常长度会变，
    用较短者作分母，表达的才是"短的那句有多少比例被原句覆盖"。
    """
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    length = longest_common_substring(na, nb)[2]
    return min(1.0, length / min(len(na), len(nb)))


def _ngrams(text: str, n: int) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def ngram_containment(new: str, old: str, *, n: int = DEFAULT_NGRAM) -> float:
    """新文本的 n-gram 里，有多大比例能在旧文本中找到。

    这是**包含率**而不是 Jaccard：我们关心的是
    「**新句子里还有多少是老句子里的词组**」，
    老句子独有的部分不该拉低这个比例。

    短文本特判：新文本短于 ``n`` 时切不出 n-gram，
    直接退化成**子串包含判断**（在 → 1.0，不在 → 0.0）。
    不特判的话短句会一律返回 0.0 —— 那是在说"完全没抄"，
    而实际上一个 4 字的短语可能整句都来自原文。
    """
    normalized_new, normalized_old = normalize(new), normalize(old)
    if not normalized_new or not normalized_old:
        return 0.0
    if len(normalized_new) < n:
        return 1.0 if normalized_new in normalized_old else 0.0

    n_new = _ngrams(normalized_new, n)
    n_old = _ngrams(normalized_old, n)
    return len(n_new & n_old) / len(n_new)


def overlap_report(
    new: str, old: str, *, n: int = DEFAULT_NGRAM
) -> dict[str, float]:
    """一次给出两个度量，便于报告与测试直接使用。"""
    return {
        "longest_run": float(longest_common_run(new, old)),
        "overlap_ratio": overlap_ratio(new, old),
        "ngram_containment": ngram_containment(new, old, n=n),
    }


def best_match_region(
    needle: str, haystack: str, *, min_run: int = 8
) -> tuple[int, int, int] | None:
    """在 ``haystack`` 中找与 ``needle`` 重叠最长的区域。

    返回 ``(haystack_start, haystack_end, 重叠长度)``；重叠不足 ``min_run``
    时返回 ``None``（**宁可报"定位失败"，也不要硬指一个位置**）。

    用途：外部报告里的片段往往来自**旧版本**的文档（用户之后又改过），
    因此精确匹配可能失败。这里退回"找最像的一段"，
    并在返回 ``None`` 时让调用方如实报告"没能定位"，
    而不是静默丢弃或胡乱指一个位置。
    """
    nn, nh = normalize(needle), normalize(haystack)
    # 注意参数顺序：a=needle, b=haystack →
    # 返回的第二个分量才是 haystack 里的起点
    _needle_start, hay_start, run = longest_common_substring(nn, nh)
    if run < min_run:
        return None
    return (hay_start, hay_start + run, run)
