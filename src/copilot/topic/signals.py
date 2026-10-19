"""A1 选题分析：关键词共现 / 时间趋势 / 研究空白候选。

## 这一节点**完全确定性**

给定同一份语料，任何人重跑都得到同样的数字。这是本节点的价值所在 ——
选题讨论里最容易变成"我觉得这个方向有前景"，而统计出来的东西可复核。

## 三个子部分各自解决什么

**① 关键词共现** —— 哪些术语经常一起出现。
共现强的两个词往往构成一个**成熟子领域**；
共现弱但各自高频的两个词，**交叉处常常就是空白**。

**② 时间趋势** —— 按年份统计热度。
区分"已经做烂了"和"刚起来"。一个方向连续几年占比下降，选题前知道能省半年。

**③ 研究空白候选** —— 用「高频 × 低共现」找候选。
**这是唯一有价值的部分，也是唯一危险的部分**：
"低频组合"不等于"真空白"，可能只是没人觉得值得做。
所以全部标 ``inferred``，并且**必须带 support**（哪几篇文献支撑这个判断）。

## 为什么空白候选必须带 support

没有语料支撑的"空白"是猜测。把它写进开题报告，整篇论文的出发点就是错的。
所以 ``support`` 为空时**不输出这条候选** —— 宁可少报。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Mapping, Sequence

from copilot.evidence.text import content_tokens

from copilot.topic.models import Gap, KeywordStat, TopicSignals

__all__ = ["analyze_topic", "STOPWORDS", "MIN_TERM_LEN"]


#: 停用词：在学术标题/摘要里高频但没有区分度
STOPWORDS: frozenset[str] = frozenset(
    {
        # 学术套话
        "研究", "方法", "本文", "我们", "基于", "一种", "问题", "分析", "系统",
        "模型", "结果", "实验", "提出", "进行", "通过", "以及", "可以", "能够",
        "方面", "情况", "过程", "工作",
        # 虚词二字组（content_tokens 会产出二字组）
        # ★ 实测踩过：不滤这些的话，空白候选会报出「推荐」与「中的」这种组合，
        # 一眼就知道是噪声 —— 一旦报告里混进这种东西，整份报告就没人看了。
        "中的", "上的", "下的", "用的", "对于", "关于", "由于", "因为",
        "所以", "如果", "那么", "这些", "那些", "这种", "那种", "其中",
        "之间", "及其", "并且", "而是", "不是", "没有", "具有", "需要",
        # 英文虚词
        "the", "and", "for", "with", "using", "based", "a", "an", "of", "in", "on",
    }
)

#: 词长下限（中文 2 字、英文 3 字符）—— 太短的多是虚词
MIN_TERM_LEN = 2


#: 虚词字符 —— 含这些字的多字组几乎一定是跨词噪声。
#:
#: ★ 这是"没有分词器"的代价：``content_tokens`` 产出的是**字面二字组**，
#: 会把「检索增强」切成 检索/索增/增强 三个，中间那个 `索增` 是跨词噪声。
#: 用中文虚词表过滤能去掉大部分明显噪声，但去不干净（``索增`` 两个字都是实字）。
#:
#: **所以本节点的关键词必须被当作"候选"看，不能当作"领域术语表"。**
#: 真要准确，需要接分词器（jieba）或让模型归纳 —— 那属于模型节点。
_FUNCTION_CHARS: frozenset[str] = frozenset(
    "的了在和与是中上下之其也就都而及或对为以于把被让给从向很更最们个"
)


def _terms(text: str) -> set[str]:
    out: set[str] = set()
    for t in content_tokens(text):
        if len(t) < MIN_TERM_LEN or t.lower() in STOPWORDS:
            continue
        # 含虚词字符的多字组 → 跨词噪声
        if len(t) > 1 and any(ch in _FUNCTION_CHARS for ch in t):
            continue
        out.add(t)
    return out


def analyze_topic(
    texts: Sequence[str],
    *,
    top_k: int = 20,
    min_support: int = 2,
    years: Mapping[int, int] | None = None,
    max_gaps: int = 5,
) -> TopicSignals:
    """从语料里挖信号。

    ``texts``：每篇文献的标题 + 摘要（拼接后传入即可）。
    ``years``：``{语料下标: 年份}``，给了才算时间趋势。
    """
    if not texts:
        return TopicSignals(corpus_size=0, notes=["语料为空，未做任何统计"])

    doc_terms: list[set[str]] = [_terms(t) for t in texts]
    freq: Counter[str] = Counter()
    for s in doc_terms:
        freq.update(s)

    top = [t for t, _c in freq.most_common(top_k)]

    # ── ① 关键词统计 + 年份 ──
    by_term_years: dict[str, list[int]] = defaultdict(list)
    if years:
        for idx, termset in enumerate(doc_terms):
            y = years.get(idx)
            if y is None:
                continue
            for t in termset & set(top):
                by_term_years[t].append(y)

    keywords = [
        KeywordStat(
            term=t,
            count=freq[t],
            years=tuple(sorted(set(by_term_years.get(t, [])))),
        )
        for t in top
    ]

    # ── ② 共现 ──
    pair_counts: Counter[tuple[str, str]] = Counter()
    topset = set(top)
    for termset in doc_terms:
        present = sorted(termset & topset)
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                pair_counts[(present[i], present[j])] += 1

    cooccurrence = [(a, b, c) for (a, b), c in pair_counts.most_common(15)]

    # ── ③ 研究空白候选：高频 × 低共现 ──
    # 只在**出现过 min_support 次以上**的词之间找，避免噪声。
    frequent = [t for t in top if freq[t] >= min_support]
    # ★ 共现表按**字母序**存 key（上方 present 是 sorted 过的），
    # 而这里的 frequent 是按**词频**排的 —— 两者顺序不一致。
    # 实测踩过：直接 pair_lookup.get((a, b)) 会因为键顺序不同而恒返回 0，
    # 于是把"明明同篇出现过"的词对报成"从未同篇出现"（假空白）。
    # 查表必须统一成字母序。
    pair_lookup = {tuple(sorted((a, b))): c for (a, b), c in pair_counts.items()}

    gaps: list[Gap] = []
    for i in range(len(frequent)):
        for j in range(i + 1, len(frequent)):
            a, b = frequent[i], frequent[j]
            co = pair_lookup.get(tuple(sorted((a, b))), 0)
            if co > 0:
                continue
            # 支撑：分别包含 a 或 b 的文献（**必须有东西支撑**）
            support = [
                f"doc#{idx}"
                for idx, termset in enumerate(doc_terms)
                if a in termset or b in termset
            ]
            if not support:
                continue
            expected = freq[a] * freq[b] / max(1, len(doc_terms))
            gaps.append(
                Gap(
                    text=(
                        f"「{a}」与「{b}」各自高频（{freq[a]} / {freq[b]} 次），"
                        f"但从未同篇出现（期望约 {expected:.1f} 次）—— 交叉处可能是空白"
                    ),
                    support=tuple(support[:20]),
                    strength="inferred",
                )
            )
            if len(gaps) >= max_gaps:
                break
        if len(gaps) >= max_gaps:
            break

    # ── 趋势 ──
    trends: dict[str, dict[int, int]] = {}
    if years:
        for t in top[:10]:
            per_year: Counter[int] = Counter()
            for idx, termset in enumerate(doc_terms):
                y = years.get(idx)
                if y is not None and t in termset:
                    per_year[y] += 1
            if per_year:
                trends[t] = dict(sorted(per_year.items()))

    notes = [
        f"统计 {len(texts)} 篇语料，取前 {top_k} 个关键词",
        "所有空白候选均为 **inferred**（统计推断），不是原文结论",
    ]
    if len(texts) < 20:
        notes.append(
            f"⚠️ 语料只有 {len(texts)} 篇 —— 空白候选**噪声很大**，"
            "大概率是语料不足而非真有空白，必须人工读文献核实"
        )
    if not years:
        notes.append("没有年份数据，未做时间趋势")

    return TopicSignals(
        keywords=keywords,
        cooccurrence=cooccurrence,
        trends=trends,
        gaps=gaps,
        corpus_size=len(texts),
        notes=notes,
    )
