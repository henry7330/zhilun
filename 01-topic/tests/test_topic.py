"""A 阶段（选题与开题）测试。

核心验收：
1. A1 的信号是**真实统计**（同一语料重跑同结果）；
2. **空白候选必须带 support**，没有支撑的不输出；
3. **空白候选全部标 inferred**，不得冒充"原文说的"；
4. A2 没模型时给**该问的问题**，不编造 level；
5. 阶段可独立调用，无任何前置依赖。
"""

from __future__ import annotations

import pytest

from copilot.topic import analyze_topic, assess_feasibility, run_topic
from copilot.topic.signals import STOPWORDS

TEXTS = [
    "检索增强生成在推荐系统中的应用研究",
    "图神经网络用于推荐系统的方法",
    "基于检索增强的问答系统研究",
    "长尾用户的行为稀疏问题与元学习",
    "推荐系统中的冷启动问题研究",
    "图神经网络在长尾推荐中的应用",
]
YEARS = {0: 2019, 1: 2020, 2: 2020, 3: 2021, 4: 2022, 5: 2022}


# ─────────────────── A1 统计 ───────────────────


def test_keywords_are_statistical_and_reproducible() -> None:
    """同一份语料重跑，必须得到完全一样的词表 —— 这是"可复核"的前提。"""
    a = analyze_topic(TEXTS)
    b = analyze_topic(TEXTS)
    assert [k.term for k in a.keywords] == [k.term for k in b.keywords]
    assert [k.count for k in a.keywords] == [k.count for k in b.keywords]


def test_empty_corpus_is_reported_not_crashed() -> None:
    signals = analyze_topic([])
    assert signals.corpus_size == 0
    assert any("语料为空" in n for n in signals.notes)


def test_stopwords_are_excluded() -> None:
    terms = {k.term for k in analyze_topic(TEXTS).keywords}
    assert not (terms & {"研究", "方法", "我们"})


def test_function_char_bigrams_are_filtered() -> None:
    """★ 含虚词字符的多字组是跨词噪声，必须滤掉。

    实测踩过：不过滤时会报出「推荐」与「中的」这种组合 ——
    报告里混进这种东西，整份报告就没人看了。
    """
    terms = {k.term for k in analyze_topic(TEXTS).keywords}
    for t in terms:
        assert not (set(t) & set("的了在和与是中上下")), t


def test_cooccurrence_detects_real_pairs() -> None:
    """同篇出现过的词对必须被记到共现里（不能因为键顺序不同而丢）。"""
    signals = analyze_topic(TEXTS)
    pairs = {(a, b) for a, b, _c in signals.cooccurrence} | {
        (b, a) for a, b, _c in signals.cooccurrence
    }
    assert ("应用", "推荐") in pairs or ("推荐", "应用") in pairs


# ─────────────────── 空白候选 ───────────────────


def test_gaps_always_have_support() -> None:
    """★ 没有语料支撑的"空白"是猜测，不该出现。"""
    for gap in analyze_topic(TEXTS).gaps:
        assert gap.support, gap.text


def test_gaps_are_marked_inferred() -> None:
    """★ 统计推断的空白不得冒充"原文说的"。"""
    for gap in analyze_topic(TEXTS).gaps:
        assert gap.strength == "inferred"


def test_gap_does_not_report_pairs_that_co_occur() -> None:
    """★ 实测踩过的 bug：共现表按字母序存、查表按词频序 → 恒查不到 → 假空白。

    「推荐」与「应用」在同篇出现过，因此**不该**成为空白候选。
    """
    for gap in analyze_topic(TEXTS).gaps:
        assert not ("推荐" in gap.text and "应用" in gap.text), gap.text


def test_small_corpus_gets_warning() -> None:
    signals = analyze_topic(TEXTS)
    assert any("噪声很大" in n for n in signals.notes)


def test_trends_need_years() -> None:
    assert analyze_topic(TEXTS).trends == {}
    signals = analyze_topic(TEXTS, years=YEARS)
    assert signals.trends
    assert all(isinstance(y, int) for per in signals.trends.values() for y in per)


# ─────────────────── A2 / A3 ───────────────────


def test_feasibility_skeleton_gives_questions_not_verdict() -> None:
    """★ 没模型时给"该问的问题"，不编造 level。"""
    feas = assess_feasibility("某个题目")
    assert feas.skeleton is True
    assert feas.level == "unknown"
    assert feas.risks, "骨架必须给出该自己回答的问题"
    assert any("无法评估" in b for b in feas.blockers)


def test_feasibility_uses_judge_when_given() -> None:
    def judge(topic: str, conditions: dict) -> dict:
        return {
            "level": "low",
            "blockers": ["需要 1 万条标注数据，你没有"],
            "risks": ["数据拿不到"],
            "suggestions": ["换公开数据集"],
        }

    feas = assess_feasibility("某题目", judge=judge)
    assert feas.skeleton is False
    assert feas.level == "low"
    assert feas.blockers == ["需要 1 万条标注数据，你没有"]


# ─────────────────── 阶段入口 ───────────────────


def test_run_topic_works_without_prerequisites() -> None:
    """A 阶段不依赖任何前置阶段 —— 只给题目也能跑。"""
    report = run_topic(title="某个题目")
    assert report.stage == "A"
    assert {r.node for r in report.results} == {"A1", "A2", "A3"}


def test_run_topic_with_corpus_produces_signals() -> None:
    report = run_topic(title="推荐方法", texts=TEXTS, years=YEARS)
    a1 = report.by_node()["A1"]
    assert a1.ok
    assert a1.stats["corpus"] == len(TEXTS)
    assert a1.stats["keywords"] > 0


def test_proposal_has_all_sections() -> None:
    report = run_topic(title="某个题目")
    proposal = report.to_dict()  # 只要能序列化
    assert proposal["stage"] == "A"
    a3 = report.by_node()["A3"]
    assert a3.stats["sections"] == 6
    assert a3.stats["skeleton"] is True


def test_report_markdown_renders() -> None:
    md = run_topic(title="某个题目", texts=TEXTS).to_markdown()
    assert "阶段 A 报告" in md
    assert "各节点执行情况" in md
