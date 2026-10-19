"""E5 · 风格特征提取 —— 把"AI 味"变成可计算的数字。

为什么必须先有"可测量的特征"
----------------------------
**不可测量的东西无法改进，也无法验证。**

如果系统只说"这段读起来像 AI"，改写就没有目标，改完也无法判断有没有改善。
这与 E4 需要 ``overlap`` 度量是同一条理由：
**每一步改进都要有可测量的判据。**

★ 先说清楚本模块**不做什么**
----------------------------
「这段文字是不是 AI 生成的」是一个**不可靠的推断** ——
现有检测器的准确率与误报率都达不到可以据处分的程度。
本模块**不对此做任何判断**。

它处理的是：**外部报告说了什么**（由外部工具负责），
以及**可测量的风格特征**。报告里只出现特征的**数值与变化**，
不出现"这段是/不是 AI 写的"这类断言。

六个特征，各自对应一种常见的"机器味"
------------------------------------
============================  ================================
特征                          对应的观感
============================  ================================
**句长变异系数**               句式过于均匀
**套话连接词密度**             "此外/然而/综上所述"偏多
**词汇丰富度（TTR）**          用词重复或过于"标准"
**具体性**                     缺少数字与专名，泛泛而谈
**排比/对仗密度**              结构过于工整
**段落长度分布**               各段长短过于接近
============================  ================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from copilot.textseg import iter_segments

__all__ = [
    "StyleProfile",
    "profile",
    "TRANSITIONS",
    "PARALLEL_PATTERNS",
]

#: 常见套话连接词。**刻意只收"可替换为具体逻辑"的那一类** ——
#: 并不是所有连接词都是毛病（"但是"表转折就常有必要）。
TRANSITIONS: tuple[str, ...] = (
    "此外",
    "然而",
    "因此",
    "综上所述",
    "总的来说",
    "总之",
    "值得注意的是",
    "与此同时",
    "另一方面",
    "首先",
    "其次",
    "最后",
    "不难看出",
    "显而易见",
    "众所周知",
    "需要指出的是",
    "由此可见",
)

#: 排比/对仗的典型框架
PARALLEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"一方面.{0,40}另一方面"),
    re.compile(r"既.{0,20}又"),
    re.compile(r"不仅.{0,30}而且"),
    re.compile(r"首先.{0,80}其次.{0,80}(最后|再次)"),
)

#: 数字（含百分号、单位）
_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|个|倍|次|秒|ms|GB|MB)?")

#: 中文数字 + 量词。
#:
#: ★ 为什么必须数中文数字（实测踩过）：只数阿拉伯数字时，
#: 「两个真实业务场景」「不到一百行」这类**完全正常的真人写作**
#: 会被判为"缺少具体信息"，于是系统会去硬改本来就好的文字。
#: 中文写作里数字常常写成汉字，漏掉它会让这个特征系统性偏低。
_CN_NUMBER = re.compile(
    r"[一二三四五六七八九十百千万两半几]+\s*"
    r"(?:个|次|年|月|日|倍|成|种|类|项|条|行|步|轮|层|维|组|篇|页|章|节|位|名)"
)

#: 长得像「数字 + 量词」但其实是固定说法，**计数前先剔除**。
#:
#: ★ 实测踩过：`_CN_NUMBER` 把「进一步」匹配成「一步」（`步` 在量词表里），
#: 于是一个纯 idiom 被算成了"具体信息点" → `specificity` 虚高 →
#: 本来该被判"缺少具体信息"的段落**逃过诊断**。
#:
#: 这个 bug 是在接真实模型做 E5 改写时暴露的：模型的改写里多了个「进一步」，
#: specificity 就从 0.000 跳到 3.448 —— 一个纯 idiom 让指标翻了个身。
_CN_NUMBER_IDIOMS: tuple[str, ...] = (
    "进一步",
    "进一步地",
    "第一步",   # 「第一步」在方法论里是结构标记，不是可核对的量
)


def _count_cn_numbers(text: str) -> int:
    """数中文数字，**扣掉 idiom 造成的假阳性**。"""
    cleaned = text
    for idiom in _CN_NUMBER_IDIOMS:
        cleaned = cleaned.replace(idiom, "")
    return len(_CN_NUMBER.findall(cleaned))

#: 拉丁专名（如 LLM、BERT）与书名
_LATIN_TERM = re.compile(r"\b[A-Za-z][A-Za-z0-9\-]{1,}\b")
_BOOK_TITLE = re.compile(r"《[^》]{1,30}》")


def _sentences(text: str) -> list[str]:
    return [t for _s, _e, t in iter_segments(text) if t.strip()]


def _cv(values: list[float]) -> float:
    """变异系数（标准差 / 均值）。**越小越均匀**。

    为什么用 CV 而不是标准差：长短段落的绝对差异无法跨文本比较，
    用相对差异才能判断"是不是过于均匀"。
    """
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return (var ** 0.5) / mean


@dataclass
class StyleProfile:
    """一段文字的风格画像。"""

    chars: int
    sentences: int
    avg_sentence_length: float
    #: 句长变异系数 —— 越小说明句式越均匀
    sentence_length_cv: float
    #: 每句平均的套话连接词数
    transition_density: float
    #: 词汇丰富度（type-token ratio）
    lexical_diversity: float
    #: 每百字的具体信息量（数字 + 拉丁专名 + 书名）
    specificity: float
    #: 每千字的排比/对仗框架数
    parallelism: float

    def to_dict(self) -> dict:
        return {
            "chars": self.chars,
            "sentences": self.sentences,
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "sentence_length_cv": round(self.sentence_length_cv, 3),
            "transition_density": round(self.transition_density, 3),
            "lexical_diversity": round(self.lexical_diversity, 3),
            "specificity": round(self.specificity, 3),
            "parallelism": round(self.parallelism, 3),
        }


def profile(text: str) -> StyleProfile:
    """计算风格画像。

    所有量都是**直接可复核**的：给出同一段文字，任何人算出来的值应当一致。
    这也是"可测量的客观特征"与"身份推断"的区别。
    """
    sents = _sentences(text)
    n_sent = max(1, len(sents))
    chars = len(text)

    lengths = [len(s) for s in sents]

    transitions = sum(text.count(t) for t in TRANSITIONS)
    numbers = len(_NUMBER.findall(text)) + _count_cn_numbers(text)
    latin_terms = len({t for t in _LATIN_TERM.findall(text) if len(t) >= 2})
    titles = len(_BOOK_TITLE.findall(text))
    parallel = sum(1 for p in PARALLEL_PATTERNS if p.search(text))

    # 词汇丰富度：中文按二字组近似（没有分词器时这是最稳的近似）
    compact = re.sub(r"\s", "", text)
    if len(compact) >= 2:
        bigrams = {compact[i : i + 2] for i in range(len(compact) - 1)}
        ttr = len(bigrams) / max(1, len(compact) - 1)
    else:
        ttr = 0.0

    return StyleProfile(
        chars=chars,
        sentences=len(sents),
        avg_sentence_length=(sum(lengths) / len(lengths)) if lengths else 0.0,
        sentence_length_cv=_cv([float(x) for x in lengths]),
        transition_density=transitions / n_sent,
        lexical_diversity=ttr,
        specificity=((numbers + latin_terms + titles) / chars * 100) if chars else 0.0,
        parallelism=(parallel / chars * 1000) if chars else 0.0,
    )
