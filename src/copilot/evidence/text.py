"""文本特征工具：无第三方依赖，中英混排可用。

为什么自己写而不上分词器
------------------------
1. 覆盖率只需要一个**稳定、可复现**的相似度代理，不需要语言学级分词；
2. 引 jieba 之类会引入词典依赖，且分词结果随版本变化，破坏可复现性；
3. 本模块的所有函数都是**纯函数**，便于单测与消融对照。

中文用「字符二元组」，英文用「小写词」。这样对中文有效，
又不至于因为分词粒度问题引入噪声。
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "content_tokens",
    "tokenize",
    "coverage",
    "numbers_in",
    "normalize_number",
    "has_hedge",
    "has_negation",
    "polarity_conflicts",
    "ANTONYM_AXES",
    "ANTONYM_PAIRS",
    "HEDGE_WORDS",
    "NEGATION_WORDS",
]

# ── 分词 ──────────────────────────────────────────────────────

_LATIN_WORD = re.compile(r"[a-zA-Z0-9]+")


def _is_cjk(ch: str) -> bool:
    """判断是否为中日韩表意文字（用 Unicode 类别，避免硬编码码点区间出错）。"""
    if "\u4e00" <= ch <= "\u9fff":       # CJK 统一表意文字
        return True
    if "\u3400" <= ch <= "\u4dbf":       # 扩展 A
        return True
    if "\uf900" <= ch <= "\ufaff":       # 兼容表意文字
        return True
    return False


def tokenize(text: str) -> list[str]:
    """切成**带重复**的 token 列表（BM25 需要词频，不能用集合）。

    切分规则与 :func:`content_tokens` 完全一致（同一套归一化与 CJK 二元组），
    只是保留重复。两者共用实现，避免"评分用一个口径、检索用另一个口径"
    这种难以察觉的不一致。
    """
    if not text:
        return []

    text = unicodedata.normalize("NFKC", text)
    tokens: list[str] = list(_LATIN_WORD.findall(text.lower()))

    for run in re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+", text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    return tokens


def content_tokens(text: str) -> set[str]:
    """把文本切成可比较的内容单元**集合**（用于覆盖率类指标）。

    - 拉丁字母/数字：按词取，转小写；
    - CJK：取相邻两字组成的二元组（单字太容易撞，三元组太严）。

    归一化（NFKC）后再切，避免全角/半角差异造成假不匹配。
    """
    return set(tokenize(text))


def coverage(needle: str, haystack: str) -> float:
    """needle 的内容单元有多大比例出现在 haystack 中。

    返回 0.0–1.0。空 needle 返回 0.0（而不是 1.0）——
    "没有内容"不应被当成"完全覆盖"。
    """
    n = content_tokens(needle)
    if not n:
        return 0.0
    h = content_tokens(haystack)
    return len(n & h) / len(n)


# ── 数字 ──────────────────────────────────────────────────────

_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|倍|个百分点|个|篇|条|次|例|名)?")

_UNIT_NORMALIZE = {"％": "%"}


def normalize_number(raw: str) -> str:
    """归一化数字写法，让 ``5 %``、``5%``、``5％`` 视为同一个。"""
    text = unicodedata.normalize("NFKC", raw).strip()
    for old, new in _UNIT_NORMALIZE.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", "", text)


def numbers_in(text: str) -> tuple[str, ...]:
    """抽取文本中的数字（含单位），保持出现顺序并去重。

    数字是最强的事实锚点：``提升 5%`` 与证据里的 ``下降 12%``
    既数字不符又极性相反，这类矛盾用规则就能确定性地抓到。
    """
    if not text:
        return ()
    seen: dict[str, None] = {}
    for m in _NUMBER.finditer(unicodedata.normalize("NFKC", text)):
        seen.setdefault(normalize_number(m.group()), None)
    return tuple(seen)


# ── 对冲与否定 ────────────────────────────────────────────────

HEDGE_WORDS = (
    "可能", "或许", "大概", "也许", "似乎", "大约", "约",
    "倾向于", "一般来说", "通常", "推测", "估计", "有望", "应该",
    "may", "might", "perhaps", "possibly", "probably", "likely",
    "suggests", "appears", "seems", "approximately", "roughly",
)

NEGATION_WORDS = (
    "不", "无", "未", "非", "没有", "无法", "不能", "并非", "缺乏",
    "not", "no", "never", "without", "cannot", "none",
)


def has_hedge(text: str) -> bool:
    """是否含不确定性措辞。用于调低 confidence，而不是直接判伪。"""
    low = unicodedata.normalize("NFKC", text or "").casefold()
    return any(w in low for w in HEDGE_WORDS)


def has_negation(text: str) -> bool:
    low = unicodedata.normalize("NFKC", text or "").casefold()
    return any(w in low for w in NEGATION_WORDS)


# ── 极性（反义轴）────────────────────────────────────────────

#: 反义「轴」。每条轴的两组词语义上互斥，命中**跨组**即为极性冲突。
#:
#: 为什么用「轴」而不是两两配对：配对枚举必然遗漏。例如"降低"的对立面
#: 不只是"提高"，还有"上升""增长"；两两枚举写不全，按轴分组则自然覆盖
#: 同义表达。同时，分轴（升降/好坏/立场）能避免跨语义轴的误报 ——
#: "改善"与"上升"不是对立关系，不该被判成矛盾。
ANTONYM_AXES: tuple[tuple[str, frozenset[str], frozenset[str]], ...] = (
    (
        "升降",
        frozenset({"提升", "提高", "增加", "上升", "增长", "增强", "增大",
                   "高于", "超过", "加快", "加速", "增高",
                   "increase", "higher", "grow", "faster"}),
        frozenset({"下降", "降低", "减少", "减小", "减弱", "低于",
                   "减慢", "减速", "放缓", "回落",
                   "decrease", "lower", "reduce", "slower"}),
    ),
    (
        "好坏",
        frozenset({"改善", "优化", "有效", "优点", "优于", "成功", "通过",
                   "improve", "better", "effective", "success"}),
        frozenset({"恶化", "劣化", "无效", "缺点", "劣于", "失败", "未通过",
                   "degrade", "worse", "ineffective", "failure"}),
    ),
    (
        "立场",
        frozenset({"支持", "赞同", "认可"}),
        frozenset({"反对", "否定", "质疑"}),
    ),
)

#: 兼容旧名（部分调用方可能引用）。新代码请用 :data:`ANTONYM_AXES`。
ANTONYM_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (p, n) for _, pos, neg in ANTONYM_AXES for p in sorted(pos) for n in sorted(neg)
)


def polarity_conflicts(claim_text: str, evidence_text: str) -> tuple[tuple[str, str], ...]:
    """找出 claim 与证据之间的极性冲突。

    返回 ``((claim 中的词, 证据中的词), ...)``，每轴最多一条。
    这是**确定性**的矛盾检测：结论说"降低延迟"而原文说"延迟上升"，
    不需要任何模型就能判定二者不相容。

    已知边界（有意为之，避免误报）：若 **claim 自身**在同一轴上同时出现
    两组词（如"提升了准确率但降低了延迟"未被成功拆分），则该轴无法判定
    方向，直接跳过。宁可漏报，不可误报 —— 误报率是要一并报告的代价指标。
    """
    a = unicodedata.normalize("NFKC", claim_text or "").casefold()
    b = unicodedata.normalize("NFKC", evidence_text or "").casefold()
    found: list[tuple[str, str]] = []

    for _axis, pos, neg in ANTONYM_AXES:
        a_pos = sorted(w for w in pos if w in a)
        a_neg = sorted(w for w in neg if w in a)
        if a_pos and a_neg:
            continue  # claim 内部混合极性，方向不可判定
        b_pos = sorted(w for w in pos if w in b)
        b_neg = sorted(w for w in neg if w in b)

        if a_pos and b_neg:
            found.append((a_pos[0], b_neg[0]))
        elif a_neg and b_pos:
            found.append((a_neg[0], b_pos[0]))

    return tuple(found)

