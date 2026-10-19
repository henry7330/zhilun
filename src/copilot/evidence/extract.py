"""claim 原子化：把一段文字拆成「可独立判断真伪」的最小断言。

为什么要原子化
--------------
"该方法在五个数据集上提升 5%，且在低资源场景下降 12%" 是**两条**结论。
如果不拆：

- 无法定位到底哪一部分没有证据（是"提升"缺证据，还是"下降"缺证据？）
- 只要有一条对不上，整句就被判伪，粒度太粗
- 引用支持率这类指标会被稀释

拆得太细的代价同样真实（产生大量无意义碎片、指标虚高），所以拆分策略
本身就是一个需要对照实验的选择 —— 见 P9 消融。

本模块提供两种实现
------------------
- :class:`RuleBasedExtractor`：**确定性、离线、可复现**，作为基线。
- ``LlmExtractor``：后续加入（需要模型），用它与基线做消融对照。

先有基线再有"更好的方案"，才能回答"到底好多少"。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from copilot.evidence.models import Claim, SupportStatus
from copilot.evidence.text import content_tokens, has_hedge, numbers_in

__all__ = [
    "RuleBasedExtractor",
    "ClaimExtractor",
    "ExtractionStats",
    "split_sentences",
    "split_clauses",
    "MIN_FRAGMENT_CHARS",
]

#: 过短的碎片不构成一条可验证断言（例如切分残留的"以及""因此"）。
#:
#: 取值 4 而非更大值，是因为**中文信息密度高**：
#: "降低了延迟"只有 5 个字，却是一条完整断言。阈值设成 6 会误杀大量
#: 合法的短 claim。真正过滤碎片的主力是下面的 ``_is_meaningful``
#: （要求至少 2 个内容单元），字符数只用来挡掉"好的""因此"这类残留。
MIN_FRAGMENT_CHARS = 4

# 句子边界：中英标点 + 换行
_SENTENCE_SPLIT = re.compile(r"[。！？；!?;\n]+")

# 子句连接词：只收**连接两个独立断言**的强连接词。
# 刻意不收"和/与/及/或" —— 它们多用于名词并列（"A 和 B 的方法"），
# 在此处切分会把名词短语撕成碎片，制造假 claim。
_CLAUSE_SPLIT = re.compile(
    r"(?:并且|而且|同时|此外|另外|以及|但是|然而|不过|因此|所以|故而|从而|而且)"
)

# 保持 confidence 的基准与折扣
_BASE_CONFIDENCE = 0.8
_HEDGE_PENALTY = 0.5


def _strip_punct(text: str) -> str:
    """去掉首尾标点与空白，便于统计与展示。"""
    return text.strip().strip("，,、:：()（）[]【】\"'“”‘’ ")


def split_sentences(text: str) -> list[str]:
    """按句子边界切分。"""
    if not text:
        return []
    return [s for s in (_strip_punct(p) for p in _SENTENCE_SPLIT.split(text)) if s]


def split_clauses(sentence: str) -> list[str]:
    """按强连接词把句子切成子句。"""
    if not sentence:
        return []
    return [c for c in (_strip_punct(p) for p in _CLAUSE_SPLIT.split(sentence)) if c]


def _is_meaningful(fragment: str) -> bool:
    """过滤切分残留的碎片。

    判据用「内容单元数」而不是纯字符数：二元组数量能更好地反映
    是否包含实质信息，避免把"因此""以及"这类连接词残留当成 claim。
    """
    if len(fragment) < MIN_FRAGMENT_CHARS:
        return False
    return len(content_tokens(fragment)) >= 2


class ClaimExtractor(Protocol):
    """原子化接口。便于把规则基线与模型方案做替换对照。"""

    def extract(self, text: str, *, prefix: str = "c") -> list[Claim]: ...


@dataclass
class ExtractionStats:
    """记录切分过程的规模，便于诊断"是不是拆得太碎/太粗"。"""

    sentences: int = 0
    clauses: int = 0
    kept: int = 0
    dropped: int = 0

    @property
    def drop_ratio(self) -> float:
        total = self.kept + self.dropped
        return self.dropped / total if total else 0.0


class RuleBasedExtractor:
    """规则式原子化：**确定性、离线、可复现**的基线。

    流程：句子切分 → 子句切分 → 过滤碎片 → 打标（数字 / 对冲）。

    ``confidence`` 在此阶段只反映**表述的确定性程度**（含"可能""大约"
    这类对冲词就调低），不代表是否有证据支持 —— 后者由支持判断负责。
    把这两件事分开，是为了避免"措辞谨慎"被误当成"证据不足"。
    """

    def __init__(self, *, min_fragment_chars: int = MIN_FRAGMENT_CHARS) -> None:
        self.min_fragment_chars = min_fragment_chars
        self.last_stats = ExtractionStats()

    def extract(self, text: str, *, prefix: str = "c") -> list[Claim]:
        stats = ExtractionStats()
        claims: list[Claim] = []

        for sentence in split_sentences(unicodedata.normalize("NFKC", text or "")):
            stats.sentences += 1
            for clause in split_clauses(sentence) or [sentence]:
                stats.clauses += 1
                if len(clause) < self.min_fragment_chars or not _is_meaningful(clause):
                    stats.dropped += 1
                    continue

                confidence = _BASE_CONFIDENCE
                if has_hedge(clause):
                    confidence -= _HEDGE_PENALTY
                # 含具体数字的断言更可核验，给一点加成便于排序
                if numbers_in(clause):
                    confidence = min(1.0, confidence + 0.1)

                claims.append(
                    Claim(
                        claim_id=f"{prefix}-{len(claims) + 1}",
                        text=clause,
                        status=SupportStatus.UNSUPPORTED,
                        confidence=round(confidence, 3),
                    )
                )
                stats.kept += 1

        self.last_stats = stats
        return claims
