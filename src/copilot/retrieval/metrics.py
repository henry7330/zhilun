"""检索评测指标与统计。

为什么指标要带置信区间
----------------------
单次跑出的一个数字（例如"Recall@5 = 91.2%"）**无法说明任何稳定性问题**：
换个随机种子、换一批样本，它可能是 88% 也可能是 94%。
没有区间就无法判断"改了检索策略后 +1.5% 到底是真提升还是噪声"。

因此本模块的所有汇总指标都提供 bootstrap 置信区间，
并且 bootstrap 使用**固定种子**，保证同一份数据跑出同样的区间。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "EvalQuery",
    "QueryMetrics",
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "bootstrap_ci",
    "evaluate_retriever",
    "summarize",
]


@dataclass(frozen=True)
class EvalQuery:
    """一条评测查询。

    ``gold_doc_ids`` 是标准答案集合。多文档是常态（一个主题可能对应多篇），
    所以用集合而不是单个 id。
    """

    query_id: str
    text: str
    gold_doc_ids: frozenset[str]
    kind: str = "known_item"


# ─────────────────────── 单查询指标 ───────────────────────


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """前 k 个结果覆盖了多少比例的标准答案。"""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return len(set(ranked[:k]) & gold_set) / len(gold_set)


def precision_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    return len(set(ranked[:k]) & gold_set) / k


def mrr_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """第一个命中位置的倒数（Mean Reciprocal Rank 的单查询分量）。

    对"用户只想找到那一篇"的场景比 Recall 更贴切。
    """
    gold_set = set(gold)
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """二值相关性的 nDCG@k。

    标准答案通常没有分级相关性标注，所以用二值（相关/不相关）。
    理想排序 = 所有相关文档排在最前。
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in gold_set
    )
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ─────────────────────── 统计 ───────────────────────


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20260101,
) -> tuple[float, float]:
    """对均值做 bootstrap 置信区间，返回 ``(下界, 上界)``。

    用固定种子：否则同一份数据每次跑出不同区间，可复现性就没了。
    样本量为 1 时区间退化为该点值（无法估计方差）。
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


# ─────────────────────── 评测执行 ───────────────────────


@dataclass
class QueryMetrics:
    query_id: str
    kind: str
    retrieved: list[str]
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: float = 0.0


@dataclass
class EvalReport:
    """一次评测的完整结果，便于落盘与对照。"""

    retriever: str
    ks: tuple[int, ...]
    queries: list[QueryMetrics]
    depth: int

    def values(self, metric: str, k: int | None = None) -> list[float]:
        if metric == "recall":
            assert k is not None
            return [q.recall.get(k, 0.0) for q in self.queries]
        if metric == "mrr":
            return [q.mrr for q in self.queries]
        if metric == "ndcg":
            return [q.ndcg for q in self.queries]
        raise ValueError(f"未知指标：{metric}")

    def summary(self, *, ci: bool = True) -> dict[str, dict[str, float]]:
        """汇总成 ``{指标名: {"mean":…, "lo":…, "hi":…}}``。"""
        out: dict[str, dict[str, float]] = {}
        for k in self.ks:
            vals = self.values("recall", k)
            entry = {"mean": float(np.mean(vals))}
            if ci:
                lo, hi = bootstrap_ci(vals)
                entry.update({"lo": lo, "hi": hi})
            out[f"recall@{k}"] = entry

        for name in ("mrr", "ndcg"):
            vals = self.values(name)
            entry = {"mean": float(np.mean(vals))}
            if ci:
                lo, hi = bootstrap_ci(vals)
                entry.update({"lo": lo, "hi": hi})
            out[name] = entry
        return out


def evaluate_retriever(
    retriever,
    queries: Sequence[EvalQuery],
    *,
    ks: Sequence[int] = (1, 5, 10),
    depth: int = 10,
) -> EvalReport:
    """跑一个检索器，产出逐查询指标。"""
    ks = tuple(sorted(set(ks)))
    depth = max(depth, max(ks))
    per_query: list[QueryMetrics] = []

    for q in queries:
        results = retriever.retrieve(q.text, k=depth)
        # 逐片段记录，再聚合成文档级排名（max-pool 在 rank_documents 内完成）
        from copilot.retrieval.base import rank_documents

        ranked = rank_documents(results, k=depth)
        per_query.append(
            QueryMetrics(
                query_id=q.query_id,
                kind=q.kind,
                retrieved=ranked,
                recall={k: recall_at_k(ranked, q.gold_doc_ids, k) for k in ks},
                mrr=mrr_at_k(ranked, q.gold_doc_ids, depth),
                ndcg=ndcg_at_k(ranked, q.gold_doc_ids, depth),
            )
        )

    return EvalReport(retriever=getattr(retriever, "name", "unknown"), ks=ks, queries=per_query, depth=depth)


def summarize(reports: Sequence[EvalReport]) -> str:
    """把多个检索器的结果对齐成一张对照表（消融实验的输出）。"""
    if not reports:
        return "(无结果)"

    metrics = list(reports[0].summary().keys())
    header = f"{'检索器':<22}" + "".join(f"{m:>18}" for m in metrics)
    lines = [header, "-" * len(header)]
    for rep in reports:
        s = rep.summary()
        row = f"{rep.retriever:<22}"
        for m in metrics:
            e = s[m]
            cell = f"{e['mean']:.3f}"
            if "lo" in e:
                cell += f" [{e['lo']:.2f},{e['hi']:.2f}]"
            row += f"{cell:>18}"
        lines.append(row)
    lines.append("")
    lines.append("注：方括号为 95% bootstrap 置信区间（固定种子，可复现）。")
    return "\n".join(lines)
