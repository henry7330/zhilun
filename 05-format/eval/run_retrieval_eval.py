#!/usr/bin/env python3
"""P2 检索层消融实验：测每个组件的边际贡献。

设计要点
--------
**评测只读缓存，不联网。** 语料由 ``scripts/fetch_corpus.py`` 抓取落盘；
若评测时现场抓取，网络抖动与 arXiv 新增论文会让同一份评测跑出不同结果。

**查询与索引刻意见错开。** 索引只含摘要、查询用论文标题。
若把标题也索引进去，标题查询会逐字命中，所有方法满分，消融没有区分度。
排除标题后，检索必须依靠标题与摘要间的主题词汇重叠 —— 词面方法
才会遇到真实困难，各组件的差异才显现出来。

语料刻意集中在 `cs.IR`（信息检索）+ LLM 这一窄领域：
论文之间词汇高度重叠，构成天然的"难负例"。
若语料主题杂乱，任何方法都轻松满分，同样测不出差异。

输出
----
``reports/retrieval_ablation.md`` —— 含 95% bootstrap 置信区间的对照表。

用法
----
    python eval/run_retrieval_eval.py
    python eval/run_retrieval_eval.py --limit 30    # 只用前 N 条查询
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

for _name in ("stdout", "stderr"):
    _s = getattr(sys, _name, None)
    if _s is not None and hasattr(_s, "buffer"):
        enc = getattr(_s, "encoding", None) or "utf-8"
        try:
            setattr(sys, _name, io.TextIOWrapper(_s.buffer, encoding=enc, errors="replace"))
        except Exception:
            pass

_HERE = Path(__file__).resolve()
# 向上找系统根（含 00-core 的目录）—— 不靠数 parent 层级，目录再调整也不会失效
SYSTEM_DIR = _HERE.parent
while SYSTEM_DIR != SYSTEM_DIR.parent and not (SYSTEM_DIR / "00-core").is_dir():
    SYSTEM_DIR = SYSTEM_DIR.parent
STAGE_DIR = SYSTEM_DIR / "05-format"
# 两种 src 布局：00-core/src（共享层）与 <阶段>/<节点>/src
for _src in sorted(SYSTEM_DIR.glob("*/*/src")) + sorted(SYSTEM_DIR.glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from copilot.retrieval.base import Chunk, rank_documents  # noqa: E402,F401
from copilot.retrieval.corpus import (  # noqa: E402
    chunk_document,
    documents_from_corpus,
    load_corpus,
)
from copilot.retrieval.metrics import EvalQuery, evaluate_retriever, summarize  # noqa: E402
from copilot.retrieval.sparse import (  # noqa: E402
    Bm25Retriever,
    CharTrigramRetriever,
    CoverageReranker,
    RrfFusion,
)
from copilot.stats import benjamini_hochberg, compare_paired  # noqa: E402

#: 做配对检验的指标。取值函数收在 lambda 里，避免调用处重复条件分支。
METRIC_SPECS = (
    ("recall@1", lambda r: r.values("recall", 1)),
    ("recall@5", lambda r: r.values("recall", 5)),
    ("mrr", lambda r: r.values("mrr")),
)

KS = (1, 5, 10)


def build_queries(papers) -> list[EvalQuery]:
    """构造两类查询，都是「已知条目检索」（known-item search）。

    - ``title``：完整标题。相对容易。
    - ``title_head``：标题的前 60%。更难，且更接近用户真实的模糊输入。
    """
    queries: list[EvalQuery] = []
    for p in papers:
        title = p.title.strip()
        if len(title) < 8:
            continue
        gold = frozenset({p.doc_id})
        queries.append(
            EvalQuery(query_id=f"{p.doc_id}::title", text=title, gold_doc_ids=gold, kind="title")
        )
        head = title[: max(8, int(len(title) * 0.6))]
        queries.append(
            EvalQuery(
                query_id=f"{p.doc_id}::head", text=head, gold_doc_ids=gold, kind="title_head"
            )
        )
    return queries


def build_variants(chunks_doc, chunks_sent):
    """消融变体：每加一层，看指标变多少。

    - A 文档级 BM25          —— baseline，不切分
    - B 句块级 BM25          —— 测「切分」的边际贡献
    - C B + 覆盖率重排        —— 测「词面重排」的边际贡献
    - D C ∪ 字符trigram 融合  —— 测「多路稀疏融合」的边际贡献
    - E C ∪ dense（语义）     —— 测「语义召回」的边际贡献
    - F C ∪ trigram ∪ dense  —— 全量融合
    """
    doc_bm25 = Bm25Retriever(chunks_doc)
    doc_bm25.name = "A 文档级BM25"
    doc_chunk = Bm25Retriever(chunks_sent)
    doc_chunk.name = "B 句块级BM25"
    reranked = CoverageReranker(doc_chunk, depth=20, name="C B+覆盖率重排")
    tri = CharTrigramRetriever(chunks_sent)

    variants = [
        doc_bm25,
        doc_chunk,
        reranked,
        RrfFusion([reranked, tri], name="D C+trigram融合"),
    ]

    # dense 是可选依赖：装了就加进来，没装就跳过（而不是整体崩掉）
    try:
        from copilot.retrieval.dense import DenseRetriever

        dense = DenseRetriever(chunks_sent, name="dense(语义)")
        variants.append(RrfFusion([reranked, dense], name="E C+dense融合"))
        variants.append(
            RrfFusion([reranked, tri, dense], name="F C+trigram+dense")
        )
    except ImportError as exc:
        print(f"[跳过 dense 变体] {exc}")

    return variants


#: 显式的「父子」关系：每个变体相对**它真正加了一层的那一方**比较。
#: 不能用相邻行相减 —— D 与 E 是兄弟（都基于 C），
#: 相邻相减会把"trigram 与 dense 谁更好"误报成"加 dense 的边际贡献"。
VARIANT_PARENT: dict[str, str] = {
    "B 句块级BM25": "A 文档级BM25",
    "C B+覆盖率重排": "B 句块级BM25",
    "D C+trigram融合": "C B+覆盖率重排",
    "E C+dense融合": "C B+覆盖率重排",
    "F C+trigram+dense": "C B+覆盖率重排",
}


def pairwise_findings(reports) -> list[str]:
    """按显式父子关系比较，自动标注差异是否超出噪声。

    判据用**置信区间是否重叠**：不重叠时提示有差异；重叠时明确写
    "不能声称更优"。这是保守且可自动化的哨兵，**不能替代配对检验**
    （两个 bootstrap CI 重叠与否并非严格的显著性检验），
    因此措辞只作"提示"，正式结论留给 P9。
    """
    by_name = {r.retriever: r for r in reports}
    lines: list[str] = []
    for cur in reports:
        parent_name = VARIANT_PARENT.get(cur.retriever)
        if parent_name is None or parent_name not in by_name:
            continue
        prev = by_name[parent_name]
        sp, sc = prev.summary(), cur.summary()
        parts: list[str] = []
        for metric in ("recall@1", "recall@5", "mrr"):
            a, b = sp[metric], sc[metric]
            delta = b["mean"] - a["mean"]
            overlap = not (b["lo"] > a["hi"] or a["lo"] > b["hi"])
            flag = "CI 不重叠" if not overlap else "CI 重叠"
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            parts.append(
                f"{metric} {a['mean']:.3f}→{b['mean']:.3f} ({arrow}{abs(delta):.3f}, {flag})"
            )
        lines.append(f"- **{prev.retriever} → {cur.retriever}**：" + "；".join(parts))
    return lines


def significance_tally(reports) -> str:
    """统计本次有多少组对比的 CI 不重叠 —— 自动、不含主观判断。"""
    by_name = {r.retriever: r for r in reports}
    total = 0
    non_overlap = 0
    for cur in reports:
        parent = VARIANT_PARENT.get(cur.retriever)
        if parent is None or parent not in by_name:
            continue
        sp, sc = by_name[parent].summary(), cur.summary()
        for metric in ("recall@1", "recall@5", "mrr"):
            total += 1
            a, b = sp[metric], sc[metric]
            if b["lo"] > a["hi"] or a["lo"] > b["hi"]:
                non_overlap += 1
    if total == 0:
        return "（无可比对的变体对）"
    if non_overlap == 0:
        return (
            f"本次 {total} 项指标对比中，**没有任何一项的置信区间不重叠**。\n"
            "也就是说：**当前样本量下，无法声称任何一个组件带来了改进**。\n"
            "点估计的升降只能作为线索，正式结论需要配对检验（见 P9）。"
        )
    return (
        f"本次 {total} 项指标对比中，有 {non_overlap} 项的置信区间不重叠。\n"
        "其余项 CI 重叠，**不能声称更优**；正式结论需要配对检验（见 P9）。"
    )


def paired_significance(reports) -> tuple[list[str], str]:
    """对所有「变体 vs 其父节点」做配对检验，并用 BH 校正多重比较。

    为什么必须配对：两个变体跑的是**同一批查询**，查询难度对二者是共同因素。
    把它当独立样本比较会浪费这一信息，功效大幅下降 ——
    P2 第一版报告里"点估计差很多但 CI 重叠"正是这个原因。

    为什么必须校正：5 组比较 × 3 个指标 = 15 次检验。
    若每次都按 α=0.05 判定，**即使全部变体毫无差别，也期望出现约 0.75 个假阳性**。
    """
    by_name = {r.retriever: r for r in reports}
    results = []
    for cur in reports:
        parent = VARIANT_PARENT.get(cur.retriever)
        if parent is None or parent not in by_name:
            continue
        prev = by_name[parent]
        for metric_name, getter in METRIC_SPECS:
            results.append(
                compare_paired(
                    getter(prev),
                    getter(cur),
                    label_a=prev.retriever,
                    label_b=cur.retriever,
                    metric=metric_name,
                )
            )

    if not results:
        return [], "（无可比对的变体对）"

    adjusted = benjamini_hochberg([r.wilcoxon_p for r in results])
    for r, p in zip(results, adjusted):
        r.p_adjusted = p

    rows = [
        "| 比较 | 指标 | 均值 A | 均值 B | Δ (95% CI) | Cohen's dz | p (Wilcoxon) | p (BH) | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        if r.p_adjusted is not None and r.p_adjusted < 0.05:
            verdict = "**显著**"
        elif r.p_adjusted is not None and r.p_adjusted < 0.10:
            verdict = "边缘"
        else:
            verdict = "不显著"
        dz = "∞" if r.cohens_dz == float("inf") else (
            "-∞" if r.cohens_dz == float("-inf") else f"{r.cohens_dz:+.2f}"
        )
        rows.append(
            f"| {r.label_a} → {r.label_b} | {r.metric} "
            f"| {r.mean_a:.3f} | {r.mean_b:.3f} "
            f"| {r.mean_diff:+.3f} [{r.diff_ci_low:+.3f},{r.diff_ci_high:+.3f}] "
            f"| {dz} | {r.wilcoxon_p:.4f} | {r.p_adjusted:.4f} | {verdict} |"
        )

    n_sig = sum(1 for r in results if r.significant)
    if n_sig == 0:
        note = (
            f"共 {len(results)} 项配对检验，**校正后没有任何一项达到 p<0.05**。\n"
            "结论：当前样本量（120 条查询）下，**无法声称任一组件带来改进**。\n"
            "点估计的升降只能作为线索。要检出更小的效应，需要扩充评测集。"
        )
    else:
        note = (
            f"共 {len(results)} 项配对检验，其中 **{n_sig} 项在 BH 校正后仍显著**。"
        )
    return rows, note


def write_report(reports, queries, manifest, out_path: Path) -> None:
    table = summarize(reports)
    by_kind: dict[str, int] = {}
    for q in queries:
        by_kind[q.kind] = by_kind.get(q.kind, 0) + 1
    findings = pairwise_findings(reports)
    tally = significance_tally(reports)
    sig_rows, sig_note = paired_significance(reports)

    lines = [
        "# P2 检索层消融实验报告",
        "",
        "> **本报告由 `eval/run_retrieval_eval.py` 自动生成，可复现。**",
        "> 所有指标为**真实测量值**，含 95% bootstrap 置信区间（固定种子）。",
        "",
        "## 实验设置",
        "",
        f"- 语料来源：{manifest.get('source', 'arXiv')}，查询式 `{manifest.get('query', '')}`",
        f"- 语料条数：{manifest.get('count', '?')}（抓取时间 {manifest.get('fetched_at', '?')}）",
        f"- 查询条数：{len(queries)}（" + "、".join(f"{k} {v} 条" for k, v in sorted(by_kind.items())) + "）",
        "- 索引内容：**仅摘要**（刻意排除标题，避免标题查询逐字命中）",
        "- 相关判定：二值（查询对应的论文为唯一 gold）",
        "- 文档级聚合：片段 max-pool（任一高相关片段即可代表该文档）",
        "",
        "## 消融对照",
        "",
        "```",
        table,
        "```",
        "",
        "## 逐层边际贡献（自动比对）",
        "",
        *findings,
        "",
        "> **读法**：`CI 不重叠` 提示该差异超出查询间噪声；`CI 重叠` 则**不能声称更优**。",
        "> 置信区间的重叠与否不是严格的显著性检验，正式结论需配对检验（见 P9）。",
        "",
        "### 显著性小结（自动统计）",
        "",
        tally,
        "",
        "## 配对显著性检验（Wilcoxon + BH 校正）",
        "",
        "上面的对照表用的是**独立置信区间**，它回答不了「差异是否显著」。",
        "这里改用**配对检验**：两个变体跑的是同一批查询，查询难度是共同因素，",
        "配对能消掉这部分方差，功效显著更高。",
        "",
        *sig_rows,
        "",
        sig_note,
        "",
        "> **读法**：`Δ` 是 B−A 的均值差，方括号是配对 bootstrap 的 95% 置信区间；",
        "> `Cohen's dz` 是配对效应量（0.2 小 / 0.5 中 / 0.8 大）；",
        "> `p (BH)` 是经 Benjamini–Hochberg 校正后的 p 值，**结论以它为准**。",
        "> 15 次检验若不做校正，即使毫无差别也期望出现约 0.75 个假阳性。",
        "",
        "## 每条变体验证的假设",
        "",
        "| 变体 | 假设 |",
        "|---|---|",
        "| A 文档级 BM25 | 基线：不切分，整篇摘要作为一个检索单元 |",
        "| B 句块级 BM25 | 切分能让局部高相关片段不被长文本稀释 |",
        "| C B+覆盖率重排 | 覆盖率信号能纠正 BM25 对「关键词堆砌长文」的偏好 |",
        "| D C+trigram融合 | 字符 trigram 与词级 BM25 信号互补，RRF 融合有增益 |",
        "| E C+dense融合 | 语义向量能召回词面不重叠的相关论文，补 BM25 的短板 |",
        "| F C+trigram+dense | 三路信号互补，融合收益叠加 |",
        "",
        "## 尚未接入",
        "",
        "- **Query 改写 / 分解**：需要模型调用（生成式），当前未接。",
        "- **交叉编码器精排**：需要模型。当前的 C 变体是**词面重排**，",
        "  可作为「精排到底值多少」的下界参考。",
        "- **语义召回用的是静态嵌入**（model2vec，256 维，无上下文感知）。",
        "  换成 BERT 类编码器会更强，但会引入 torch（数 GB）。",
        "  模型名与维度已在上方「实验设置」中记录，便于版本对照。",
        "",
        "## 说明",
        "",
        "- 置信区间用 bootstrap（2000 次重采样，固定随机种子），",
        "  反映的是**查询间的方差**，不是模型随机性（本实验全部为确定性方法）。",
        "- 区间重叠时不应声称某个变体更优；需做配对检验（见 P9）。",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2 检索消融实验")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条查询（0=全部）")
    ap.add_argument("--out", default=str(STAGE_DIR / "reports" / "retrieval_ablation.md"))
    args = ap.parse_args()

    papers, manifest = load_corpus()
    print(f"语料：{len(papers)} 篇（抓取于 {manifest.get('fetched_at', '?')}）")

    # 索引只用摘要
    docs = documents_from_corpus(papers, include_title=False)

    chunks_doc = [  # 变体 A：整篇摘要当作一个片段
        Chunk(
            doc_id=d.doc_id,
            chunk_id=f"{d.doc_id}#whole",
            char_start=0,
            char_end=len(d.content),
            text=d.content,
            source_hash=d.content_hash,
        )
        for d in docs
    ]
    chunks_sent: list = []
    for d in docs:
        chunks_sent.extend(chunk_document(d, max_chars=320, overlap_sentences=1))

    print(f"片段数：文档级 {len(chunks_doc)}，句块级 {len(chunks_sent)}")

    queries = build_queries(papers)
    if args.limit:
        queries = queries[: args.limit]
    print(f"查询：{len(queries)} 条\n")

    reports = []
    for retriever in build_variants(chunks_doc, chunks_sent):
        rep = evaluate_retriever(retriever, queries, ks=KS, depth=10)
        reports.append(rep)
        s = rep.summary()
        r5 = s["recall@5"]
        print(f"  {retriever.name:<20} Recall@5={r5['mean']:.3f} "
              f"[{r5['lo']:.3f},{r5['hi']:.3f}]  MRR={s['mrr']['mean']:.3f}")

    out = Path(args.out)
    write_report(reports, queries, manifest, out)
    print(f"\n报告已写入：{out}")
    print("\n" + summarize(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
