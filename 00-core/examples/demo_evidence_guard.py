"""演示：证据绑定层全流水线 —— 从「模型生成的答案」到「带证据与状态的结构化结果」。

运行：
    python examples/demo_evidence_guard.py

模拟真实场景：模型基于检索到的文献生成了一段答案，我们要逐条核实它。

流水线：

    模型生成的答案 + 检索到的文档
        ↓  ① 原子化     拆成可独立验证的断言（extract）
        ↓  ② 自动绑定   为每条断言找出原文片段（bind）
        ↓  ③ 确定性校验  引用真实性 + 片段三重一致（verify）
        ↓  ④ 支持判断    一致 / 部分 / 缺失 / 矛盾（support）
    带证据、可回溯、可告警的结构化结果

其中 ②③④ 都不需要调用模型 —— 这正是本项目的核心论点：
**防幻觉的很大一部分可以做成确定性的。**
"""

from __future__ import annotations

import sys
from pathlib import Path

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

from copilot.evidence.bind import RuleBinder, bind_evidence
from copilot.evidence.extract import RuleBasedExtractor
from copilot.evidence.models import (
    CitationMeta,
    Document,
    InMemoryDocStore,
    StaticDoiResolver,
    SupportStatus,
)
from copilot.evidence.support import RuleBasedJudge
from copilot.evidence.verify import (
    InMemoryCitationLibrary,
    VerdictKind,
    verify_claim,
)

REAL_DOI = "10.1234/real.2024.001"

# ── 检索到的文献 ──────────────────────────────────────────────
DOC_REAL = Document.create(
    "doc-real",
    "本文提出了一种新的检索方法。"
    "实验表明，该方法在五个数据集上准确率提升了 5%。"
    "然而在低资源场景下，性能下降约 12%，这是本方法的主要局限。",
)
DOC_OTHER = Document.create(
    "doc-other",
    "本文研究了电商推荐场景下的点击率预估问题，与检索方法无关。",
)

# ── 模型生成的答案 ──
# 刻意包含四种情况，以展示四种不同裁决：
#   ① 有据可依            → SUPPORTED
#   ② 方向写反（与原文矛盾）→ FABRICATED
#   ③ 编造引用            → FABRICATED
#   ④ 完全没有依据         → UNSUPPORTED（是"信息不足"，不是"编造"）
GENERATED = (
    "该方法在五个数据集上准确率提升了 5%。"
    "该方法提升了低资源场景下的性能。"
    "该方法在医疗问诊场景同样有效。"
    "该结论具有重要的理论意义。"
)

STATUS_MARK = {
    SupportStatus.SUPPORTED: "✅ 支持",
    SupportStatus.PARTIALLY_SUPPORTED: "🟡 部分支持",
    SupportStatus.UNSUPPORTED: "⚪ 证据不足",
    SupportStatus.FABRICATED: "❌ 编造/矛盾",
}


def rule(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main() -> None:
    docs = [DOC_REAL, DOC_OTHER]
    store = InMemoryDocStore(docs)
    library = InMemoryCitationLibrary(
        [
            CitationMeta(
                doi=REAL_DOI,
                title="A New Retrieval Method",
                authors=("Zhang, S.",),
                year=2024,
                venue="ACL",
            )
        ]
    )
    resolver = StaticDoiResolver({REAL_DOI})

    print("检索到的文献：")
    for d in docs:
        print(f"  [{d.doc_id}] {d.content}")
    print(f"\n模型生成的答案：\n  {GENERATED}")

    # ── ① 原子化 ──
    rule("① 原子化：把答案拆成可独立验证的断言")
    extractor = RuleBasedExtractor()
    claims = extractor.extract(GENERATED)
    for c in claims:
        print(f"  [{c.claim_id}] {c.text}")
    print(
        f"\n  切分统计：句子 {extractor.last_stats.sentences} / "
        f"子句 {extractor.last_stats.clauses} / 保留 {extractor.last_stats.kept}"
    )

    # 给"医疗问诊"那条挂一个编造的引用，模拟模型编文献。
    # 刻意按内容定位而非按索引，避免语句顺序变化后挂错对象。
    for c in claims:
        if "医疗" in c.text:
            c.citations.append(
                CitationMeta(
                    doi="10.9999/fake.2099.999",
                    title="Fabricated Study on Medical QA",
                )
            )

    # ── ② 自动绑定 ──
    rule("② 自动绑定：为每条断言在文献中找出原文片段（不需要模型）")
    binder = RuleBinder(min_relevance=0.3, max_spans=1)
    for c in claims:
        result = bind_evidence(c, docs, binder=binder)
        if c.has_evidence:
            span = c.evidence[0]
            print(f"\n  [{c.claim_id}] {c.text}")
            print(f"      绑定 → [{span.doc_id}] {span.text}")
            print(f"      偏移 [{span.char_start}:{span.char_end}]，"
                  f"hash {span.source_hash[:12]}…")
        else:
            print(f"\n  [{c.claim_id}] {c.text}")
            print(f"      未绑定（{result.rejection_reason}）")

    # ── ③④ 校验 + 支持判断 ──
    rule("③④ 校验与支持判断：这条链路完全不依赖模型")
    judge = RuleBasedJudge()
    final: list[tuple[str, SupportStatus]] = []

    for c in claims:
        det = verify_claim(c, store, library, resolver)
        print(f"\n  [{c.claim_id}] {c.text}")

        if det.kind is VerdictKind.FABRICATED:
            print(f"      确定性层 → 编造")
            for r in det.reasons:
                print(f"        · {r}")
            final.append((c.text, SupportStatus.FABRICATED))
            continue

        if det.kind is VerdictKind.INSUFFICIENT:
            print(f"      确定性层 → 证据不足")
        else:
            print(f"      确定性层 → 通过（交给支持判断）")

        judgment = judge.judge(c)
        print(f"      支持判断 → {STATUS_MARK[judgment.status]}"
              f"   覆盖率 {judgment.signals.coverage}")
        if judgment.signals.polarity_conflicts:
            pairs = "、".join(f"{a}↔{b}" for a, b in judgment.signals.polarity_conflicts)
            print(f"        · 极性冲突：{pairs}（结论与原文相反）")
        if judgment.signals.missing_numbers:
            print(f"        · 缺失数字：{judgment.signals.missing_numbers}")
        for r in judgment.reasons:
            print(f"        · {r}")
        final.append((c.text, judgment.status))

    # ── 汇总 ──
    rule("最终裁决")
    for text, status in final:
        print(f"  {STATUS_MARK[status]:<12} {text}")

    blocked = [s for _, s in final if s is SupportStatus.FABRICATED]
    print(
        f"\n  共 {len(final)} 条断言：{len(blocked)} 条被判为编造/矛盾，已拦截。\n"
        "\n  要点：\n"
        "    · ②③④ 全程未调用模型 —— 极性矛盾、数字缺失、伪造引用都是规则可判的；\n"
        "    · 「证据不足」与「编造」被严格区分：前者提示补检索，后者告警；\n"
        "    · 规则做不到的同义改写与跨句推理，才是模型的价值所在（P9 消融量化）。"
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
