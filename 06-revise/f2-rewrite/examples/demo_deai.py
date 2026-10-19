#!/usr/bin/env python3
"""演示：E5 降AI —— 从外部 AI 检测报告到风格改写。

    python e5-deai/examples/demo_deai.py

⚠️ **改写候选是人手写的**（``CannedRewriter``），真实使用时由模型生成。
演示的是**整条链路**在拿到好候选时的表现，不代表模型能力。

演示五件事：

1. **诊断** —— 说清"AI 味在哪"，带具体数字
2. **好改写 → 采纳**，并给出特征前后对比
3. **编造数字 → 保真拦下**（风格再好看也不用）
4. **风格正常 → 不改**（不为"降 AI"硬改好文字）
5. 报告**不做身份判断**

输出：`reports/deai_report.md`
"""

from __future__ import annotations

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
STAGE_DIR = SYSTEM_DIR / "06-revise"
# 两种 src 布局：00-core/src（共享层）与 <阶段>/<节点>/src
for _src in sorted(SYSTEM_DIR.glob("*/*/src")) + sorted(SYSTEM_DIR.glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from copilot.quality.deai import (  # noqa: E402
    DeaiStatus,
    diagnose,
    profile,
    run_deai,
)
from copilot.quality.deai.report import STATUS_LABEL  # noqa: E402
from copilot.quality.report_parse import FlagType, build_checklist  # noqa: E402
from copilot.quality.rewrite import CannedRewriter, StubRewriter  # noqa: E402

OUT = STAGE_DIR / "reports" / "deai_report.md"

#: 典型的"机器味"片段：句式整齐 + 套话连接词多 + 无具体信息
AI_LIKE = "首先，该方法是有效的。其次，该方法具有优势。此外，该方法易于实现。最后，该方法值得推广。"

#: 风格正常的片段（不该被改）
NORMAL = "我们在两个真实业务场景里跑过，准确率比原来的规则方案高。维护成本也低了不少。核心逻辑不到一百行。"

#: ★ 人手写的"好改写"：长短句交错、无套话连接词
GOOD = (
    "这个方法管用。我们在两个真实业务场景里跑过，准确率明显高于原来的规则方案，"
    "维护成本也低了不少。核心逻辑不到一百行，团队里两个人两天就能接手。"
)

#: 假改写：**编造了原文没有的数字**
FAKE = "该方法有效。在 37 个数据集上准确率达到 94.2%。实现简单。"

THESIS = f"""# 方法

{AI_LIKE}

# 讨论

{NORMAL}

# 结论

本文的方法取得了预期效果。
"""

REPORT = f"{AI_LIKE}\n{NORMAL}\n"


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_named(title: str, text: str) -> None:
    p = profile(text)
    print(f"  {title}")
    print(f"    句长变异系数 {p.sentence_length_cv:.2f}  "
          f"连接词密度 {p.transition_density:.2f}  "
          f"具体性 {p.specificity:.2f}  排比 {p.parallelism:.2f}")


def main() -> int:
    rule("① 风格特征：把「AI 味」变成可计算的数字")
    show_named("机器味片段", AI_LIKE)
    show_named("正常片段  ", NORMAL)
    print()
    print("  → 四个特征都是**直接可复核**的客观量，不是身份推断。")

    rule("② 诊断：说清「AI 味在哪」（带具体数字）")
    for d in diagnose(AI_LIKE):
        print(f"  · {d.label}")
        print(f"      {d.explanation}")
        print(f"      建议：{d.advice}")
    print()
    print("  → 诊断出来「病因」才能对症下药；否则模型只会做同义词替换。")

    rule("③ 从外部 AI 检测报告得到清单（复用 E3）")
    checklist = build_checklist(THESIS, REPORT, flag_type=FlagType.AI)
    print(f"  {checklist.summary_line()}")

    rule("④ 好改写 → 采纳（含特征前后对比）")
    good_rep = run_deai(checklist, rewriter=CannedRewriter({AI_LIKE: [GOOD]}))
    for s in good_rep.suggestions:
        if s.status != DeaiStatus.ACCEPTED:
            continue
        print(f"  原文：{s.unit.text[:60]}")
        print(f"  改写：{s.chosen[:60]}")
        print(f"  诊断：{'；'.join(d.label for d in s.diagnoses)}")
        print(f"  策略：{'、'.join(st.label for st in s.instruction.strategies)}")
        print("  特征变化：")
        if s.verification:
            for ch in s.verification.changes:
                print(f"    {ch.describe()}")
            if s.verification.fidelity:
                print(f"  保真：{s.verification.fidelity.summary()}")
        print()

    rule("⑤ 编造数字 → 保真拦下（风格再好看也不用）")
    fake_rep = run_deai(checklist, rewriter=CannedRewriter({AI_LIKE: [FAKE]}))
    for s in fake_rep.suggestions:
        if s.status == DeaiStatus.REJECTED_FIDELITY:
            print(f"  候选：{FAKE}")
            print(f"  状态：{STATUS_LABEL[s.status]}")
            print(f"  说明：{s.note}")
            print()
            print("  → 「补充具体信息」只能用手上**已有**的信息。")
            print("     编造数据的危害远大于 AI 味 —— 一段风格差的内容只是风格问题，")
            print("     一段编造的实验数据是学术不端。")

    rule("⑥ 风格正常 → 不改（不为「降 AI」硬改好文字）")
    stub_rep = run_deai(checklist, rewriter=StubRewriter())
    for s in stub_rep.suggestions:
        if s.status == DeaiStatus.NO_DIAGNOSIS:
            print(f"  {s.unit.location}：{s.unit.text[:52]}")
            print(f"  说明：{s.note}")

    rule("⑦ 报告")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(good_rep.to_markdown(), encoding="utf-8")
    print(f"  已写入：{OUT}")
    print(f"  {good_rep.summary_line()}")

    rule("要点")
    print(
        "  1. 本模块**不做 AI 检测**，只处理外部报告指出的片段 + 可测特征；\n"
        "  2. **先诊断再改写** —— 「改得像人一点」不是可执行指令；\n"
        "  3. **保真是否决条件**：改动原意就弃用，哪怕风格改善明显；\n"
        "  4. 风格特征跨多句才成立 —— 所以 E5 用**段落级**单元，不是单句；\n"
        "  5. 风格正常就别改；报告也**不做身份判断**。"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
