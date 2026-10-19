#!/usr/bin/env python3
"""演示：E4 降重 —— 从外部查重报告到改写建议。

    python e4-rewrite/examples/demo_rewrite.py

⚠️ **重要说明**：本演示里的"改写候选"是**人手写的**（``CannedRewriter``）。
真实使用时由模型生成。这样做的目的是演示**整条链路**（策略 → 度量 → 保真 → 报告）
在拿到好候选时会怎样，**不代表模型能力**。

演示四件事：

1. 好改写 → **被采纳**，报告给出效果数字（连续重叠 / n-gram 包含率）
2. 改写丢了数字 → **保真拦下**，即使降重效果很好
3. 引用片段 → **不改**（改了就是篡改他人原文）
4. 定义类 → **建议改为引用**，而不是改写

输出：`reports/rewrite_report.md`
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

from copilot.quality.report_parse import FlagType, build_checklist  # noqa: E402
from copilot.quality.rewrite import (  # noqa: E402
    CannedRewriter,
    StubRewriter,
    SuggestionStatus,
    check_fidelity,
    measure_effect,
    run_rewrite,
)
from copilot.quality.rewrite.report import STATUS_LABEL  # noqa: E402

OUT = STAGE_DIR / "reports" / "rewrite_report.md"

# ── 用户文档（出报告后没改过，便于演示精确匹配）──
DUP = "本文提出了一种基于图神经网络的推荐方法，并在三个数据集上验证了效果。"
QUOTED = "本文提出了一种基于图神经网络的推荐方法并在三个数据集上验证了效果"
DEFINITION = "精确率定义为正确预测的正样本数除以所有预测为正的样本数。"
NUMERIC = "实验结果显示准确率提升了 8.3%，召回率提升了 12.5%。"

THESIS = f"""# 绪论

学术文献的数量在过去二十年里呈现指数级增长。

# 相关工作

{DUP}

如文献所述「{QUOTED}」，可见该方法有一定效果。

# 方法

{DEFINITION}

{NUMERIC}

我们设计了一种双阶段的检索策略。

# 结论

本文的方法取得了预期效果。
"""

REPORT = f"""{DUP}
{QUOTED}
{DEFINITION}
{NUMERIC}
"""

#: ★ 人手写的改写候选（演示用，不是模型产出）
GOOD_REWRITE = "我们的方案借助图结构对用户与物品的关联建模，在三个公开数据集上完成了评估。"
BAD_REWRITE = "有提升，也在若干数据集上做过验证。"      # 把数字全删了


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_report(rep) -> None:
    for s in rep.suggestions:
        print(f"  {STATUS_LABEL.get(s.status, s.status):<14} {s.unit.location:<10} "
              f"{s.kind.label:<8} {s.strategy.value}")
        print(f"    原文：{s.unit.text[:56]}")
        if s.status == SuggestionStatus.ACCEPTED:
            print(f"    改写：{s.chosen[:56]}")
            if s.effect:
                print(f"    效果：最长连续重叠 {s.effect.after_run} 字 / "
                      f"n-gram 包含率 {s.effect.after_containment:.0%}"
                      f"（{'达标' if s.effect.meets_target else '未达标'}）")
            if s.fidelity:
                print(f"    保真：{s.fidelity.summary()}")
        elif s.note:
            print(f"    说明：{s.note[:76]}")
        print()


def main() -> int:
    rule("① 从外部查重报告得到问题清单（复用 E3）")
    checklist = build_checklist(THESIS, REPORT, flag_type=FlagType.DUPLICATE)
    print(f"  {checklist.summary_line()}")

    rule("② 好改写 → 被采纳（且给出效果数字）")
    rep = run_rewrite(checklist, rewriter=CannedRewriter({DUP: [GOOD_REWRITE]}))
    show_report(rep)

    rule("③ 改写丢了数字 → 保真拦下（降重效果再好也不用）")
    bad = run_rewrite(checklist, rewriter=CannedRewriter({NUMERIC: [BAD_REWRITE]}))
    for s in bad.suggestions:
        if s.unit.text.startswith("实验结果显示"):
            print(f"  原文：{s.unit.text}")
            print(f"  候选：{BAD_REWRITE}")
            print(f"  状态：{STATUS_LABEL.get(s.status)}")
            print(f"  说明：{s.note}")
            print()
            print("  → 降重效果看起来很好（数字全删了，当然不重合），")
            print("     但**数字是事实，不能因为改写而消失** —— 所以弃用。")

    rule("④ 保真校验的四个具体检查")
    bad_fid = check_fidelity(BAD_REWRITE, NUMERIC)
    for issue in bad_fid.issues:
        print(f"  ❌ {issue}")
    good_fid = check_fidelity(GOOD_REWRITE, DUP)
    print(f"\n  好改写：{good_fid.summary()}")

    rule("⑤ 没有模型时的表现（StubRewriter）")
    stub = run_rewrite(checklist, rewriter=StubRewriter())
    print(f"  {stub.summary_line()}")
    print("  → StubRewriter 原样返回，效果度量显示「没改善」，")
    print("     于是走「未达标」分支并如实报告 —— **这正好把失败路径也测到了**。")

    rule("⑥ 报告")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(rep.to_markdown(), encoding="utf-8")
    print(f"  已写入：{OUT}")
    print(f"  {rep.summary_line()}")

    rule("要点")
    print(
        "  1. 改写必须用模型 —— 这是 E 阶段唯一不能确定性实现的一环；\n"
        "  2. **保真是否决条件**：不保真的候选即使降重效果再好也弃用；\n"
        "  3. 效果要**两个指标**：只看连续重叠会被「拆短句」骗过；\n"
        "  4. 引用不改、定义建议引用、数据不改 —— 准确性优先于降重；\n"
        "  5. 改不动就如实说「需人工改写」，不假装成功。"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
