#!/usr/bin/env python3
"""演示：E6 审稿挑错。

    python e6-review/examples/demo_review.py

演示 E6 能检出、而且**人工很难核对出来**的几类问题：

1. **术语不一致** —— 同一概念用了两种叫法（"召回率" / "查全率"）
2. **关键词脱节** —— 摘要把某词列为关键词，正文一次都没提
3. **章节过短** —— 一级章节论述不充分
4. **图表引用断裂** —— 复用 E1 的规则

并明确写出 E6 **不做**什么（论证逻辑不在其中）。

输出：`reports/review_report.md`
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
STAGE_DIR = SYSTEM_DIR / "05-format"
# 两种 src 布局：00-core/src（共享层）与 <阶段>/<节点>/src
for _src in sorted(SYSTEM_DIR.glob("*/*/src")) + sorted(SYSTEM_DIR.glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from copilot.quality.format.template import Template  # noqa: E402
from copilot.quality.review import review_text  # noqa: E402

OUT = STAGE_DIR / "reports" / "review_report.md"

THESIS = """# 摘要

本文研究了一种面向学术检索的增强方法，在多个公开数据集上完成了验证。

**关键词**：检索增强；鲁棒性；推荐系统

# 绪论

检索增强生成在近年受到广泛关注。已有工作讨论了召回率对最终效果的影响。
如 图 5 所示，该方法在多个场景下表现稳定。

# 方法

我们采用两阶段流程：先做粗筛，再做精排。粗筛用词面信号召回候选，
精排用语义信号重新排序，两阶段之间用一个小型重排模块连接。
整个流程的查全率在实验中得到了验证。该流程在推荐系统的排序任务上也做了初步尝试。

# 结论

有效。

# 参考文献

[1] 张三. 一种检索增强方法[J]. 计算机学报, 2020, 43(2): 1-10.
"""


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    tpl = Template(
        name="演示模板",
        required_sections=["摘要", "绪论", "方法", "结论", "参考文献"],
        min_body_words=None,
    )

    rule("① 运行审稿检查")
    report = review_text(THESIS, tpl)
    print(f"  {report.summary_line()}")

    rule("② 检出结果")
    for i in report.issues:
        print(f"  {i.severity.label}  [{i.code}]  {i.location}")
        print(f"    问题：{i.message}")
        print(f"    建议：{i.fix}")
        print()

    rule("③ 这些检查为什么值得自动化")
    print(
        "  · **术语不一致**：全文搜一遍才知道，人工读到后面早就忘了前面怎么写的；\n"
        "  · **关键词脱节**：摘要与正文分开写，两者对不上自己很难发现；\n"
        "  · **图表引用断裂**：增删图表后最容易漏改，需要通读全文；\n"
        "  · **章节过短**：字数统计一下就知道，但很少有人会去数。"
    )

    rule("④ 复用：E6 与 E1 共享同一套规则")
    print(
        "  图表引用检查（FIG002）直接来自 E1 —— **同一能力只有一份实现**。\n"
        "  Issue / Severity 词汇也共用，所以两份报告可以直接合并成一份质检报告。"
    )

    rule("⑤ E6 不做的事")
    print(
        "  「论证链是否断裂」**不在 E6 的检查范围** —— 那需要跨句语义理解，属模型范畴。\n"
        "  一份声称「已检查论证逻辑」实则只查了术语的报告，会让用户误以为可以放心提交。"
    )

    rule("⑥ 报告")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report.to_markdown(), encoding="utf-8")
    print(f"  已写入：{OUT}")

    rule("要点")
    print(
        "  1. E6 关注**结构与内容一致性**，不是格式（那是 E1）；\n"
        "  2. 复用 E1 的文档模型、Issue 词汇、图表规则 —— 判定逻辑只有一份；\n"
        "  3. 只做可判定的部分，**并写明没做什么**；\n"
        "  4. 「挑错」不等于「评价」：指可验证的问题，不给主观评分。"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
