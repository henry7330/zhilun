#!/usr/bin/env python3
"""演示：E3 报告解析与定位。

    python e3-report/examples/demo_report_parse.py

演示四件事（对应四条验收标准）：

1. **精确定位** —— 报告片段与文档一致
2. **模糊定位** —— 文档在出报告之后被改过（★ 最真实的场景）
3. **未定位** —— 完全对不上时如实报告，**不硬指位置**
4. **多格式** —— 纯文本 / JSON / CSV 都能吃

输出：`reports/report_parse.md`
"""

from __future__ import annotations

import io
import json
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

from copilot.quality.report_parse import (  # noqa: E402
    FlagType,
    LocateMethod,
    available_formats,
    build_checklist,
    parse_report,
)

OUT = STAGE_DIR / "reports" / "report_parse.md"

#: 用户文档的**当前**内容（出报告之后又改过）
THESIS = """# 绪论

学术文献的数量在过去二十年里呈现指数级增长，研究者需要在海量文献中快速定位相关工作。

# 相关工作

本文提出了一种基于异构图神经网络的推荐方法，并在三个数据集上验证了效果。

相关工作还讨论了冷启动问题，并提出了一种基于元学习的缓解策略。

# 方法

我们设计了一种双阶段的检索策略：第一阶段用词面信号快速筛出候选。

# 结论

本文的方法取得了预期效果。
"""

#: 外部查重报告给出的片段（来自**修改前**的版本）
REPORT_PLAIN = """本文提出了一种基于图神经网络的推荐方法，并在三个数据集上验证了效果。
相关工作还讨论了冷启动问题，并提出了一种基于元学习的缓解策略。
本研究首次提出了一个全新的量子推荐框架并彻底解决了所有已知问题。
"""


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show(checklist) -> None:
    for it in checklist.items:
        mark = {
            LocateMethod.EXACT: "✅ 精确",
            LocateMethod.FUZZY: "🟡 模糊",
            LocateMethod.UNLOCATED: "❌ 未定位",
        }[it.method]
        print(f"  {mark:<10} {it.location:<12} 可信度 {it.confidence:.2f}")
        print(f"            报告片段：{it.item.text[:52]}")
        if it.doc_text:
            print(f"            文档实际：{it.doc_text[:52]}")
        if it.note:
            print(f"            说明：{it.note}")
        print()


def main() -> int:
    rule("① 支持的报告格式")
    print(f"  {', '.join(available_formats())}")
    print("  优先保证「直接粘贴片段列表」这条路径可靠 —— 它零歧义。")

    rule("② 纯文本报告 → 三级定位")
    checklist = build_checklist(THESIS, REPORT_PLAIN, flag_type=FlagType.DUPLICATE)
    print(f"  {checklist.summary_line()}\n")
    show(checklist)

    rule("③ 同一份报告，JSON 格式（带来源与分数）")
    payload = json.dumps(
        [
            {
                "text": "本文提出了一种基于图神经网络的推荐方法，并在三个数据集上验证了效果。",
                "score": 0.87,
                "source_title": "A Graph-based Recommendation Method, 2019",
                "source_text": "We propose a graph neural network based recommendation method...",
            }
        ],
        ensure_ascii=False,
        indent=2,
    )
    items, fmt = parse_report(payload, flag_type=FlagType.DUPLICATE)
    print(f"  解析格式：{fmt}，{len(items)} 条")
    print(f"  来源文献：{items[0].source_title}")
    print(f"  报告给了来源原文：{items[0].has_source_text}")
    print("  → 有来源原文时，E4 降重可以**直接度量**降重效果（而不是代理指标）")

    rule("④ 按位置归并（下游对同一段只改一次）")
    duplicate_report = "本文提出了一种基于图神经网络的推荐方法，并在三个数据集上验证了效果。"
    groups = build_checklist(THESIS, duplicate_report + "\n" + duplicate_report).by_location()
    for pos, group in groups:
        print(f"  位置 {pos} → {len(group)} 条报告条目合并为一组")

    rule("⑤ 报告")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(checklist.to_markdown(), encoding="utf-8")
    print(f"  已写入：{OUT}")

    rule("要点")
    print(
        "  1. 本模块**不做检测** —— 报告来自外部工具，这里只负责读懂与定位；\n"
        "  2. 真正的难点是「报告来自旧版本文档」：用户出完报告往往又改过稿；\n"
        "  3. 三级定位：精确（去空白后逐字相同）→ 模糊（最长公共子串）→ 未定位；\n"
        "  4. **未定位绝不硬指位置** —— 静默丢弃会漏改，硬指位置会让用户改错地方；\n"
        "  5. 去空白是为了比对，但**偏移必须映射回原文**，否则定位静默错位。"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
