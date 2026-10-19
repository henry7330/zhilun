#!/usr/bin/env python3
"""演示：F2 提交前检查。

    python f2-prcheck/examples/demo_prcheck.py

演示五件事：

1. **占位符残留**（TODO / 待补 / 空占位）—— 最高频、最低级的交付事故
2. **敏感信息**（邮箱 / 手机号 / 绝对路径 / 密钥）—— 报告里已脱敏
3. **匿名性**（致谢非空 / 基金号 / 机构名）—— 双盲投稿的真实事故
4. **复用 E1 的规则但重新定级**（空章节：E1 的 INFO → F2 的阻断项）
5. **明确的门禁结论**（能不能提交）

报告：`reports/prcheck_report.md`
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
MODULE_DIR = _HERE.parent.parent
STAGE_DIR = MODULE_DIR.parent
SYSTEM_DIR = STAGE_DIR.parent
for _src in sorted(SYSTEM_DIR.glob("*/*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from copilot.delivery.prcheck import check_submission  # noqa: E402
from copilot.quality.format import check_format, parse_markdown  # noqa: E402
from copilot.quality.format.template import Template  # noqa: E402

REPORT_DIR = STAGE_DIR / "reports"

DIRTY = """# 一篇还有问题的论文

## 摘要

本文研究了一种方法。

## 绪论

TODO: 这里要补充研究背景。

联系方式：zhangsan@example.com，手机 13800138000。
实验数据放在 C:\\Users\\zhang\\data\\results.csv。

## 方法

我们采用两阶段流程。

## 结论

（待补）

## 致谢

感谢清华大学李教授的指导。本工作受国家自然科学基金（No. 12345678）资助。

## 参考文献

[1] 张三. 一种方法[J]. 计算机学报, 2020, 43(2): 1-10.
"""

CLEAN = """# 一篇可以提交的论文

## 摘要

本文研究了一种用于文献检索的增强方法，并在三个公开数据集上完成了验证。

## 绪论

检索增强生成在近年受到广泛关注。已有工作讨论了检索质量对最终效果的影响，
本文在此基础上进一步分析了长尾查询上的表现差异。

## 方法

我们采用两阶段流程：先做粗筛，再做精排。粗筛用词面信号召回候选，
精排用语义信号重新排序，两阶段之间用一个小型重排模块连接。

## 结论

本文的方法在多个数据集上取得了稳定效果，在低资源场景下仍有提升空间。

## 参考文献

[1] 张三. 一种方法[J]. 计算机学报, 2020, 43(2): 1-10.
"""

LENIENT = Template(
    name="宽松模板（演示用）",
    required_sections=["摘要", "结论", "参考文献"],
    min_body_words=None,
)


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    rule("① 有问题的稿子 → 逐项检出")
    report = check_submission(DIRTY, anonymity_required=True, template=LENIENT)
    print(f"  {report.summary_line()}\n")
    for issue in report.issues:
        mark = {"error": "🚫", "warning": "⚠️", "info": "ℹ️"}[issue.severity.value]
        print(f"  {mark} [{issue.code}] {issue.location}")
        print(f"      {issue.message}")
    print()
    print("  → 注意敏感信息已**脱敏**（报告本身也不能泄露）。")

    rule("② 双盲要求：致谢与基金号是真实事故源")
    print(
        "  「致谢」章节非空 → 直接暴露作者身份；\n"
        "  「国家自然科学基金（No. 12345678）」→ 一查就知道是谁。\n"
        "  这类问题作者往往完全没意识到，而它是双盲被拒/被警告的高频原因。"
    )

    rule("③ ★ 复用 E1 的规则，但重新定级")
    text_with_empty = (
        "# 论文\n\n## 摘要\n\n有内容。\n\n## 结论\n\n## 参考文献\n\n"
        "[1] 张三. 一种方法[J]. 计算机学报, 2020.\n"
    )
    e1 = check_format(text_with_empty, LENIENT)
    f2 = check_submission(text_with_empty, template=LENIENT)
    e1_empty = [i for i in e1.issues if i.code == "SEC005"]
    f2_empty = [i for i in f2.issues if i.code == "SEC005"]
    print(f"  E1 把空章节判为：{e1_empty[0].severity.value if e1_empty else '（未检出）'}")
    print(f"  F2 把它判为    ：{f2_empty[0].severity.value if f2_empty else '（未检出）'}")
    print()
    print("  → 同一次检测，两种判决：E1 管「格式合不合规」，F2 管「能不能交」。")
    print("     检测只做一遍，判决按用途各定各的。")

    rule("④ 合格稿子 → 可以提交")
    clean = check_submission(CLEAN, anonymity_required=False, template=LENIENT)
    print(f"  {clean.summary_line()}")
    hidden = [i for i in clean.issues if i.code.startswith("PRE")]
    print(f"  PRE 类问题：{len(hidden)} 个" + ("（零误报）" if not hidden else ""))

    rule("⑤ 默认模板是严格的（这是刻意设计）")
    strict = check_submission(CLEAN)
    print(f"  不传模板时：{strict.verdict()}")
    print()
    print("  默认模板是**本科论文的严格要求**（9 个必需章节 + 8000 字），")
    print("  所以一份 200 字的草稿会被判为不可提交 —— 这是**正确行为**，不是 bug。")

    rule("⑥ 报告")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "prcheck_report.md"
    out.write_text(report.to_markdown(), encoding="utf-8")
    print(f"  已写入：{out.relative_to(SYSTEM_DIR)}")

    rule("要点")
    print(
        "  1. F2 回答的是「能不能交」，给的是**明确的 yes/no**，不是一串问题；\n"
        "  2. 敏感信息与匿名性**完全可规则检测**，没有理由不做；\n"
        "  3. 报告本身要脱敏 —— 否则泄露面从正文扩大到报告；\n"
        "  4. 结构类问题**复用 E1 的检测**，只按「能不能交」重新定级。"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
