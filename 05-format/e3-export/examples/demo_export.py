#!/usr/bin/env python3
"""演示：F1 文档导出。

    python f1-export/examples/demo_export.py

演示五件事：

1. 参考文献解析（E1 的 ``Reference`` → 更全的 ``BibEntry``）
2. **BibTeX 生成 + LaTeX 转义**（标题含 ``&`` ``%`` ``_`` 时的正确处理）
3. GB/T 7714 生成
4. LaTeX 骨架生成（只生成结构，不生成正文）
5. **回读一致性校验**（含"人为删掉一章"的对照）

产物：`exports/thesis.tex` · `exports/refs.bib` · `exports/references_gbt7714.txt`
报告：`reports/export_report.md`
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

from copilot.delivery.export import (  # noqa: E402
    escape_latex,
    export_paper,
    to_bibtex,
    verify_latex,
)
from copilot.quality.format import parse_markdown  # noqa: E402

EXPORT_DIR = STAGE_DIR / "exports"
REPORT_DIR = STAGE_DIR / "reports"

THESIS = """# 基于图神经网络的推荐方法研究

## 摘要

本文研究了一种检索增强与图神经网络结合的方法。

## 绪论

检索增强生成在近年受到广泛关注，整体框架如图 1 所示。

**图 1** 研究框架

## 方法

我们采用两阶段流程，参数见表 1。

**表 1** 参数设置

## 结论

本文的方法取得了预期效果。

## 参考文献

[1] 张三, 李四. 一种基于图神经网络的方法[J]. 计算机学报, 2020, 43(2): 1-10.
[2] 王五. 深度学习与 R&D 实践[M]. 北京: 科学出版社, 2021.
[3] 赵六. 检索增强综述[J]. 软件学报, 2019, 30(5): 100-120.
[4] 钱七. 面向长尾查询的检索优化[D]. 北京: 清华大学, 2022.
"""


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    doc = parse_markdown(THESIS)

    rule("① 参考文献解析")
    result = export_paper(doc)
    for e in result.entries:
        print(f"  {e.key:<22} {e.entry_type:<14} {e.cn_type:<4} {e.title[:30]}")
    print(f"\n  共 {len(result.entries)} 条，引用键互不重复 ✓")

    rule("② ★ LaTeX 转义：为什么必须做")
    tricky = "R&D: 50% 提升_a_b"
    print(f"  原始文本：{tricky}")
    print(f"  转义之后：{escape_latex(tricky)}")
    print()
    print("  不转义会怎样：标题里一个 & 就让整个 .bib 编译失败，")
    print("  而 LaTeX 的报错指向生成后几百行的 .tex —— 极难定位源头。")

    rule("③ BibTeX 生成（转义已生效）")
    for e in result.entries[:2]:
        print(to_bibtex(e))
        print()

    rule("④ GB/T 7714 生成")
    for line in result.references_gbt7714.splitlines():
        print(f"  {line}")

    rule("⑤ LaTeX 骨架（只生成结构，不生成正文）")
    for line in result.tex.splitlines():
        if line.startswith(("\\section", "\\subsection", "\\begin{figure", "\\begin{table", "\\caption", "% TODO")):
            print(f"  {line}")

    rule("⑥ ★ 回读一致性校验")
    print(f"  {result.latex_check.summary()}")
    print(f"  {result.bib_check.summary()}")
    print()
    print("  对照实验：人为删掉一章，看校验能不能发现 ——")
    broken = result.tex.replace("\\section{方法}", "")
    check = verify_latex(broken, doc, document_title="基于图神经网络的推荐方法研究")
    print(f"  {check.summary()}")
    print()
    print("  → 这正是「回读反查」的价值：肉眼看 .tex 根本不会去数 section 数量。")

    rule("⑦ 写出产物")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPORT_DIR / "thesis.tex").write_text(result.tex, encoding="utf-8")
    (EXPORT_DIR / "refs.bib").write_text(result.bibtex, encoding="utf-8")
    (EXPORT_DIR / "references_gbt7714.txt").write_text(
        result.references_gbt7714, encoding="utf-8"
    )
    for name in ("thesis.tex", "refs.bib", "references_gbt7714.txt"):
        p = EXPORT_DIR / name
        print(f"  {p.relative_to(SYSTEM_DIR)}  ({p.stat().st_size} 字节)")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = [
        "# 文档导出报告",
        "",
        f"- 章节：{len(doc.flat_sections())} 个",
        f"- 图表：{len(doc.captions)} 个",
        f"- 参考文献：{len(result.entries)} 条",
        "",
        "## 一致性校验",
        "",
        f"- 结构：{result.latex_check.summary()}",
        f"- 文献：{result.bib_check.summary()}",
        "",
        "## 引用键",
        "",
        "| 键 | 类型 | 题名 |",
        "|---|---|---|",
    ]
    report += [f"| `{e.key}` | {e.entry_type} | {e.title[:40]} |" for e in result.entries]
    report += ["", "## GB/T 7714 参考文献表", "", "```", result.references_gbt7714, "```", ""]
    (REPORT_DIR / "export_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"  {REPORT_DIR.joinpath('export_report.md').relative_to(SYSTEM_DIR)}")

    rule("要点")
    print(
        "  1. **零依赖**：纯文本生成，不需要 python-docx / LibreOffice；\n"
        "  2. **转义放在生成器里** —— 属于「生成 BibTeX」的语义，不属于「写文件」；\n"
        "  3. 转义必须**单遍字符映射**：顺序 replace 会命中自己产生的文本；\n"
        "  4. **导出必须回读校验** —— 「我相信我写对了」不是可验证。"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
