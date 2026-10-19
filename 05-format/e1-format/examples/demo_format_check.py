#!/usr/bin/env python3
"""演示：E1 格式校验 Agent 跑一份真实草稿。

    python examples/demo_format_check.py

样例草稿里**故意埋了 7 类问题**，覆盖三条规则各自的典型场景。
脚本会把结果打印出来，并写出 Markdown 报告。
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

from copilot.quality.format import Severity, builtin_templates, check_format  # noqa: E402

# 评测数据与报告放在**阶段根**（跨模块共用），不在单个模块内
DRAFT = STAGE_DIR / "eval" / "dataset" / "sample_thesis_draft.md"
OUT = STAGE_DIR / "reports" / "format_report.md"


def rule(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main() -> int:
    templates = builtin_templates()
    tpl = templates["undergraduate"]

    rule("输入")
    print(f"  草稿    : {DRAFT.name}")
    print(f"  模板    : {tpl.name}（参考文献风格 {tpl.reference_style}）")

    report = check_format(DRAFT, tpl)

    rule("文档模型解析结果（子部分 ①）")
    doc = report.doc
    print(f"  章节      : {len(doc.flat_sections())} 个")
    for s in doc.flat_sections()[:8]:
        indent = "  " * (s.level - 1)
        print(f"    {indent}- {s.title}  （第 {s.line + 1} 行，{s.level} 级）")
    if len(doc.flat_sections()) > 8:
        print(f"    … 其余 {len(doc.flat_sections()) - 8} 个")
    print(f"  参考文献  : {len(doc.references)} 条")
    print(f"  图表题注  : {len(doc.captions)} 个")
    print(f"  正文引用  : {len(doc.cross_refs)} 处")
    print(f"  正文字数  : {doc.body_words} 字")

    rule("校验结果（子部分 ③④⑤ → ⑥）")
    c = report.counts()
    print(f"  {report.summary_line()}\n")

    for severity in (Severity.ERROR, Severity.WARNING, Severity.INFO):
        items = report.by_severity(severity)
        if not items:
            continue
        print(f"  {severity.label}（{len(items)} 项）")
        for i in items:
            print(f"    [{i.code}] {i.location}  {i.message}")
            if i.fix:
                print(f"           → {i.fix}")
        print()

    rule("规则码分布")
    counts: dict[str, int] = {}
    for i in report.issues:
        counts[i.code] = counts.get(i.code, 0) + 1
    for code in sorted(counts):
        print(f"  {code}  {counts[code]} 处")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report.to_markdown(), encoding="utf-8")
    print(f"\n报告已写入：{OUT}")

    print("\n" + "=" * 74)
    print("要点")
    print("=" * 74)
    print(
        "  1. 三条规则全部只读「文档模型」，不碰原始 Markdown ——\n"
        "     将来支持 DOCX 时，规则一行不用改（子部分 ① 是防腐层）；\n"
        "  2. 合格标准全部来自模板配置，不在代码里硬编码学校要求（子部分 ②）；\n"
        "  3. 「引用了不存在的图 4」这类全局一致性问题，人工要通读全文才能发现；\n"
        "  4. 报告分三级 —— 一堆同权重告警等于没有告警（子部分 ⑥）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
