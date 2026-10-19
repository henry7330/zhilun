"""D3 图表规范：分析编号与引用，给出**具体的**修复建议。

## ★ 为什么只给建议、不自动改

听起来"自动重编号"很贴心，但它会**制造更严重的问题**：

```
原文：  图 3 展示了框架。        ← 正文引用
        **图 3** 框架图          ← 题注

假设前面少了一个图，程序把题注重编号成 图 2
→ 正文里的「图 3」失去了目标，引用断裂
```

**重编号题注而不动引用 = 把"编号不连续"换成"引用全是错的"。**
而引用分散在全文，自动改需要判断"这个数字指的是图还是页码/年份"，
判断错就改坏了用户的正文。

所以 D3 的定位是：**把该改的地方精确指出来（第几行、改成什么），由人执行。**
这与 E4/E5 的"给改写建议而不自动改稿"是同一条纪律。

## 唯一会给出具体替换值的情况

**题注编号不连续**（第 k 个图写成 图 n，n≠k）。
这种情况**无歧义**：题注是它自己的标签，改成顺序号一定是对的。
即便这样也只给建议，不动文件。
"""

from __future__ import annotations

import re
from typing import Sequence

from copilot.quality.format.docmodel import DocModel
from copilot.writing.models import (
    CaptionFact,
    FigureFix,
    FigureReport,
    ReferenceFact,
)

__all__ = ["analyze_figures", "REFERENCE_RE"]

#: 正文里的图表引用。`(?<![A-Za-z])` 而不是 `\w`：中文里 `\w` 会匹配 CJK，
#: 导致「如图 3」这种正常写法被 lookbehind 挡掉（E1 踩过这个坑）。
REFERENCE_RE = re.compile(r"(?<![A-Za-z])(图|表)\s*([0-9]{1,3})(?![0-9])")

_KIND = {"图": "figure", "表": "table"}


def _to_int(raw: str | None) -> int:
    """从 ``"1"`` / ``"2-1"`` 里取出首个整数；取不到返回 -1。"""
    m = re.search(r"[0-9]+", raw or "")
    return int(m.group()) if m else -1


def analyze_figures(doc: DocModel) -> FigureReport:
    """扫描题注与引用，产出修复建议。"""
    captions: list[CaptionFact] = [
        CaptionFact(
            kind=c.kind,
            line=c.line,
            number=_to_int(c.number),
            text=c.text or "",
            raw_number=str(c.number or ""),
        )
        for c in doc.captions
    ]

    caption_lines = {c.line for c in captions}
    references: list[ReferenceFact] = []
    for i, line in enumerate(doc.lines):
        if i in caption_lines:
            continue
        if line.lstrip().startswith("#"):
            continue
        for m in REFERENCE_RE.finditer(line):
            references.append(
                ReferenceFact(
                    kind=_KIND[m.group(1)],
                    line=i,
                    number=int(m.group(2)),
                    raw=m.group(0),
                )
            )

    fixes: list[FigureFix] = []

    #: 全文是否用了按章编号（图 2-1 / 图 3-1）
    #:
    #: 一旦是，**顺序重排与引用存在性检查全部跳过** —— 因为：
    #: ① 按章编号本来就是刻意的，不该被"重排成 1,2,3"；
    #: ② 我们的引用正则只能取到 `图 2-1` 里的首个整数 `2`，
    #:    拿它跟题注比会得出错误结论（实测：`图 2-1` 被当成 `图 2`）。
    #: **宁可少报，不可误报** —— 这是本项目的统一取向。
    chapter_based = any(c.chapter_based for c in captions)
    if chapter_based:
        return FigureReport(
            captions=captions,
            references=references,
            fixes=[],
            unreferenced=[],
        )

    # ── ① 题注编号不连续：无歧义，给具体替换值 ──
    for kind in ("figure", "table"):
        group = [c for c in captions if c.kind == kind]
        for idx, cap in enumerate(group, start=1):
            if cap.number > 0 and cap.number != idx:
                fixes.append(
                    FigureFix(
                        line=cap.line,
                        what=f"{cap.label}题注编号",
                        old=f"{cap.label} {cap.number}",
                        new=f"{cap.label} {idx}",
                        why=f"这是第 {idx} 个{cap.label}，按出现顺序应为 {idx}",
                    )
                )

    # ── ② 引用了不存在的图表：只报，不给替换值 ──
    existing = {
        kind: {c.number for c in captions if c.kind == kind and c.number > 0}
        for kind in ("figure", "table")
    }
    for ref in references:
        nums = existing[ref.kind]
        if not nums:
            continue
        if ref.number not in nums:
            fixes.append(
                FigureFix(
                    line=ref.line,
                    what=f"{ref.raw} 引用",
                    old=ref.raw,
                    new="（需人工确认）",
                    why=(
                        f"全文没有编号为 {ref.number} 的{ref.label}，"
                        f"现有编号：{sorted(nums)}"
                    ),
                )
            )

    # ── ③ 未被引用的图表 ──
    referenced = {(r.kind, r.number) for r in references}
    unreferenced = [
        c for c in captions if c.number > 0 and (c.kind, c.number) not in referenced
    ]

    return FigureReport(
        captions=captions,
        references=references,
        fixes=fixes,
        unreferenced=unreferenced,
    )
