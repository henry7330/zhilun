#!/usr/bin/env python3
"""演示：**主 Agent 如何把一句话拆成执行计划**。

    python examples/demo_orchestrator.py

## 这个演示要说明什么

1. **不是固定流程** —— 同一套系统，问"格式"就只跑 E，问"降重"就只跑 F；
2. **三层路由**都能看见：调哪个阶段 / 阶段里跑哪几个节点 / 用哪个模型；
3. **节点收窄只减不增**，且依赖会被补齐（说"改写"会自动带上"定位"）；
4. **缺输入就跳过并说明**，不硬跑、也不假装通过。

输出：`05-format/reports/orchestrator_demo.md`
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
# 向上找系统根（含 00-core 的目录）
SYSTEM_DIR = _HERE.parent
while SYSTEM_DIR != SYSTEM_DIR.parent and not (SYSTEM_DIR / "00-core").is_dir():
    SYSTEM_DIR = SYSTEM_DIR.parent
STAGE_DIR = SYSTEM_DIR / "05-format"
for _src in sorted(SYSTEM_DIR.glob("*/*/src")) + sorted(SYSTEM_DIR.glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from copilot.bib import BibEntry  # noqa: E402
from copilot.orchestrator import ModelRouter, run_request  # noqa: E402

REPORT_DIR = STAGE_DIR / "reports"

DRAFT = """# 摘要

本文研究推荐系统中的长尾问题。

# 方法

我们设计了两阶段检索策略。

# 结论

方法有效。

# 参考文献

[1] 张三. 一种推荐方法[J]. 计算机学报, 2020.
"""

LIBRARY = [
    BibEntry(key="zhang2020", authors=["张三"], title="一种推荐方法",
             venue="计算机学报", year=2020, doi="10.1234/abc", cn_type="J"),
]


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_plan(request: str, ctx: dict) -> None:
    plan = run_request(request, context=ctx, plan_only=True).plan
    if not plan.steps:
        print(f"  {request:34} → （没有匹配到阶段）")
        return
    desc = []
    for s in plan.steps:
        nodes = "全部" if s.nodes is None else "/".join(s.nodes)
        desc.append(f"{s.stage}[{nodes}]")
    print(f"  {request:34} → {' + '.join(desc)}")
    for s in plan.steps:
        print(f"      {s.stage}: {s.reason}")


def main() -> int:
    rule("① 同一个系统，不同的问法 → 不同的计划")
    print("  （每行是一次独立请求，互不依赖）\n")
    base = {"title": "图神经网络推荐", "draft": DRAFT, "question": "A 比 B 好吗"}
    for q in [
        "帮我看看格式",
        "字体和字号对不对",
        "帮我降重",
        "这段改写一下",
        "引用来源靠谱吗",
        "帮我做个选题",
        "看一下研究空白和趋势",
        "把格式整理一下，再看看有没有重复的段落",
        "随便说点什么",
    ]:
        show_plan(q, {**base, "report": "本文设计了两阶段检索策略。"})

    rule("② 三层路由，逐层看")
    plan = run_request(
        "把格式整理一下，再看看有没有重复的段落",
        context={**base, "report": "本文设计了两阶段检索策略。"},
        router=ModelRouter(default="gpt-5.6-luna", by_node={"E1": "strict-model"}),
        plan_only=True,
    ).plan
    for s in plan.steps:
        scope = "全部节点" if s.nodes is None else "、".join(s.nodes)
        print(f"  {s.stage}  节点={scope:12} 模型={s.model or '（确定性，不用模型）'}")
    print()
    print("  → 第③层是「模型选择」：E1 登记了 strict-model，所以用了它；")
    print("    F 没单独登记，落到默认模型。")

    rule("③ 缺输入时：跳过，并说人话")
    r = run_request("把格式整理一下，再看看有没有重复的段落", context={"draft": DRAFT})
    for s in r.steps:
        print(f"  {s.summary_line()[:76]}")
    print()
    print(f"  整体结论：{r.summary_line()}")

    rule("④ 补齐输入后，两个阶段都跑")
    r = run_request(
        "把格式整理一下，再看看有没有重复的段落",
        context={
            "draft": DRAFT,
            "requirement": "标题：黑体三号\n正文：宋体小四",
            "report": "本文设计了两阶段检索策略。",
        },
    )
    for s in r.steps:
        print(f"  {s.summary_line()[:76]}")
    print()
    print(f"  整体结论：{r.summary_line()}")

    rule("⑤ 执行报告落盘")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "orchestrator_demo.md"
    out.write_text(r.to_markdown(), encoding="utf-8")
    print(f"  reports/orchestrator_demo.md（{len(out.read_text(encoding='utf-8').splitlines())} 行）")

    rule("要点")
    print(
        "  1. **它是 AI 系统，不是固定流程** —— 计划由请求决定，六阶段不必全跑；\n"
        "  2. **三层路由**：选阶段 → 选节点 → 选模型，每层都能在报告里看到；\n"
        "  3. **收窄只减不增**：用户没要求的节点不会跑（成本与风险不该替他承担）；\n"
        "  4. **依赖必须补齐**：说「改写」会自动带上「定位」，否则下游拿不到输入；\n"
        "  5. **缺输入就跳过**：跳过 ≠ 通过，报告里会写明缺哪个输入；\n"
        "  6. **规划可以先用 plan_only 看一眼**，确认后再花钱执行。"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
