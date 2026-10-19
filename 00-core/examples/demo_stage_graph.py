#!/usr/bin/env python3
"""演示：**每个阶段内部都是一张 LangGraph 状态图**。

    python examples/demo_stage_graph.py

## 这个演示要说明四件事

1. **图是真的** —— 能直接 `draw_mermaid()` 画出阶段内部结构；
2. **`only` 在建图前裁剪** —— 说"只跑 E1"，E2/E3 根本不进图；
3. **状态只有两个键** —— `data` + `results`，新增节点不用改声明；
4. **失败隔离** —— 一个节点炸了，后面的照样跑，但报告判为不可通过。

## 为什么状态只留一个 `data`

LangGraph 的图状态**只保留 TypedDict 里声明过的键**。
少声明一个，那个产物就**静默消失**（不报错，运行时表现为「下游拿到 None」）。

六个阶段往 `patch` 里写的业务字段有三十多个且会持续增加，
逐个声明必然漏。所以全部收进 `data`：

```python
class StageState(TypedDict, total=False):
    data: dict[str, Any]                             # 业务字段全在这里
    results: Annotated[list[dict], operator.add]     # 每个节点的结果（累加）
```

`results` 用 `operator.add` 做 reducer —— 每个节点只返回自己那一条，
累加而不是覆盖。没有它，后跑的节点会冲掉前面的结果。
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
SYSTEM_DIR = _HERE.parent
while SYSTEM_DIR != SYSTEM_DIR.parent and not (SYSTEM_DIR / "00-core").is_dir():
    SYSTEM_DIR = SYSTEM_DIR.parent
for _src in sorted(SYSTEM_DIR.glob("*/*/src")) + sorted(SYSTEM_DIR.glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from copilot.contracts import Finding  # noqa: E402
from copilot.format_stage import build_format_nodes, run_format  # noqa: E402
from copilot.quality.format.report import Severity  # noqa: E402
from copilot.stage import (  # noqa: E402
    StageNode,
    StageState,
    build_stage_graph,
    ok_outcome,
    run_stage,
)

DRAFT = """# 摘要

本文研究长尾推荐问题。

# 方法

我们设计了两阶段策略。

# 结论

方法有效。

# 参考文献

[1] 张三. 一种推荐方法[J]. 计算机学报, 2020.
"""

REQUIREMENT = "标题：黑体三号\n正文：宋体小四"


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def edges(mermaid: str) -> list[str]:
    return [ln.strip() for ln in mermaid.splitlines() if "-->" in ln]


def main() -> int:
    rule("① 状态只有两个键")
    print(f"  StageState 的键：{list(StageState.__annotations__)}")
    print("  → 业务字段全在 data 里，新增节点不用改状态声明。")
    print("  → results 用 operator.add 累加，避免后跑的节点冲掉前面的。")

    rule("② E 阶段的图（完整节点）")
    nodes = build_format_nodes(draft=DRAFT, requirement=REQUIREMENT)
    app = build_stage_graph(nodes).compile()
    for line in edges(app.get_graph().draw_mermaid()):
        print(f"  {line}")
    print("  → 线性链：START → E1 → E2 → E3 → END")

    rule("③ only 在**建图前**裁剪：只跑 E1，E2/E3 根本不进图")
    only_one = [n for n in nodes if n.code == "E1"]
    app1 = build_stage_graph(only_one).compile()
    for line in edges(app1.get_graph().draw_mermaid()):
        print(f"  {line}")
    print("  → 图里只有 E1。这就是主 Agent 做节点级路由的实现方式。")

    rule("④ 跑一遍（走图），报告与普通调用一致")
    report = run_format(draft=DRAFT, requirement=REQUIREMENT)
    for r in report.results:
        print(f"  {r.node}  {r.summary_line().split('：', 1)[-1][:46]}")
    print(f"  → {report.summary_line()}")

    rule("⑤ 失败隔离：中间节点炸了，后面的照样跑")
    def boom(_state):
        raise RuntimeError("这个节点故意失败")

    demo_nodes = (
        StageNode("n1", "正常节点", lambda s: ok_outcome(
            findings=[Finding("n1", "N1", Severity.INFO, "n1 跑了")])),
        StageNode("n2", "会失败的节点", boom),
        StageNode("n3", "正常节点", lambda s: ok_outcome(
            findings=[Finding("n3", "N3", Severity.INFO, "n3 跑了")])),
    )
    broken = run_stage(stage="X", label="demo", nodes=demo_nodes, state={})
    for r in broken.results:
        print(f"  {r.node}  ok={r.ok}  {r.error or ''}")
    print(f"  → 结论：{broken.verdict()}")
    print("  → 节点 n2 挂了，n3 仍然跑了；但报告判为不可通过 ——")
    print("    **跑挂的检查 ≠ 这项没问题。**")

    rule("⑥ 可暂停：传 checkpointer 就能带 thread_id 跑")
    from langgraph.checkpoint.memory import InMemorySaver

    paused_ready = run_stage(
        stage="E", label="demo", nodes=nodes, state={"draft": DRAFT},
        checkpointer=InMemorySaver(), thread_id="stage-demo",
    )
    print(f"  跑了 {len(paused_ready.results)} 个节点，结论：{paused_ready.verdict()}")
    print("  → 现在默认不传 checkpointer（阶段跑完就出报告，无需人工确认），")
    print("    但接口是通的：哪天某个阶段要加'改完让人确认'，传一个就行。")

    rule("要点")
    print(
        "  1. **每个阶段内部都是一张 StateGraph**，六个阶段共用同一套执行器；\n"
        "  2. **only 在建图前裁剪** —— 没选中的节点不进图，不是进了图再跳过；\n"
        "  3. **状态只有 data + results** —— 避免'少声明一个键就静默丢数据'；\n"
        "  4. **results 必须用 reducer** —— 否则后跑的节点会覆盖前面的结果；\n"
        "  5. **失败隔离写在节点包装里** —— 图不会因为一个节点抛异常而整体失败。"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
