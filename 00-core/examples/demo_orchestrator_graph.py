#!/usr/bin/env python3
"""演示：主 Agent 的 **LangGraph 版** —— 图、计划确认、断点续。

    python examples/demo_orchestrator_graph.py

## 与 demo_orchestrator.py 的分工

| | `run_request`（走图） | 本演示 `run_request_graph` |
|---|---|---|
| 依赖 | 无（不需要 langgraph） | 需要 langgraph |
| 执行前确认计划 | 只能调两次 | **`interrupt`，同一请求内暂停** |
| 崩溃恢复 | 不能 | **checkpointer + thread_id** |
| 编排可视化 | 读代码 | **`draw_mermaid()`** |

**两版给出一致的计划与结论** —— 这是本演示要验证的第一件事。

输出：半段控制台 + 一段 mermaid（可直接贴进 README）
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
STAGE_DIR = SYSTEM_DIR / "05-format"
for _src in sorted(SYSTEM_DIR.glob("*/*/src")) + sorted(SYSTEM_DIR.glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from copilot.orchestrator import run_request  # noqa: E402
from copilot.orchestrator.graph import (  # noqa: E402
    build_orchestrator_graph,
    run_request_graph,
)

REPORT_DIR = STAGE_DIR / "reports"

DRAFT = """# 摘要

本文研究推荐系统中的长尾问题。

# 方法

我们设计了两阶段检索策略。

# 结论

方法有效。
"""

CTX = {
    "title": "图神经网络推荐",
    "draft": DRAFT,
    "requirement": "标题：黑体三号\n正文：宋体小四",
    "report": "我们设计了两阶段检索策略。",
}


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    rule("① 图长什么样")
    mermaid = build_orchestrator_graph(context=CTX).get_graph().draw_mermaid()
    print(mermaid)
    print("\n  读法：")
    print("    plan → (需确认?) approve → (同意?) pick → run → pick …→ summarize")
    print("    pick → run → pick 是**循环** —— 手写 for 循环在图里的等价形式。")

    rule("② 两版必须等价（同一请求 → 同计划、同结论）")
    request_text = "把格式整理一下，再看看有没有重复的段落"
    manual = run_request(request_text, context=CTX)
    graphed = run_request_graph(request_text, context=CTX).to_report()
    print(f"  run_request 计划：{manual.plan.describe()}  结论：{manual.summary_line()}")
    print(f"  run_request_graph 计划：{graphed.plan.describe()}  结论：{graphed.summary_line()}")
    same = (
        manual.plan.describe() == graphed.plan.describe()
        and manual.summary_line() == graphed.summary_line()
    )
    print(f"  → 一致：{same}")

    rule("③ 计划确认：跑到一半停下，等人点头")
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    run = run_request_graph(
        request_text, context=CTX, require_approval=True,
        checkpointer=saver, thread_id="demo-approve",
    )
    print(f"  需要确认：{run.needs_approval}")
    print(f"  已执行步数：{len(run.state.get('results') or [])}   ← 确认前一步都没跑")
    payload = run.pending or {}
    print(f"  待确认计划：{payload.get('stages')}")
    for s in payload.get("steps", []):
        nodes = "全部" if s["nodes"] is None else "、".join(s["nodes"])
        print(f"    · {s['stage']}  节点={nodes}  模型={s['model'] or '—'}")

    rule("④ 拒绝执行 → 什么都不跑（而且如实说）")
    rejected = run.resume(False).to_report()
    print(f"  {rejected.summary_line()}")
    print(f"  执行步数：{len(rejected.ran_steps)}")

    rule("⑤ 同意执行 → 按计划跑完")
    run2 = run_request_graph(
        request_text, context=CTX, require_approval=True,
        checkpointer=saver, thread_id="demo-approve-yes",
    )
    done = run2.resume(True).to_report()
    for s in done.steps:
        print(f"  {s.summary_line()[:72]}")
    print(f"  → {done.summary_line()}")

    rule("⑥ 落盘")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "orchestrator_graph.md"
    out.write_text(
        "# 主 Agent（LangGraph 版）\n\n"
        "## 图\n\n```mermaid\n" + mermaid + "\n```\n\n"
        "## 运行\n\n" + done.to_markdown(),
        encoding="utf-8",
    )
    print(f"  reports/orchestrator_graph.md")

    rule("要点")
    print(
        "  1. **图是真的**：plan → approve? → pick → run → pick（循环）→ summarize；\n"
        "  2. **interrupt 让「先看计划再执行」变成一次请求内的事**，不用调两次；\n"
        "  3. **拒绝执行 = 一步都不跑**，且结论如实说「没有任何阶段被执行」；\n"
        "  4. **两版等价**：换编排方式不改变业务行为；\n"
        "  5. **包级导入是延迟的** —— import 本包不会拉起 langgraph。"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
