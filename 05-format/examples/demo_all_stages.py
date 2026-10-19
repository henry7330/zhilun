#!/usr/bin/env python3
"""六个阶段各自独立调用 —— **没有全自动链路**。

    python examples/demo_all_stages.py

## 这个演示要说明的一件事

**每个阶段都是一个独立服务。用户需要哪个就调哪个，阶段之间不传递数据。**

```
run_topic(...)        ← 只给题目和语料，不认识 B/C/D
run_literature(...)   ← 只给关键词和文献，不认识 A/C/D
run_research(...)     ← 只给问题和数据，不认识 A/B/D
run_writing(...)      ← 只给标题和设计字典，不认识 A/B/C
run_quality(...)      ← 只给草稿和报告，不认识 A/B/C/D
（F 阶段见 06-delivery/）
```

**注意**：下面每一段都用**自己造的输入**，不是上一段的输出。
这正是"阶段独立"的含义 —— 如果你只想用质检，不需要先跑选题。

## 阶段内部的节点是全自动的

调用一次，该阶段的节点按声明顺序跑完（E 阶段还有条件分支）。
用户不需要逐个调节点。

输出：`reports/stages_demo.md`
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

from copilot.bib import BibEntry  # noqa: E402
from copilot.contracts import STAGES  # noqa: E402
from copilot.format_stage import run_format  # noqa: E402
from copilot.literature import run_literature  # noqa: E402
from copilot.research import Observation, run_research  # noqa: E402
from copilot.revise import run_revise  # noqa: E402
from copilot.topic import run_topic  # noqa: E402
from copilot.writing import run_writing  # noqa: E402

REPORT_DIR = STAGE_DIR / "reports"


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show(report, name: str) -> None:
    print(f"  {name}")
    print(f"  {report.summary_line()}")
    for r in report.results:
        mark = "✅" if r.ok else "❌"
        notes = ("；".join(r.notes))[:64]
        print(f"    {mark} {r.node:<3} {r.summary_line().split('：', 1)[1]:<28} {notes}")
    print()


def main() -> int:
    rule("① 六个 Agent（阶段）")
    for spec in STAGES.values():
        print(f"  {spec.key} · {spec.name:<8} 节点 {len(spec.nodes)} 个  入口 {spec.entry}")
        print(f"      节点：{'、'.join(n.name for n in spec.nodes)}")
    print()
    print("  → **一个阶段 = 一个 Agent**；节点是它内部的工作内容。")

    rule("② 阶段 A · 选题与开题（独立调用）")
    a = run_topic(
        title="基于图神经网络的推荐方法研究",
        texts=[
            "检索增强生成在推荐系统中的应用研究",
            "图神经网络用于推荐系统的方法",
            "基于检索增强的问答系统研究",
            "长尾用户的行为稀疏问题与元学习",
            "推荐系统中的冷启动问题研究",
            "图神经网络在长尾推荐中的应用",
        ],
        years={0: 2019, 1: 2020, 2: 2020, 3: 2021, 4: 2022, 5: 2022},
    )
    show(a, "输入：题目 + 6 篇语料 + 年份")

    rule("③ 阶段 B · 文献（独立调用）")
    entries = [
        BibEntry(key="a", authors=["张三"], title="检索增强方法研究",
                 venue="计算机学报", year=2020, doi="10.1/x", cn_type="J"),
        BibEntry(key="b", authors=["张三"], title="检索增强 方法研究",
                 venue="计算机学报", year=2020, doi="10.1/x", cn_type="J"),
        BibEntry(key="c", authors=["李四"], title="图神经网络综述",
                 venue="软件学报", year=2022, cn_type="J"),
    ]
    b = run_literature(entries=entries, label="refs")
    show(b, "输入：3 条参考文献（其中 2 条重复）")

    rule("④ 阶段 C · 研究（独立调用）")
    data = {
        "ndcg@10": [Observation(f"u{i:02d}", 0.30 + 0.001 * i, 0.34 + 0.0012 * i) for i in range(30)],
        "latency": [Observation(f"u{i:02d}", 50 + i, 46 + i) for i in range(30)],
    }
    c = run_research("图神经网络在长尾推荐上是否优于矩阵分解？", data=data)
    show(c, "输入：研究问题 + 30 条配对观测 × 2 个指标")

    rule("⑤ 阶段 D · 论文写作（独立调用）")
    library = [
        BibEntry(key="zhang2020", authors=["张三"], title="一种推荐方法",
                 venue="计算机学报", year=2020, doi="10.1234/abc", cn_type="J"),
    ]
    d = run_writing(
        title="基于图神经网络的推荐方法研究",
        design={
            "method": "两阶段：粗筛 + 精排",
            "alternatives": ["单阶段端到端"],
            "baselines": ["BM25", "矩阵分解"],
            "metrics": [{"name": "nDCG@10"}],
            "sample_size": {"n": 30},
            "risks": ["长尾覆盖不足"],
            "variables": {"independent": ["检索策略"], "dependent": ["nDCG"], "controlled": ["语料"]},
        },
        draft=(
            "# 论文\n\n## 方法\n\n如 图 5 所示。\n\n**图 2** 框架\n\n"
            "## 参考文献\n\n[1] 张三. 一种推荐方法[J]. 计算机学报, 2020.\n"
        ),
        required_sections=["摘要", "研究方法", "实验与分析", "结论"],
        entries=library,
    )
    show(d, "输入：标题 + 设计字典 + 草稿 + **文献库**（跑 D3 图表 / D4 引用来源）")

    rule("⑥ 阶段 E · 格式规整（独立调用）")
    e = run_format(
        draft=(
            "# 摘要\n\n本文研究长尾推荐问题。\n\n"
            "# 方法\n\n我们设计了两阶段策略。\n\n"
            "# 结论\n\n方法有效。\n\n"
            "# 参考文献\n\n[1] 张三. 一种推荐方法[J]. 计算机学报, 2020.\n"
        ),
        requirement="标题：黑体三号居中\n正文：宋体小四，行距 1.5 倍\n参考文献：GB/T 7714\n",
        formatter=lambda text, req: text.replace("我们设计了", "本文设计了"),
    )
    show(e, "输入：草稿 + 学校格式要求（一段式 LLM 规整 + 导出）")

    rule("⑦ 阶段 F · 降重降AI（独立调用，**必须给外部报告**）")
    f = run_revise(
        draft="# 方法\n\n本文提出了一种基于图神经网络的推荐方法，并在五个数据集上验证了效果。\n",
        report="本文提出了一种基于图神经网络的推荐方法，并在五个数据集上验证了效果。",
        report_type="both",
    )
    show(f, "输入：草稿 + 外部检测报告（未接模型 → 一律不采纳）")
    try:
        run_revise(draft="# 方法\n\n内容。\n", report="")
    except ValueError as exc:
        print(f"  没给报告时的行为：ValueError -> {str(exc)[:60]}…")
    print()

    rule("⑧ 落盘")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    parts = [
        "# 六个阶段独立调用演示",
        "",
        "> 每一段都用自己的输入，**不是上一段的输出** —— 这就是「阶段独立」。",
        "",
        "## 阶段清单",
        "",
        "| 阶段 | Agent | 节点 | 入口 |",
        "|---|---|---|---|",
    ]
    for spec in STAGES.values():
        parts.append(
            f"| {spec.key} | {spec.name} | {'、'.join(n.name for n in spec.nodes)} | `{spec.entry}` |"
        )
    for name, rep in (
        ("A 选题分析", a), ("B 文献检索", b), ("C 研究路线", c),
        ("D 论文写作", d), ("E 格式规整", e), ("F 降重降AI", f),
    ):
        parts += ["", f"## {name}", "", rep.to_markdown()]
    out = REPORT_DIR / "stages_demo.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"  reports/stages_demo.md")

    rule("要点")
    print(
        "  1. **一个阶段 = 一个 Agent**，节点是它内部的工作内容；\n"
        "  2. **阶段之间不传数据** —— 每个入口都接受独立输入，可以单独使用；\n"
        "  3. **阶段内部全自动** —— 调一次，节点按顺序跑完；\n"
        "  4. **无模型也能跑** —— 需要模型的节点退化为骨架，并如实标注；\n"
        "  5. **零模型的部分是主体** —— A1 / B1 / B3 / C2 / D1 / D3 / D4 全部确定性；\n"
        "  6. **F 阶段没有报告就不运行** —— 跳过会被误读成「检查过了没问题」。"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
