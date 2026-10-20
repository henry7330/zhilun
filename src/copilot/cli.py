"""命令行入口：**不写代码也能试**。

    python -m copilot --demo                        # 六阶段全跑（内置样例，无需模型）
    python -m copilot --plan "帮我看看格式"           # 只看主 Agent 的规划
    python -m copilot "把格式整理一下" --draft a.md   # 按请求执行
    python -m copilot --stage E --draft a.md         # 直接调某一个阶段

设计上只有一条规则：**没有模型也能跑**。
需要模型的节点会退化成骨架并如实标注，不会假装做完了。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main", "SAMPLE_DRAFT"]

SAMPLE_DRAFT = """# 摘要

本文研究推荐系统中的长尾问题。

# 方法

我们设计了两阶段检索策略，并在 5 个数据集上验证了效果。

# 结论

方法有效。

# 参考文献

[1] 张三. 一种推荐方法[J]. 计算机学报, 2020.
"""

SAMPLE_REPORT = "我们设计了两阶段检索策略，并在 5 个数据集上验证了效果。"

SAMPLE_REQUIREMENT = "标题：黑体三号居中\n正文：宋体小四，行距 1.5 倍"


def _read(path: str | None, default: str = "") -> str:
    if not path:
        return default
    p = Path(path)
    if not p.is_file():
        print(f"❌ 文件不存在：{path}", file=sys.stderr)
        raise SystemExit(2)
    return p.read_text(encoding="utf-8")


def _print_plan(report) -> None:
    plan = report.plan
    print("=" * 70)
    print("主 Agent 规划")
    print("=" * 70)
    print(f"请求：{plan.request}")
    if not plan.steps:
        print("（没有匹配到任何阶段）")
        for n in plan.notes:
            print(f"  · {n}")
        return
    print(f"计划：{plan.describe()}")
    print()
    print(f"{'阶段':<4} {'节点':<14} {'模型':<14} 为什么")
    print("-" * 70)
    for s in plan.steps:
        nodes = "全部" if s.nodes is None else "、".join(s.nodes)
        print(f"{s.stage:<4} {nodes:<14} {s.model or '—':<14} {s.reason}")
    if plan.missing:
        from copilot.orchestrator import INPUT_LABEL

        names = "、".join(INPUT_LABEL.get(m, m) for m in plan.missing)
        print(f"\n⚠️ 缺少输入：{names}（缺了会跳过对应阶段）")
    for n in plan.notes:
        print(f"  · {n}")


def _run_demo() -> int:
    """六阶段各跑一遍（每段用自己的输入，互不依赖）。"""
    from copilot.bib import BibEntry
    from copilot.format_stage import run_format
    from copilot.literature import run_literature
    from copilot.research import run_research
    from copilot.revise import run_revise
    from copilot.topic import run_topic
    from copilot.writing import run_writing

    library = [
        BibEntry(key="zhang2020", authors=["张三"], title="一种推荐方法",
                 venue="计算机学报", year=2020, doi="10.1234/abc", cn_type="J")
    ]
    runs = [
        ("A 选题分析", lambda: run_topic(title="基于图神经网络的推荐方法")),
        ("B 文献检索", lambda: run_literature(entries=library)),
        ("C 研究路线", lambda: run_research("方法是否优于基线？")),
        ("D 论文写作", lambda: run_writing(title="基于图神经网络的推荐方法",
                                          draft=SAMPLE_DRAFT, entries=library)),
        ("E 格式规整", lambda: run_format(draft=SAMPLE_DRAFT, requirement=SAMPLE_REQUIREMENT)),
        ("F 降重降AI", lambda: run_revise(draft=SAMPLE_DRAFT, report=SAMPLE_REPORT)),
    ]

    print("=" * 70)
    print("织论 Zhilun —— 六个阶段各跑一遍（无需模型）")
    print("=" * 70)
    for name, fn in runs:
        try:
            report = fn()
            print(f"\n{name}")
            print(f"  {report.summary_line()}")
            for r in report.results:
                mark = "✅" if r.ok else "❌"
                print(f"    {mark} {r.node:<4} {r.summary_line().split('：', 1)[-1][:56]}")
        except Exception as exc:  # noqa: BLE001 - 演示不该因单阶段失败而中断
            print(f"\n{name}\n  ❌ {type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print("提示：")
    print("  · 接模型：在仓库根建 .env（COPILOT_LLM_API_KEY / _BASE_URL / _MODEL）")
    print('  · 只出计划不做：python -m copilot --plan "帮我看看格式"')
    print("  · 交互页面：streamlit run app.py")
    print("=" * 70)
    return 0


def _build_rewriter():
    """构造真实改写器；没配模型时返回 ``None``（退回骨架并如实标注）。

    只有 **F2 改写** 有模型实现。E 阶段刻意不接模型（只改格式、不改内容），
    A/B/C 与 D1/D3/D4 是确定性问题，D2 目前只有骨架生成器。
    """
    try:
        from copilot.llm import LlmConfig
        from copilot.quality.rewrite import LlmRewriter

        return LlmRewriter(LlmConfig.from_env())
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ 未能启用模型（{type(exc).__name__}）：退回骨架路径", file=sys.stderr)
        return None


def _run_stage(stage: str, draft: str, report: str, requirement: str, rewriter=None):
    if stage == "E":
        from copilot.format_stage import run_format

        return run_format(draft=draft, requirement=requirement)
    if stage == "F":
        from copilot.revise import run_revise

        return run_revise(draft=draft, report=report or SAMPLE_REPORT, rewriter=rewriter)
    if stage == "D":
        from copilot.writing import run_writing

        return run_writing(title="（未命名）", draft=draft)
    if stage == "C":
        from copilot.research import run_research

        return run_research("（未提供研究问题）")
    if stage == "B":
        from copilot.literature import run_literature

        return run_literature()
    from copilot.topic import run_topic

    return run_topic(title="（未命名）")


def _prepare_streams() -> None:
    """让输出在**非 tty**（CI / 管道 / 重定向）下也不会因编码崩掉。

    实测：GitHub Actions 的 Windows runner 把 stdout 当管道，Python 用
    ``locale.getpreferredencoding()`` = **cp1252**，打印中文直接
    ``UnicodeEncodeError`` —— 演示命令因此在 CI 上失败（本地终端不复现，
    因为 Windows 控制台走 UTF-16 API）。

    处理方式：
    - **非 tty**（管道 / 重定向）→ 切 UTF-8，日志里中文可读；
    - **tty**（真实终端）→ 保持原编码，避免中文控制台变乱码；
    - 两种情况都加 ``errors="replace"``，保证任何环境都不会因一个字符崩掉。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if stream.isatty():
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _prepare_streams()
    # 有 .env 就读进来（仓库根 / 当前目录都能找到）；没有也不报错，
    # 需要模型的节点会退化成骨架并在报告里标注。
    from copilot.llm import load_env

    load_env()
    ap = argparse.ArgumentParser(
        prog="zhilun",
        description="织论 Zhilun —— 基于 LangGraph 的多智能体 AI 论文辅助系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m copilot --demo\n"
            '  python -m copilot --plan "帮我看看格式"\n'
            '  python -m copilot "把格式整理一下" --draft a.md\n'
            "  python -m copilot --stage E --draft a.md\n"
        ),
    )
    ap.add_argument("request", nargs="?", help="自然语言请求，交给主 Agent 拆解")
    ap.add_argument("--draft", metavar="FILE", help="论文草稿（Markdown）")
    ap.add_argument("--report", metavar="FILE", help="外部查重 / AI 检测报告")
    ap.add_argument("--requirement", metavar="TEXT", help="学校格式要求（纯文本）")
    ap.add_argument("--stage", choices=list("ABCDEF"), help="跳过主 Agent，直接跑某个阶段")
    ap.add_argument("--plan", action="store_true", help="只出计划，不执行")
    ap.add_argument("--demo", action="store_true", help="六阶段各跑一遍内置样例")
    ap.add_argument(
        "--llm",
        action="store_true",
        help="启用模型（F2 改写需要它才有实际效果；不加以则走骨架，不花钱）",
    )
    ap.add_argument("--json", action="store_true", help="输出 JSON（便于脚本消费）")
    args = ap.parse_args(argv)

    if args.demo:
        return _run_demo()

    draft = _read(args.draft, SAMPLE_DRAFT)
    report_text = _read(args.report, "")
    requirement = args.requirement or ""
    rewriter = _build_rewriter() if args.llm else None

    # 「不做静默失败」：E1 的 formatter 钩子还没有 LLM 实现，
    # 传了 --requirement 也不会生效 —— 必须说出来，而不是让用户以为用上了。
    if requirement.strip():
        print(
            "⚠️ --requirement 目前不会生效：E1 的 formatter 钩子尚未接入模型实现，"
            "本次只跑 14 条固定格式规则。",
            file=sys.stderr,
        )

    if args.stage:
        try:
            rep = _run_stage(args.stage, draft, report_text, requirement, rewriter=rewriter)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(rep.to_markdown())
        return 0 if rep.can_submit else 1

    if not args.request:
        ap.print_help()
        return 0

    from copilot.orchestrator import run_request

    context = {
        "draft": draft,
        "report": report_text,
        "requirement": requirement,
        "title": "（未命名）",
        "question": "（未提供）",
        "rewriter": rewriter,
    }
    out = run_request(args.request, context=context, plan_only=args.plan)

    if args.json:
        print(json.dumps(out.to_dict(), ensure_ascii=False, indent=2))
        return 0 if out.can_submit else 1

    _print_plan(out)
    if args.plan:
        return 0

    print("\n" + "=" * 70)
    print("执行结果")
    print("=" * 70)
    for s in out.steps:
        print(f"  {s.summary_line()[:74]}")
    print(f"\n{out.summary_line()}")
    return 0 if out.can_submit else 1


if __name__ == "__main__":
    raise SystemExit(main())