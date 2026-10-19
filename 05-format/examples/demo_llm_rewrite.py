#!/usr/bin/env python3
"""演示：接真实模型的改写（E4 降重 / E5 降AI）。

    python examples/demo_llm_rewrite.py            # 用 .env 里的配置
    python examples/demo_llm_rewrite.py --dry-run  # 只看提示词，不调模型（省钱）

**没配密钥时不会报错**，会打印"跳过"并返回 0 ——
这样"跑全部演示"这条命令在没有密钥的机器上也成立。

演示四件事：

1. **提示词长什么样** —— 硬约束逐条列死（数字 / 专名 / 引用 / 结论方向）
2. **真实模型改写 + 双重校验** —— 效果达标 & 保真通过，才算采纳
3. **保真校验真的会拦** —— 同一套校验，换个不保真的候选立刻被拒
4. **E5 用同一个改写器** —— 只是策略不同，不另写一套提示词

输出：`reports/llm_rewrite_report.md`
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

from copilot.llm import LlmConfig, LlmError, load_env, ping  # noqa: E402
from copilot.quality.deai import run_deai  # noqa: E402
from copilot.quality.report_parse import FlagType, build_checklist  # noqa: E402
from copilot.quality.rewrite import (  # noqa: E402
    ContentKind,
    LlmRewriter,
    RewriteInstruction,
    RewriteRequest,
    Strategy,
    build_messages,
    check_fidelity,
    run_rewrite,
)

REPORT_DIR = STAGE_DIR / "reports"

THESIS = """# 基于图神经网络的推荐方法研究

## 方法

本文提出了一种基于图神经网络的推荐方法，并在三个数据集上验证了效果。
实验在 8 张 V100 上完成，批量大小为 256，训练了 30 个 epoch。

## 结论

该方法在低资源场景下仍有提升空间。
"""

#: 外部查重报告标的重复片段
DUP_SNIPPET = "本文提出了一种基于图神经网络的推荐方法，并在三个数据集上验证了效果。"

#: 一个**故意不保真**的候选：把 256 改成 512，把 30 改成 40
UNFAITHFUL = "我们采用图神经网络做推荐，在 8 张 V100 上把批量大小设为 512，训练了 40 个 epoch。"

#: 外部 AI 检测报告标的片段（风格问题）
AI_SNIPPET = "该方法在低资源场景下仍有提升空间。"


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    dry = "--dry-run" in sys.argv
    load_env()
    try:
        cfg = LlmConfig.from_env()
    except LlmError as exc:
        print("\n  ⏭️  跳过：", exc)
        print("     在 05-quality/.env 里配置 COPILOT_LLM_API_KEY / _BASE_URL / _MODEL 后重跑。")
        print("     （本次演示不需要模型也能跑其余环节，见 examples/demo_system.py）")
        print("=" * 78)
        return 0

    rule("① 配置（密钥已脱敏）")
    print(f"  {cfg}")
    print()
    print("  → 打印配置时不会泄露密钥：`api_key=<已设置 N 字符>`")

    rule("② 探活")
    try:
        models = ping(cfg)
        print(f"  端点可达，可用模型 {len(models)} 个：{', '.join(models[:6])}…")
    except LlmError as exc:
        print(f"  ❌ {exc}")
        return 1

    rule("③ 一次真实改写（E4 降重）")
    checklist = build_checklist(THESIS, DUP_SNIPPET, flag_type=FlagType.DUPLICATE)
    print(f"  清单：{checklist.summary_line()}\n")

    unit_text = checklist.items[0].item.text if checklist.items else DUP_SNIPPET
    req = RewriteRequest(
        text=unit_text,
        instruction=RewriteInstruction(
            strategy=Strategy.REPHRASE, kind=ContentKind.METHOD,
            hint="改句式与语序，保留全部术语与数字", rationale="方法描述里的术语换了就不准确",
        ),
        context="",
    )
    if dry:
        msgs = build_messages(req)
        print("  【--dry-run】只看提示词，不调模型：\n")
        print("  [system]")
        for line in msgs[0]["content"].splitlines()[:8]:
            print(f"    {line}")
        print(f"    …（共 {len(msgs[0]['content'].splitlines())} 行硬约束）")
        print("\n  [user]")
        for line in msgs[1]["content"].splitlines():
            print(f"    {line}")
        print("=" * 78)
        return 0

    rewriter = LlmRewriter(cfg)
    print(f"  改写器：{rewriter.name}（调用中，约 10–30 秒）…\n")
    try:
        report = run_rewrite(checklist, rewriter=rewriter)
    except LlmError as exc:
        print(f"  ❌ 模型调用失败：{exc}")
        return 1

    print(f"  {report.summary_line()}\n")
    for s in report.suggestions:
        status = getattr(s.status, "value", s.status)
        print(f"  状态：{status}")
        print(f"    原文：{s.unit.text[:80]}")
        print(f"    改写：{(s.chosen or '（无）')[:80]}")
        if s.effect:
            print(f"    效果：连续重叠 {s.effect.after_run} 字 / "
                  f"n-gram 包含 {s.effect.after_containment:.0%} / 达标={s.effect.meets_target}")
        if s.fidelity is not None:
            print(f"    保真：{s.fidelity.summary()}")
        print()

    st = rewriter.stats()
    print(f"  运行统计：调用 {st['calls']} 次 / 请求候选 {st['n_candidates']} 个 / "
          f"服务端忽略 n 的次数 {st['n_ignored']}")
    if not st["n_supported"]:
        print("  ⚠️  该端点**不支持 n 参数**（请求多个只回一个，且不报错）。")
        print("     所以提高通过率要靠 `max_attempts` 重试，而不是多候选。")

    rule("④ ★ 保真校验会真的拦：把数字改掉试试")
    print("  同一套校验，换一个「数字被改」的候选：\n")
    print(f"    原文  ：{unit_text[:70]}")
    print(f"    假候选：{UNFAITHFUL}")
    fid = check_fidelity(rewritten=UNFAITHFUL, original=unit_text)
    print(f"\n    结论  ：{fid.summary()}")
    if not fid.ok:
        for issue in fid.issues:
            print(f"      · [{issue.kind}] {issue.detail}")
    print()
    print("  → 这就是「提示词是软约束、校验器才是硬闸门」：")
    print("     模型不听话也拦得住。效果再好，只要不保真就弃用。")

    rule("⑤ E5 降AI：同一个改写器，不同策略")
    ai_checklist = build_checklist(THESIS, AI_SNIPPET, flag_type=FlagType.AI)
    from copilot.quality.deai import run_deai

    try:
        deai_report = run_deai(ai_checklist, rewriter=rewriter)
    except LlmError as exc:
        print(f"  ❌ 模型调用失败：{exc}")
        return 1
    print(f"  {deai_report.summary_line()}\n")
    for s in deai_report.suggestions:
        status = getattr(s.status, "value", s.status)
        print(f"  状态：{status}")
        print(f"    原文：{s.unit.text[:70]}")
        print(f"    改写：{(s.chosen or '（无）')[:70]}")
        if getattr(s, "verification", None):
            for ch in s.verification.changes:
                print(f"    {ch.describe()}")
        print()

    rule("⑥ 落盘")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "llm_rewrite_report.md"
    parts = [
        "# 真实模型改写报告",
        "",
        f"- 模型：`{cfg.model}`",
        f"- 端点：`{cfg.base_url}`",
        "",
        "## E4 降重",
        "",
        report.to_markdown(),
        "",
        "## 保真校验的对照实验",
        "",
        f"- 原文：{unit_text}",
        f"- 故意改坏数字的候选：{UNFAITHFUL}",
        f"- 校验结论：{fid.summary()}",
        "",
        "## E5 降AI",
        "",
        deai_report.to_markdown(),
        "",
        "> ⚠️ 本报告由**真实模型**生成，但样本只有 2 段。",
        "> 它证明的是「链路通了、校验在起作用」，**不是**「模型改写质量达标」。",
        "",
    ]
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"  reports/llm_rewrite_report.md")

    rule("要点")
    print(
        "  1. **密钥只从环境变量读**，`repr` 已脱敏 —— 打印配置不泄露；\n"
        "  2. **提示词是软约束，校验器是硬闸门** —— 模型不听话也拦得住；\n"
        "  3. **失败向上抛**，不吞成空列表（否则会被误读成「模型认为不用改」）；\n"
        "  4. **E4 与 E5 共用一个改写器** —— 只是策略不同，不另写一套提示词；\n"
        "  5. 无密钥时**优雅跳过**，整条链路仍可跑（退化为 StubRewriter）。"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
