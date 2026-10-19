#!/usr/bin/env python3
"""E4 改写通过率评测：提示词 v1/v2 × 单次/重试，**带配对检验**。

    python eval/run_rewrite_eval.py                 # 默认 6 片段 × 2 次
    python eval/run_rewrite_eval.py --trials 1 --limit 2   # 快速试跑

## 要回答的问题

实测发现单次调用通过率约 50%（连续重叠 7~21 字波动）。两个可能的手段：

| 手段 | 配置 | 想验证什么 |
|---|---|---|
| **提示词迭代** | A → B | 加「结构要求」能不能提高通过率 |
| **重试** | B → C | `max_attempts` 从 1 提到 3 能不能提高通过率 |

A→B 单独隔离提示词的作用，B→C 单独隔离重试的作用。
**这样才能说清"提升来自哪里"**，而不是笼统说"我改了两个地方，变好了"。

## 为什么用 McNemar 而不是 Wilcoxon

通过率是 0/1 数据。真正携带信息的是**不一致对**
（A 通过而 B 不通过 / 反之），两边都通过的样本对"哪个更好"没有发言权。
连续型检验会把这少数翻转摊薄到全部样本上，功效低得多。
详见 `copilot.stats.mcnemar_test`。

## 成本

默认配置约 44 次模型调用（每次 10–30 秒），总计 10–15 分钟。
``--limit`` 与 ``--trials`` 可以缩小规模。

输出：`reports/rewrite_pass_rate.md`
"""

from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass, field
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

from copilot.llm import LlmConfig, LlmError, load_env  # noqa: E402
from copilot.quality.report_parse import FlagType, build_checklist  # noqa: E402
from copilot.quality.rewrite import LlmRewriter, SuggestionStatus, run_rewrite  # noqa: E402
from copilot.stats import compare_proportions  # noqa: E402

OUT = STAGE_DIR / "reports" / "rewrite_pass_rate.md"

#: 6 个真实风格的中文学术句子（都是"该被改写"的类型：方法/结果/方案描述）
FRAGMENTS: tuple[str, ...] = (
    "本文提出了一种基于图神经网络的推荐方法，并在三个数据集上验证了效果。",
    "实验结果表明，该模型在准确率上相比基线提升了 8.3%，同时推理延迟降低了 12.5%。",
    "我们采用两阶段训练策略，先在大规模语料上预训练，再在目标领域上微调。",
    "为了验证方法的有效性，本文在公开数据集上进行了消融实验，分析了各模块的贡献。",
    "针对长尾用户的行为稀疏问题，本文引入了一种基于元学习的冷启动方案。",
    "该方法在保持精度的同时显著降低了计算开销，适合部署在资源受限的设备上。",
)


@dataclass(frozen=True)
class Config:
    """一个待评估的配置。"""

    key: str
    label: str
    structured_guidance: bool
    max_attempts: int
    note: str = ""


CONFIGS: tuple[Config, ...] = (
    Config("A", "v1 提示词 + 单次", structured_guidance=False, max_attempts=1,
           note="基线（当前实现）"),
    Config("B", "v2 提示词 + 单次", structured_guidance=True, max_attempts=1,
           note="只改提示词"),
    Config("C", "v2 提示词 + 重试 3 次", structured_guidance=True, max_attempts=3,
           note="提示词 + 重试"),
)


@dataclass
class Trial:
    config: str
    fragment_index: int
    trial_index: int
    passed: bool
    status: str
    run_overlap: int = -1
    containment: float = -1.0
    fidelity_ok: bool = False
    elapsed: float = 0.0
    error: str = ""


@dataclass
class RunResult:
    trials: list[Trial] = field(default_factory=list)
    calls: int = 0

    def by_config(self, key: str) -> list[Trial]:
        return [t for t in self.trials if t.config == key]


def _doc_for(fragment: str) -> str:
    """把片段包成一份最小文档，让 ``build_checklist`` 能定位到它。"""
    return f"# 论文\n\n## 方法\n\n{fragment}\n"


def run_one(cfg: Config, fragment: str, fragment_index: int, trial_index: int,
            llm: LlmConfig) -> Trial:
    """跑一次改写任务，返回结果。**异常记成失败，不中断评测。**"""
    rewriter = LlmRewriter(llm, structured_guidance=cfg.structured_guidance)
    started = time.time()
    try:
        checklist = build_checklist(
            _doc_for(fragment), fragment, flag_type=FlagType.DUPLICATE
        )
        report = run_rewrite(checklist, rewriter=rewriter, max_attempts=cfg.max_attempts)
    except LlmError as exc:
        return Trial(cfg.key, fragment_index, trial_index, False, "llm_error",
                     elapsed=time.time() - started, error=str(exc)[:120])
    except Exception as exc:  # noqa: BLE001
        return Trial(cfg.key, fragment_index, trial_index, False, "exception",
                     elapsed=time.time() - started, error=f"{type(exc).__name__}: {exc}"[:120])

    elapsed = time.time() - started
    if not report.suggestions:
        return Trial(cfg.key, fragment_index, trial_index, False, "no_unit", elapsed=elapsed)

    s = report.suggestions[0]
    status = getattr(s.status, "value", str(s.status))
    passed = s.status == SuggestionStatus.ACCEPTED
    effect = s.effect
    return Trial(
        config=cfg.key,
        fragment_index=fragment_index,
        trial_index=trial_index,
        passed=passed,
        status=status,
        run_overlap=(effect.after_run if effect else -1),
        containment=(effect.after_containment if effect else -1.0),
        fidelity_ok=bool(s.fidelity.ok) if s.fidelity else False,
        elapsed=elapsed,
    )


def main() -> int:
    trials_per_fragment = 2
    limit = len(FRAGMENTS)
    only: tuple[str, ...] = ()

    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--trials" and i + 1 < len(args):
            trials_per_fragment = max(1, int(args[i + 1]))
        if a == "--limit" and i + 1 < len(args):
            limit = max(1, int(args[i + 1]))
        if a == "--configs" and i + 1 < len(args):
            only = tuple(x.strip().upper() for x in args[i + 1].split(","))

    load_env()
    llm = LlmConfig.from_env()
    fragments = FRAGMENTS[:limit]
    configs = [c for c in CONFIGS if not only or c.key in only]
    total = len(fragments) * len(configs) * trials_per_fragment

    print("=" * 78)
    print("E4 改写通过率评测")
    print("=" * 78)
    print(f"  模型      : {llm.model}")
    print(f"  片段 × 重复: {len(fragments)} × {trials_per_fragment} = {len(fragments)*trials_per_fragment} 个任务/配置")
    print(f"  配置      : {', '.join(f'{c.key}({c.label})' for c in configs)}")
    print(f"  总任务数  : {total}（预计 {total*25//60}–{total*45//60} 分钟）")
    print()

    result = RunResult()
    done = 0
    for cfg in configs:
        print(f"── 配置 {cfg.key}：{cfg.label}（{cfg.note}）")
        for fi, frag in enumerate(fragments):
            for ti in range(trials_per_fragment):
                done += 1
                t = run_one(cfg, frag, fi, ti, llm)
                result.trials.append(t)
                mark = "✅" if t.passed else "❌"
                extra = ""
                if t.run_overlap >= 0:
                    extra = f"重叠{t.run_overlap:2d}字/包含{t.containment:4.0%}"
                print(
                    f"   [{done:2d}/{total}] {cfg.key}-片段{fi+1}-第{ti+1}次 "
                    f"{mark} {t.status:18} {extra} {t.elapsed:5.1f}s"
                    + (f"  ⚠ {t.error}" if t.error else "")
                )
        by = result.by_config(cfg.key)
        rate = sum(1 for t in by if t.passed) / len(by) if by else 0.0
        print(f"   → {cfg.key} 通过率 {rate:.0%}（{sum(1 for t in by if t.passed)}/{len(by)}）\n")

    write_report(result, configs, fragments, trials_per_fragment, llm)
    return 0


def _vec(result: RunResult, key: str) -> list[bool]:
    """按 (片段, 试次) 顺序取出该配置的布尔序列。

    **顺序必须稳定**，否则两个配置的向量对不上，McNemar 的配对就失效了。
    """
    rows = sorted((t.fragment_index, t.trial_index, t.passed) for t in result.by_config(key))
    return [passed for _fi, _ti, passed in rows]


def write_report(result: RunResult, configs, fragments, trials_per_fragment, llm) -> None:
    lines: list[str] = [
        "# E4 改写通过率评测",
        "",
        f"- 模型：`{llm.model}`",
        f"- 规模：{len(fragments)} 个片段 × {trials_per_fragment} 次重复 × {len(configs)} 个配置",
        f"- 达标定义：**保真通过 且** 连续重叠 ≤ 12 字 且 n-gram 包含 ≤ 35%",
        "",
        "## 通过率",
        "",
        "| 配置 | 说明 | 通过 | 通过率 | 平均耗时 |",
        "|---|---|---|---|---|",
    ]
    for cfg in configs:
        by = result.by_config(cfg.key)
        passed = sum(1 for t in by if t.passed)
        rate = passed / len(by) if by else 0.0
        avg = sum(t.elapsed for t in by) / len(by) if by else 0.0
        lines.append(f"| **{cfg.key}** {cfg.label} | {cfg.note} | {passed}/{len(by)} | **{rate:.0%}** | {avg:.1f}s |")

    # ── 配对检验 ──
    comparisons = [("A", "B", "提示词迭代的作用"), ("B", "C", "重试的作用"), ("A", "C", "两者合计")]
    available = {c.key for c in configs}
    lines += ["", "## 配对检验（McNemar + bootstrap CI）", ""]
    lines += ["| 比较 | 通过率 A→B | Δ (95% CI) | 翻转 (b/c) | p (McNemar) | 结论 |", "|---|---|---|---|---|---|"]

    stats_rows = []
    for a, b, desc in comparisons:
        if a not in available or b not in available:
            continue
        va, vb = _vec(result, a), _vec(result, b)
        n = min(len(va), len(vb))
        r = compare_proportions(va[:n], vb[:n], label_a=a, label_b=b)
        stats_rows.append((a, b, desc, r))
        if r["significant"]:
            sig = "**显著提升**" if r["diff"] > 0 else "**显著变差**"
        elif r["ci_excludes_zero"]:
            sig = "CI 不含 0 但 p≥0.05（样本太少）"
        else:
            sig = "不显著"
        lines.append(
            f"| {a} → {b}（{desc}） | {r['pass_rate_a']:.0%} → {r['pass_rate_b']:.0%} | "
            f"{r['diff']:+.0%} [{r['diff_ci_low']:+.0%}, {r['diff_ci_high']:+.0%}] | "
            f"{r['b_only_a']}/{r['b_only_b']} | {r['mcnemar_p']:.4f} | {sig} |"
        )

    # ── 逐条明细 ──
    lines += ["", "## 逐任务明细", "", "| 配置 | 片段 | 试次 | 结论 | 状态 | 重叠字数 | n-gram 包含 | 保真 |", "|---|---|---|---|---|---|---|---|"]
    for t in result.trials:
        ov = "—" if t.run_overlap < 0 else f"{t.run_overlap}"
        ct = "—" if t.containment < 0 else f"{t.containment:.0%}"
        lines.append(
            f"| {t.config} | {t.fragment_index+1} | {t.trial_index+1} | "
            f"{'✅' if t.passed else '❌'} | `{t.status}` | {ov} | {ct} | "
            f"{'✅' if t.fidelity_ok else '❌'} |"
        )

    # ── 结论 ──
    lines += ["", "## 结论", ""]
    for a, b, desc, r in stats_rows:
        if r["significant"]:
            direction = "显著有效" if r["diff"] > 0 else "显著变差"
            lines.append(
                f"- **{desc}{direction}**：通过率 {r['pass_rate_a']:.0%} → {r['pass_rate_b']:.0%}，"
                f"Δ={r['diff']:+.0%} [{r['diff_ci_low']:+.0%}, {r['diff_ci_high']:+.0%}]，"
                f"p={r['mcnemar_p']:.4f}"
            )
        elif r["ci_excludes_zero"]:
            lines.append(
                f"- **{desc}方向明确但统计不显著**：Δ={r['diff']:+.0%}，"
                f"CI [{r['diff_ci_low']:+.0%}, {r['diff_ci_high']:+.0%}] 不含 0，"
                f"但 McNemar p={r['mcnemar_p']:.4f} ≥ 0.05 —— **翻转对数太少，不足以断言**"
            )
        else:
            lines.append(
                f"- **{desc}不显著**：Δ={r['diff']:+.0%} "
                f"[{r['diff_ci_low']:+.0%}, {r['diff_ci_high']:+.0%}]，p={r['mcnemar_p']:.4f}"
            )

    lines += [
        "",
        "## ⚠️ 这份评测的边界（必须一起读）",
        "",
        f"- **样本很小**：{len(fragments)} 个片段 × {trials_per_fragment} 次，"
        "置信区间会很宽，只能检出较大的提升；",
        "- **只有 1 个模型**：换模型不一定成立；",
        "- **片段是我自己选的**：都是「该被改写」的典型句，不含简单句与超长句，"
        "不能代表全文分布；",
        "- **通过率 ≠ 改写质量**：达标只说明「够不像原文且没改坏事实」，"
        "不说明改得通顺、学术性够。后者需要人工评分，本评测不做。",
        "",
        "> 结论的正确用法：**在同一模型、同一类片段上比较两个配置**，"
        "而不是把通过率当成「这个改写器有 X% 好用」。",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")

    # 原始数据一并落盘：改判据时不必重跑（每次重跑要花十几分钟与几十次调用）
    import json

    raw = {
        "model": llm.model,
        "trials_per_fragment": trials_per_fragment,
        "fragments": list(fragments),
        "trials": [
            {
                "config": t.config,
                "fragment_index": t.fragment_index,
                "trial_index": t.trial_index,
                "passed": t.passed,
                "status": t.status,
                "run_overlap": t.run_overlap,
                "containment": t.containment,
                "fidelity_ok": t.fidelity_ok,
                "elapsed": round(t.elapsed, 2),
                "error": t.error,
            }
            for t in result.trials
        ],
        "comparisons": [
            {"a": a, "b": b, "desc": desc, **r} for a, b, desc, r in stats_rows
        ],
    }
    (OUT.with_suffix(".json")).write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=" * 78)
    print(f"  报告已写入：{OUT.relative_to(SYSTEM_DIR)}")
    for a, b, desc, r in stats_rows:
        print(f"  {a}→{b} ({desc}): {r['pass_rate_a']:.0%} → {r['pass_rate_b']:.0%}  "
              f"Δ={r['diff']:+.0%} [{r['diff_ci_low']:+.0%},{r['diff_ci_high']:+.0%}] "
              f"p={r['mcnemar_p']:.4f}")
    print("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
