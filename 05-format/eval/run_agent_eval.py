#!/usr/bin/env python3
"""节点层评测：E1 / E1H / E2 / F1 的**精确率 / 召回率 / F1**（带 bootstrap CI）。

    python eval/run_agent_eval.py

## 为什么要这个

在此之前，只有**检索层**和**记忆层**有真正的对照评测
（`run_retrieval_eval.py` / `run_memory_benchmark.py`）。
E1/E3/E6/F2 只有单元测试 + 演示输出 ——
**单元测试证明"代码按设计跑"，不证明"检得准"**。

这个脚本补的正是那一块：用带标准答案的用例量**精确率与召回率**。

## 指标口径

- **TP**：期望的规则码真的被报出来
- **FN**：期望的没报（漏检）
- **FP**：报了不该报的（误报）
- Precision = TP / (TP + FP)，Recall = TP / (TP + FN)

**干净用例是 precision 的关键**：如果所有用例都"应该报点什么"，
precision 会永远是 1.0 —— 那等于没测误报。

## ⚠️ 这套评测的边界

用例是**按设计写的**，不是他人标注的真实论文集。
所以它验证的是"实现符合设计"，**不是"在真实论文上够用"**。
报告里必须把这条写在显眼位置。
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

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

sys.path.insert(0, str(_HERE.parent))

from agent_cases import AGENTS_COVERED, CASES  # noqa: E402
from copilot.quality.format import check_format  # noqa: E402
from copilot.quality.format.template import Template  # noqa: E402
from copilot.delivery.prcheck import check_submission  # noqa: E402
from copilot.quality.report_parse import FlagType, build_checklist  # noqa: E402
from copilot.quality.review import review_text  # noqa: E402
from copilot.stats import DEFAULT_SEED  # noqa: E402

import numpy as np  # noqa: E402

OUT = STAGE_DIR / "reports" / "agent_eval.md"


def _template(options: dict[str, Any]) -> Template | None:
    spec = options.get("template")
    if not spec:
        return None
    return Template(
        name=spec["name"],
        required_sections=spec["required_sections"],
        min_body_words=spec.get("min_body_words"),
    )


# ─────────────────────────── 执行 ───────────────────────────


@dataclass
class CaseResult:
    case_id: str
    agent: str
    desc: str
    actual_codes: list[str] = field(default_factory=list)
    expected_codes: list[str] = field(default_factory=list)
    forbidden_hit: list[str] = field(default_factory=list)
    tp: int = 0
    fn: int = 0
    fp: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def run_case(case: dict) -> CaseResult:
    agent = case["agent"]
    opts = case.get("options") or {}
    inp = case["input"]
    res = CaseResult(case_id=case["id"], agent=agent, desc=case["desc"])

    try:
        if agent == "E1":
            report = check_format(inp["draft"], _template(opts))
            res.actual_codes = sorted({i.code for i in report.issues})

        elif agent == "E2":
            report = review_text(
                inp["draft"], _template(opts), min_section_chars=opts.get("min_section_chars")
            )
            # 只看审稿规则，图表类交给 E1 的用例测，避免重复计分
            res.actual_codes = sorted({i.code for i in report.issues if i.code.startswith("REV")})

        elif agent == "E1H":
            report = check_submission(
                inp["draft"],
                template=_template(opts),
                anonymity_required=bool(opts.get("anonymity_required")),
            )
            # 只看内容卫生五项（它属于 E1，但独立成组计分）
            res.actual_codes = sorted({i.code for i in report.issues if i.code.startswith("PRE")})
            res.extra["can_submit"] = report.can_submit

        elif agent == "F1":
            checklist = build_checklist(inp["draft"], inp["report"], flag_type=FlagType.DUPLICATE)
            located = 0
            unlocated = 0
            for item in checklist.items:
                method = getattr(getattr(item, "method", None), "value", "unlocated")
                if method == "unlocated":
                    unlocated += 1
                else:
                    located += 1
            res.extra["located"] = located
            res.extra["unlocated"] = unlocated

        else:
            res.error = f"未知节点：{agent}"
            return res

    except Exception as exc:  # noqa: BLE001 - 单条用例失败不该中断评测
        res.error = f"{type(exc).__name__}: {exc}"[:160]
        return res

    # ── 计分：规则码类 ──
    if agent in {"E1", "E2", "E1H"}:
        expected = set(case.get("codes_all") or [])
        actual = set(res.actual_codes)
        forbidden = set(case.get("codes_none") or [])
        res.expected_codes = sorted(expected)
        res.forbidden_hit = sorted(forbidden & actual)
        res.tp = len(expected & actual)
        res.fn = len(expected - actual)
        res.fp = len(actual - expected)

    # ── 计分：定位类 ──
    if agent == "F1":
        exp_loc = case.get("located")
        exp_unloc = case.get("unlocated")
        got_loc = res.extra.get("located", 0)
        got_unloc = res.extra.get("unlocated", 0)
        if exp_loc is not None:
            res.tp += min(exp_loc, got_loc)
            res.fn += max(0, exp_loc - got_loc)
            res.fp += max(0, got_loc - exp_loc)
        if exp_unloc is not None:
            res.tp += min(exp_unloc, got_unloc)
            res.fn += max(0, exp_unloc - got_unloc)
            res.fp += max(0, got_unloc - exp_unloc)

    return res


# ─────────────────────────── 汇总 ───────────────────────────


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if tp == 0:
        f1 = 0.0
    elif (precision + recall) > 0 and not (np.isnan(precision) or np.isnan(recall)):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def _bootstrap_prf(
    results: Sequence[CaseResult], *, n_boot: int = 5000, seed: int = DEFAULT_SEED
) -> dict[str, tuple[float, float]]:
    """对用例做 bootstrap，给出 P/R/F1 的 95% 区间。"""
    if not results:
        return {}
    rng = np.random.default_rng(seed)
    n = len(results)
    p_vals: list[float] = []
    r_vals: list[float] = []
    f_vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tp = sum(results[i].tp for i in idx)
        fp = sum(results[i].fp for i in idx)
        fn = sum(results[i].fn for i in idx)
        p, r, f = _prf(tp, fp, fn)
        if not np.isnan(p):
            p_vals.append(p)
        if not np.isnan(r):
            r_vals.append(r)
        f_vals.append(f)

    def ci(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return (float("nan"), float("nan"))
        lo, hi = np.quantile(vals, [0.025, 0.975])
        return (float(lo), float(hi))

    return {"precision": ci(p_vals), "recall": ci(r_vals), "f1": ci(f_vals)}


def main() -> int:
    print("=" * 78)
    print(f"节点层评测（{' / '.join(AGENTS_COVERED)}）")
    print("=" * 78)
    print(f"  用例数：{len(CASES)}（覆盖 {', '.join(AGENTS_COVERED)}）")
    print()

    results = [run_case(c) for c in CASES]

    for r in results:
        mark = "✅" if (r.tp and not r.fn and not r.fp) or (r.tp == 0 and r.fp == 0 and not r.error) else "❌"
        if r.error:
            mark = "💥"
        detail = ""
        if r.agent == "F1":
            detail = f"located={r.extra.get('located')} unlocated={r.extra.get('unlocated')}"
        else:
            detail = f"实际={r.actual_codes or '（无）'}"
        print(f"  {mark} [{r.agent}] {r.case_id:22} TP{r.tp} FN{r.fn} FP{r.fp}  {detail}")
        if r.forbidden_hit:
            print(f"       ⚠ 命中了不该出现的码：{r.forbidden_hit}")
        if r.error:
            print(f"       💥 {r.error}")

    write_report(results)

    print()
    print("=" * 78)
    print("  报告已写入：reports/agent_eval.md")
    for agent in AGENTS_COVERED:
        sub = [r for r in results if r.agent == agent]
        tp = sum(r.tp for r in sub)
        fp = sum(r.fp for r in sub)
        fn = sum(r.fn for r in sub)
        p, rcl, f = _prf(tp, fp, fn)
        print(f"  {agent:3} TP={tp:2} FP={fp:2} FN={fn:2}  "
              f"P={p:.3f} R={rcl:.3f} F1={f:.3f}")
    print("=" * 78)
    return 0


def write_report(results: Sequence[CaseResult]) -> None:
    lines: list[str] = [
        f"# 节点层评测（{' / '.join(AGENTS_COVERED)}）",
        "",
        f"- 用例数：**{len(results)}**",
        f"- 覆盖：{', '.join(AGENTS_COVERED)}",
        "- 指标：精确率（报出的对不对）/ 召回率（该报的有没有漏）/ F1",
        "",
        "## ⚠️ 先说边界",
        "",
        "用例是**按各 Agent 的设计写的**，不是他人标注的真实论文集。",
        "所以这套评测验证的是「**实现符合设计**」，",
        "**不是**「在真实论文上够用」。要后者，需要真实的、独立标注的语料 —— 本项目没有，也不假装有。",
        "",
        "## 总表",
        "",
        "| Agent | 用例 | TP | FP | FN | 精确率 | 召回率 | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for agent in AGENTS_COVERED:
        sub = [r for r in results if r.agent == agent]
        tp = sum(r.tp for r in sub)
        fp = sum(r.fp for r in sub)
        fn = sum(r.fn for r in sub)
        p, rcl, f = _prf(tp, fp, fn)
        ci = _bootstrap_prf(sub)
        p_ci = ci.get("precision", (float("nan"), float("nan")))
        r_ci = ci.get("recall", (float("nan"), float("nan")))
        lines.append(
            f"| **{agent}** | {len(sub)} | {tp} | {fp} | {fn} | "
            f"{_fmt(p)} [{_fmt(p_ci[0])}, {_fmt(p_ci[1])}] | "
            f"{_fmt(rcl)} [{_fmt(r_ci[0])}, {_fmt(r_ci[1])}] | {_fmt(f)} |"
        )

    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    p, rcl, f = _prf(tp, fp, fn)
    lines.append(f"| **合计** | {len(results)} | {tp} | {fp} | {fn} | {_fmt(p)} | {_fmt(rcl)} | {_fmt(f)} |")

    # ── 误报专项 ──
    clean = [r for r in results if not r.expected_codes and r.agent != "E3"]
    lines += [
        "",
        "## 误报专项（干净用例）",
        "",
        "> 没有这一段，精确率会永远是 1.0 —— 那等于没测误报。",
        "",
        "| 用例 | 期望 | 实际报出 | 结论 |",
        "|---|---|---|---|",
    ]
    for r in clean:
        if r.error:
            lines.append(f"| `{r.case_id}` | 无 | （执行失败） | 💥 |")
            continue
        verdict = "✅ 零误报" if r.fp == 0 else f"❌ 误报 {r.fp} 条"
        lines.append(f"| `{r.case_id}` | 无 | {r.actual_codes or '（无）'} | {verdict} |")

    # ── 逐用例 ──
    lines += ["", "## 逐用例明细", "", "| Agent | 用例 | 测什么 | 期望 | 实际 | TP/FN/FP |", "|---|---|---|---|---|---|"]
    for r in results:
        if r.agent == "F1":
            expected = f"located={r.extra.get('_exp_loc', '—')}"
            c = [c for c in CASES if c["id"] == r.case_id][0]
            expected = f"located={c.get('located')} / unlocated={c.get('unlocated')}"
            actual = f"located={r.extra.get('located')} / unlocated={r.extra.get('unlocated')}"
        else:
            expected = r.expected_codes or "（无）"
            actual = r.actual_codes or "（无）"
        verdict = "💥" if r.error else ("✅" if not r.fn and not r.fp else "❌")
        lines.append(
            f"| {r.agent} | `{r.case_id}` | {r.desc} | {expected} | {actual} | "
            f"{r.tp}/{r.fn}/{r.fp} {verdict} |"
        )
        if r.error:
            lines.append(f"| | | 💥 {r.error} | | | |")

    # ── 结论 ──
    lines += ["", "## 结论", ""]
    for agent in AGENTS_COVERED:
        sub = [r for r in results if r.agent == agent]
        tp = sum(r.tp for r in sub)
        fp = sum(r.fp for r in sub)
        fn = sum(r.fn for r in sub)
        p, rcl, f = _prf(tp, fp, fn)
        if fp == 0 and fn == 0:
            lines.append(f"- **{agent}**：{len(sub)} 条用例全部符合预期（零误报、零漏检）")
        else:
            lines.append(
                f"- **{agent}**：P={_fmt(p)} / R={_fmt(rcl)} / F1={_fmt(f)}"
                f"（漏检 {fn}，误报 {fp}）"
            )

    lines += [
        "",
        "## 为什么 E4/E5 不在这份报告里",
        "",
        "E4/E5 的核心不是「检出」而是「改写 + 双校验」，",
        "指标是**通过率**而不是 P/R —— 评测在 `eval/run_rewrite_eval.py`，",
        "报告见 `reports/rewrite_pass_rate.md`。",
        "",
        "## 为什么 A–D 阶段没有评测",
        "",
        "**它们没有实现。** 没有代码就没有可测对象 ——",
        "给一个未实现的阶段编评测数字，是最糟的一种「看起来严谨」。",
        "",
        "它们的评测方案在各自 README 的「验收标准」一节里，",
        "实现后按那份标准建评测集即可。",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")


def _fmt(x: float) -> str:
    return "—" if isinstance(x, float) and np.isnan(x) else f"{x:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
