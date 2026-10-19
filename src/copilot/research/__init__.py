"""C 阶段 · Agent「研究」—— 一个可以独立调用的服务。

## 这个 Agent 做什么

从「我要用方法 M 解决问题 P」到「一套可执行的研究设计 + 数据分析结论」。

```
C1 方法设计 ──▶ ResearchDesign（含比选、基线、指标、样本量、风险）
                     │
C2 数据分析 ──▶ StatsResult（配对检验 + 差值 CI + 效应量 + BH 校正）
```

## 怎么用（独立调用，不需要先跑 A/B 阶段）

```python
from copilot.research import run_research, Observation

report = run_research(
    "图神经网络在长尾用户推荐上是否优于矩阵分解？",
    data={"ndcg@10": [Observation("u1", 0.31, 0.36), Observation("u2", 0.28, 0.33), ...]},
)
print(report.summary_line())
print(report.to_markdown())
```

**没有数据也能调**：C2 会跳过并说明原因，C1 会给设计骨架。
但报告会是 ❌ —— 因为骨架设计本来就不合格（缺基线、缺样本量）。
这是刻意的：**不假装完成**。

## 为什么只有两个节点

研究阶段真正的工作就是"设计"和"分析"两件事。
拆更多节点（数据清洗 / 特征工程 / 调参）对论文辅助没有独立价值 ——
那些是实验执行，不是研究设计。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from copilot.contracts import Finding
from copilot.quality.format.report import Severity
from copilot.research.analysis import ALPHA, analyze_results
from copilot.research.design import DesignPlanner, propose_design
from copilot.research.models import (
    Observation,
    ResearchDesign,
    StatsResult,
)
from copilot.stage import StageNode, ok_outcome, run_stage

__all__ = [
    "run_research",
    "build_research_nodes",
    "ResearchDesign",
    "StatsResult",
    "Observation",
    "propose_design",
    "analyze_results",
]

STAGE_KEY = "C"
CODE_DESIGN = "C1-DESIGN"
CODE_STATS = "C2-STATS"
CODE_NO_BASELINE = "C2-NO-BASELINE"


def build_research_nodes(
    *,
    planner: DesignPlanner | None = None,
    label_a: str = "baseline",
    label_b: str = "method",
) -> tuple[StageNode, ...]:
    """把两个节点装配出来。"""

    # ── C1 方法设计 ──

    def c1_design(state: dict[str, Any]):
        design = propose_design(state["question"], planner=planner)
        findings: list[Finding] = []

        problems = design.validate()
        for p in problems:
            findings.append(
                Finding(
                    node="C1",
                    code=CODE_DESIGN,
                    # ★ 缺失项**永远是阻断项**，骨架也不例外。
                    # 早期写成"骨架只报 WARNING"，结果骨架设计被判为 ✅ 通过 ——
                    # 那等于告诉用户"你的研究设计合格了"，而它只是几个空字段。
                    severity=Severity.ERROR,
                    message=f"设计不完整：{p}",
                    fix="补齐设计要素后再做分析 —— 设计不完备时统计做得再对也没用",
                )
            )

        if design.skeleton:
            findings.append(
                Finding(
                    node="C1",
                    code=CODE_DESIGN,
                    severity=Severity.INFO,
                    message="当前是**设计骨架**（未接入模型，内容待填）",
                    fix="接入模型（planner）或手动填写 method / alternatives / baselines 等字段",
                )
            )

        return ok_outcome(
            findings=findings,
            stats={"problems": len(problems), "skeleton": design.skeleton},
            notes=[f"路线比选 {len(design.alternatives)} 条", f"基线 {len(design.baselines)} 个"],
            patch={"design": design},
        )

    # ── C2 数据分析 ──

    def c2_analysis(state: dict[str, Any]):
        data: Mapping[str, Sequence[Observation]] = state.get("data") or {}
        design: ResearchDesign | None = state.get("design")

        if not data:
            return ok_outcome(
                notes=[
                    "没有提供数据 —— 跳过分析。"
                    "**这不是「分析通过」，而是「没做分析」**。"
                ],
                stats={"analyzed": 0},
            )

        result = analyze_results(data, label_a=label_a, label_b=label_b)
        findings: list[Finding] = []

        # 没有对照组的分析结论不成立 —— 先拦这一条
        if design is not None:
            baseline_problems = [
                p for p in design.validate() if "对照" in p
            ]
            if baseline_problems:
                findings.append(
                    Finding(
                        node="C2",
                        code=CODE_NO_BASELINE,
                        severity=Severity.ERROR,
                        message="设计里没有有效对照组，因此下面的比较结论不成立",
                        fix="先补齐基线方法（C1），再做分析",
                    )
                )

        for c in result.comparisons:
            findings.append(
                Finding(
                    node="C2",
                    code=CODE_STATS,
                    severity=Severity.INFO if c.significant else Severity.WARNING,
                    message=(
                        f"{c.metric}：{c.describe()}"
                        if c.significant
                        else f"{c.metric} 无显著差异（p(校正)={c.p_adjusted:.4f}）"
                    ),
                    fix="" if c.significant else "样本量不足或效应太小，可增大样本或换指标",
                    payload=c.to_dict(),
                )
            )

        if any(abs(c.effect_size) < 0.2 and c.significant for c in result.comparisons):
            findings.append(
                Finding(
                    node="C2",
                    code=CODE_STATS,
                    severity=Severity.WARNING,
                    message="有指标统计显著但效应量 < 0.2（实际意义有限）",
                    fix="表述时说明效应量，不要只写「显著提升」",
                )
            )

        return ok_outcome(
            findings=findings,
            stats={
                "analyzed": len(result.comparisons),
                "significant": len(result.significant),
                "alpha": ALPHA,
            },
            notes=[
                f"BH 校正（{result.n_tests} 次检验）",
                "结论已限定在本次样本范围内",
            ],
            patch={"stats_result": result},
        )

    return (
        StageNode(
            code="C1",
            name="方法设计",
            run=c1_design,
            requires_model=planner is not None,
            description="路线比选 / 基线 / 指标 / 样本量 / 风险",
        ),
        StageNode(
            code="C2",
            name="数据分析",
            run=c2_analysis,
            when=lambda s: bool(s.get("data")),
            description="配对检验 + 差值 CI + 效应量 + BH 校正",
        ),
    )


def run_research(
    question: str,
    *,
    data: Mapping[str, Sequence[Observation]] | None = None,
    planner: DesignPlanner | None = None,
    label_a: str = "baseline",
    label_b: str = "method",
    only: Sequence[str] | None = None,
    on_node: Callable[[str, Any], None] | None = None,
):
    """**C 阶段的对外入口。** 独立调用，不需要先跑 A/B 阶段。

    ``data``：``{指标名: [Observation, ...]}``；不给则跳过 C2。
    """
    nodes = build_research_nodes(planner=planner, label_a=label_a, label_b=label_b)
    return run_stage(
        stage=STAGE_KEY,
        label=question[:48] or "<empty>",
        nodes=nodes,
        state={"question": question, "data": dict(data or {})},
        only=only,
        on_node=on_node,
    )
