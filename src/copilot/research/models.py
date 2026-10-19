"""C 阶段的数据模型：研究设计、观测、比较结果。

## 一个刻意的设计：`baselines` 不强制在构造时非空

设计文档里写的是"没有基线就无法构造"。落地时改成 **validate() + 报告阻断项**，
原因是：

- 无模型时我们**给不出真实基线**，只能给占位。构造时直接抛异常会让
  "无模型的骨架路径"彻底走不通；
- 更重要的是 —— **让问题变成用户看得见的报告项**，比抛一个异常更有用。
  用户看到「❌ C1 缺少对照组」才会去补；抛异常他就只看到一堆栈。

所以约束没有放松，只是**从崩溃变成了阻断项**：
`ResearchDesign.validate()` 返回问题清单，阶段报告据此判为不可通过。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "Metric",
    "ResearchDesign",
    "Observation",
    "Comparison",
    "StatsResult",
    "PLACEHOLDER_BASELINE",
]

#: 骨架里用的占位基线。**刻意让它显眼**，validate() 会把它当问题报出来。
PLACEHOLDER_BASELINE = "（待填：对照方法）"

_CN_TO_EN = {"J": "article", "M": "book", "C": "inproceedings", "D": "phdthesis"}


@dataclass(frozen=True)
class Metric:
    """一个评价指标。

    ``better`` 必须显式给出 —— 不知道"越大越好还是越小越好"，
    后面的结论方向就无从判断（准确率越大越好，延迟越小越好，
    搞反了会把结论说反）。
    """

    name: str
    better: Literal["higher", "lower"] = "higher"
    rationale: str = ""

    def direction_word(self) -> str:
        return "提升" if self.better == "higher" else "降低"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "better": self.better, "rationale": self.rationale}


@dataclass
class ResearchDesign:
    """研究设计。这是 C1 的产出，也是 C2 的输入。"""

    question: str
    method: str = ""
    #: 比选过的其他路线 + 为什么没选。**只有一条路线等于没做设计。**
    alternatives: list[str] = field(default_factory=list)
    #: 对照方法。**没有对照组的结论不成立。**
    baselines: list[str] = field(default_factory=list)
    variables: dict[str, list[str]] = field(
        default_factory=lambda: {"independent": [], "dependent": [], "controlled": []}
    )
    metrics: list[Metric] = field(default_factory=list)
    sample_size: dict[str, Any] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)
    #: 骨架模式标记：无模型时置 True，validate() 会据此放宽"必须有内容"的要求
    skeleton: bool = False

    def validate(self) -> list[str]:
        """返回设计上的问题清单（空 = 合格）。"""
        problems: list[str] = []

        if not self.method.strip():
            problems.append("缺少方法描述")
        if not self.alternatives:
            problems.append("缺少路线比选（只给一条路线等于没做设计）")
        if not self.baselines or PLACEHOLDER_BASELINE in self.baselines:
            problems.append("缺少对照组（没有基线的比较不成立）")
        if not self.metrics:
            problems.append("缺少评价指标")
        for m in self.metrics:
            if not m.rationale:
                problems.append(f"指标「{m.name}」没说明为什么选它")
        if not self.sample_size.get("n"):
            problems.append("缺少样本量及其依据")
        if not self.risks:
            problems.append("缺少设计层面的风险说明")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "method": self.method,
            "alternatives": list(self.alternatives),
            "baselines": list(self.baselines),
            "variables": dict(self.variables),
            "metrics": [m.to_dict() for m in self.metrics],
            "sample_size": dict(self.sample_size),
            "risks": list(self.risks),
            "skeleton": self.skeleton,
            "problems": self.validate(),
        }


@dataclass(frozen=True)
class Observation:
    """一次**配对**观测：同一个样本在两组配置下各一个值。"""

    unit: str
    a: float
    b: float


@dataclass
class Comparison:
    """一次配对比较的完整结果（三个量都要报）。"""

    metric: str
    label_a: str
    label_b: str
    n: int
    mean_a: float
    mean_b: float
    diff: float
    diff_ci_low: float
    diff_ci_high: float
    p_raw: float
    p_adjusted: float | None = None
    effect_size: float = 0.0
    significant: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "n": self.n,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "diff": round(self.diff, 4),
            "diff_ci": [round(self.diff_ci_low, 4), round(self.diff_ci_high, 4)],
            "p_raw": round(self.p_raw, 6),
            "p_adjusted": None if self.p_adjusted is None else round(self.p_adjusted, 6),
            "effect_size": round(self.effect_size, 3),
            "significant": self.significant,
        }

    def describe(self) -> str:
        return (
            f"{self.metric}：{self.label_a} {self.mean_a:.3f} → {self.label_b} {self.mean_b:.3f}，"
            f"Δ={self.diff:+.4f} [{self.diff_ci_low:+.4f},{self.diff_ci_high:+.4f}]，"
            f"dz={self.effect_size:+.2f}，p={self.p_raw:.4f}"
            + (f"（校正后 {self.p_adjusted:.4f}）" if self.p_adjusted is not None else "")
        )


@dataclass
class StatsResult:
    """C2 的产出。"""

    comparisons: list[Comparison] = field(default_factory=list)
    correction: str = ""
    n_tests: int = 0
    conclusion: str = ""
    limitations: list[str] = field(default_factory=list)

    @property
    def significant(self) -> list[Comparison]:
        return [c for c in self.comparisons if c.significant]

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "correction": self.correction,
            "n_tests": self.n_tests,
            "conclusion": self.conclusion,
            "limitations": list(self.limitations),
        }

    def to_markdown(self) -> str:
        lines = [
            "# 数据分析结果",
            "",
            f"- 比较次数：**{len(self.comparisons)}**"
            + (f"（已做 {self.correction} 校正）" if self.correction else ""),
            "",
            "| 指标 | A | B | Δ | 95% CI | dz | p | p(校正) | 显著 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c in self.comparisons:
            lines.append(
                f"| {c.metric} | {c.mean_a:.3f} | {c.mean_b:.3f} | {c.diff:+.4f} | "
                f"[{c.diff_ci_low:+.4f}, {c.diff_ci_high:+.4f}] | {c.effect_size:+.2f} | "
                f"{c.p_raw:.4f} | "
                f"{'—' if c.p_adjusted is None else f'{c.p_adjusted:.4f}'} | "
                f"{'**是**' if c.significant else '否'} |"
            )
        if self.conclusion:
            lines += ["", "## 结论", "", self.conclusion]
        if self.limitations:
            lines += ["", "## 局限", ""]
            lines += [f"- {x}" for x in self.limitations]
        lines.append("")
        return "\n".join(lines)
