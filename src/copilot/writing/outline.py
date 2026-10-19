"""D1 大纲生成：模板骨架 + 要点分配 + 篇幅分配。

## 为什么模板骨架必须确定性生成

学校要求是**硬约束**（"必须有文献综述章""结论不少于 500 字"）。
把硬约束交给模型 = 每次结果不同 = 有时漏章。

所以：**章节骨架由模板直接展开**，模型只负责"每节写哪几个要点"。
没有模型时要点留空，大纲会如实报"章节没有分配要点"。

## 为什么要点要由规则分配（而不是让模型随便想）

``ResearchDesign`` 里已经有结构化的东西：
基线、指标、变量、风险。这些**天然对应到具体章节**：

| 设计里的东西 | 该出现在哪 |
|---|---|
| 方法、路线比选 | 方法章 |
| 基线、指标 | 实验章 |
| 变量（自变量/因变量/控制变量） | 方法章 + 实验章 |
| 风险 | 讨论 / 局限 |

规则分配能让"研究设计写了什么"和"论文写了什么"**对得上** ——
这是写论文最常见的问题之一（设计了一套，写出来是另一套）。

## 篇幅分配

按模板的字数下限，用**权重**分配到各章：
摘要少、方法多、实验多、结论少。

不分配会出现"绪论写了 5000 字、方法只有 300 字"——
这是学生论文最常见的结构失衡，而 E6 的 REV003 只能事后发现。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from copilot.quality.format.template import Template, default_template
from copilot.writing.models import Outline, OutlineSection

__all__ = ["build_outline", "SECTION_WEIGHTS", "assign_points"]


#: 各章的篇幅权重（相对值）。
#:
#: 依据：正文类章节（方法/实验）需要展开论述，而摘要/结论/参考文献
#: 有天然上限。这些权重是**先验设定**，没有数据支撑 ——
#: 但"分配过"远好于"不分配"，后者必然导致结构失衡。
SECTION_WEIGHTS: dict[str, float] = {
    "摘要": 0.6,
    "abstract": 0.6,
    "绪论": 1.2,
    "引言": 1.2,
    "前言": 1.0,
    "文献综述": 1.3,
    "相关工作": 1.2,
    "研究现状": 1.2,
    "研究方法": 1.5,
    "方法": 1.5,
    "实验与分析": 1.6,
    "实验": 1.6,
    "结果与分析": 1.4,
    "讨论": 1.0,
    "结论": 0.8,
    "总结与展望": 0.8,
    "参考文献": 0.3,
    "致谢": 0.3,
}

#: 规则分配时使用的章节名关键词 → 从设计里取哪些要点
_POINT_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("方法", "研究方法", "技术路线"), "method"),
    (("实验", "结果", "评估", "验证"), "experiment"),
    (("讨论", "局限", "不足"), "risk"),
)


def _weight(title: str) -> float:
    t = title.strip().lower()
    for key, w in SECTION_WEIGHTS.items():
        if key in t:
            return w
    return 1.0


def assign_points(title: str, design: Mapping[str, Any] | None) -> list[str]:
    """按章节名把 ``ResearchDesign`` 里的内容分配成要点。

    ``design`` 用 ``ResearchDesign.to_dict()`` 的形状（避免 importing C 阶段，
    让 D 阶段对 C 阶段**零依赖** —— 阶段之间本来就不该互相 import）。
    """
    if not design:
        return []

    points: list[str] = []
    t = title.strip().lower()

    for keywords, kind in _POINT_RULES:
        if not any(k in t for k in keywords):
            continue
        if kind == "method":
            if design.get("method"):
                points.append(f"技术路线：{design['method']}")
            for alt in design.get("alternatives") or []:
                points.append(f"比选方案：{alt}")
            variables = design.get("variables") or {}
            for name in ("independent", "dependent", "controlled"):
                if variables.get(name):
                    points.append(f"{name} 变量：{'、'.join(variables[name])}")
        elif kind == "experiment":
            if design.get("baselines"):
                points.append(f"对照方法：{'、'.join(design['baselines'])}")
            for m in design.get("metrics") or []:
                name = m.get("name") if isinstance(m, dict) else str(m)
                points.append(f"评价指标：{name}")
            n = (design.get("sample_size") or {}).get("n")
            if n:
                points.append(f"样本量：{n}")
        elif kind == "risk":
            for r in design.get("risks") or []:
                points.append(f"局限：{r}")

    return points


def build_outline(
    *,
    title: str,
    template: Template | None = None,
    design: Mapping[str, Any] | None = None,
    required_sections: Sequence[str] | None = None,
) -> Outline:
    """按模板展开大纲，并把设计内容分配到对应章节。

    ``required_sections`` 可覆盖模板（便于单测与演示用更小的集合）。
    """
    tpl = template or default_template()
    sections = list(required_sections or tpl.required_sections)

    total_min = tpl.min_body_words or 0
    weights = [_weight(s) for s in sections]
    weight_sum = sum(weights) or 1.0

    out_sections: list[OutlineSection] = []
    for name, w in zip(sections, weights, strict=True):
        words = int(round(total_min * w / weight_sum)) if total_min else 0
        out_sections.append(
            OutlineSection(
                title=name,
                level=1,
                points=assign_points(name, design),
                target_words=words,
            )
        )

    notes: list[str] = []
    if not total_min:
        notes.append("模板没有字数下限，因此未分配目标字数")
    if not design:
        notes.append("没有研究设计，要点为空 —— 先跑 C 阶段或手动填写")

    outline = Outline(title=title, sections=out_sections, notes=notes)
    if not design:
        outline.skeleton = True
    return outline
