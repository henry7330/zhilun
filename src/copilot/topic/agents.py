"""A2 可行性评估 与 A3 开题报告 —— 两个**需要模型**的节点。

## 为什么这两件事不能确定性实现

「这个题目三个月做得完吗」没有任何可计算的定义。
即使强行写成规则，规则本身也是拍脑袋 —— **那比承认需要模型更糟**，
因为它会把主观判断伪装成客观结论。

所以本模块的做法是：**有模型用模型，没模型给骨架，并明确标出这是骨架**。

## ★ A2 的关键设计：必须给 blockers 而不是总评

「可行性中等」这种结论**没有决策价值**。

> 「需要标注 1 万条数据，你手上没有」—— 这句才有用：它告诉你卡在哪。

所以 ``Feasibility.blockers``（阻断性风险）是核心字段，``level`` 只是辅助。
没有模型时，骨架会给出**该问哪些问题**，而不是编一个 level。

## A3 为什么必须走证据绑定

开题报告的「国内外研究现状」一节**天然是引用密集区**。
不绑定证据，模型会编造文献 —— 这正是本系统存在的第一理由。

本节点没有模型时只出章节骨架；有模型时，生成的每一句都要过
``copilot.evidence`` 的绑定与校验（四态判定）。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from copilot.topic.models import Feasibility, Proposal, ProposalSection

__all__ = [
    "assess_feasibility",
    "generate_proposal",
    "FeasibilityJudge",
    "ProposalGenerator",
    "PROPOSAL_SECTIONS",
]

#: 评审者签名：(题目, 条件) → 可行性要点
FeasibilityJudge = Callable[[str, Mapping[str, Any]], dict[str, Any]]

#: 生成器签名：(章节标题, 要点) → 正文
ProposalGenerator = Callable[[str, list[str]], str]


#: 开题报告的标准章节
PROPOSAL_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("研究背景与意义", ("这个方向为什么现在值得做", "解决了什么问题")),
    ("国内外研究现状", ("分几派？各派代表工作", "分歧在哪", "**每条陈述都要能落到具体文献**")),
    ("研究内容", ("拆成 2–4 个可独立验证的子问题",)),
    ("研究方法与技术路线", ("用什么方法", "为什么不用现成方案")),
    ("进度安排", ("按周/月给出里程碑", "留出写作与返修时间")),
    ("预期成果", ("可交付物：代码 / 数据 / 论文",)),
)

#: 没有模型时，A2 给出的「该自己回答的问题」。
#:
#: 刻意写成**具体问题**而不是空泛提示 —— 「数据可得性如何」没人回答得了，
#: 「数据是否真的能拿到（不是『网上应该有』）」才会让人停下来想一下。
_SKELETON_RISKS: tuple[str, ...] = (
    "数据是否真的能拿到？注意区分「确实有」和「网上应该有」",
    "最耗时的一步是什么？大概需要多久？",
    "如果核心假设不成立，这个题目还能不能做？",
)


def assess_feasibility(
    topic: str,
    *,
    conditions: Mapping[str, Any] | None = None,
    judge: FeasibilityJudge | None = None,
) -> Feasibility:
    """评估可行性。

    ``conditions``：``{数据, 时间, 技术栈, ...}`` 之类的用户条件。
    ``judge`` 为 ``None`` 时返回**骨架**：给出该问的问题，不编造结论。
    """
    if judge is None:
        return Feasibility(
            level="unknown",
            blockers=["**未接入模型，无法评估可行性** —— 下面是该自己回答的问题"],
            risks=list(_SKELETON_RISKS),
            suggestions=[f"题目：{topic}"]
            + [f"已知条件 {k}：{v}" for k, v in (conditions or {}).items()],
            skeleton=True,
        )

    raw = judge(topic, dict(conditions or {})) or {}
    return Feasibility(
        level=str(raw.get("level", "unknown")),
        blockers=[str(x) for x in raw.get("blockers") or []],
        risks=[str(x) for x in raw.get("risks") or []],
        suggestions=[str(x) for x in raw.get("suggestions") or []],
        skeleton=False,
    )


def generate_proposal(
    title: str,
    *,
    signals: Any = None,
    generator: ProposalGenerator | None = None,
) -> Proposal:
    """生成开题报告。

    ``generator`` 为 ``None`` 时只出**章节骨架 + 写作要点**。
    """
    sections: list[ProposalSection] = []
    for name, points in PROPOSAL_SECTIONS:
        body = ""
        if generator is not None:
            try:
                body = generator(name, list(points)) or ""
            except Exception:  # noqa: BLE001 - 单节失败退化为骨架
                body = ""
        sections.append(ProposalSection(title=name, points=list(points), body=body))

    notes: list[str] = []
    if generator is None:
        notes.append("未接入模型 —— 这是**章节骨架**，正文需人工或模型撰写")
    if signals is not None:
        gaps = getattr(signals, "gaps", []) or []
        if gaps:
            notes.append(
                f"可作为「研究现状与空白」素材的候选空白 {len(gaps)} 条（均需人工核实）"
            )

    return Proposal(
        title=title,
        sections=sections,
        skeleton=generator is None,
        notes=notes,
    )
