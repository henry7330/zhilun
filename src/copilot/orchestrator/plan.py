"""主 Agent 的**规划器**：把一句自然语言请求拆成可执行的步骤。

## 这里为什么用规则而不是直接让模型规划

**先用规则跑通，再换模型** —— 这是本项目一以贯之的做法，理由有三个：

1. **可测**：给一句话，断言 Plan 里有哪几个 step。模型版做不到这种确定性断言；
2. **免费**：规划这一步不该花钱。用户问"帮我看看格式"，先花 3000 token 规划一遍很浪费；
3. **可回退**：模型不可用时系统仍然能用。

模型规划器通过 ``LlmPlanner`` 预留（传入一个 callable 即可），
规则版是它的 **fallback 与对照组**。

## 三层路由怎么落

| 层 | 规则版怎么决定 |
|---|---|
| ① 选阶段 | 关键词命中（`STAGE_CUES`） |
| ② 选节点 | 阶段内部再按关键词收窄（`NODE_CUES`）；**命中不了就整个阶段都跑** |
| ③ 选模型 | 见 `router.py`（按节点是否需要模型 + 用户的模型登记表） |

**节点收窄是"只减不增"**：关键词只能让一个阶段跑得更少，不能让它跑得更多 ——
少跑是省，多跑是错（跑了用户没要求的东西，成本与风险都由用户承担）。

**但依赖必须补齐**：`F2` 要读 `F1` 的清单，`D2` 要读 `D1` 的大纲。
所以收窄之后要按 `NODE_DEPS` 再做一次闭包 —— 否则下游拿不到输入，只会静默出空报告。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from copilot.contracts import STAGES
from copilot.orchestrator.models import REQUIRED_INPUTS, Plan, PlanStep

__all__ = [
    "Planner",
    "RulePlanner",
    "LlmPlanner",
    "STAGE_CUES",
    "NODE_CUES",
    "NODE_DEPS",
    "SOFT_DEPS",
    "close_dependencies",
]

#: 阶段触发词。
#:
#: ★ 刻意**不用**裸的「引用」「参考文献」——它们分别会误触发 D 和 B。
#: 只用足够具体的组合（「引用来源」「参考文献格式」）。
STAGE_CUES: dict[str, tuple[str, ...]] = {
    "A": ("选题", "开题", "题目", "研究方向", "可行性", "研究空白", "创新点", "写什么"),
    "B": ("文献检索", "检索文献", "查文献", "综述", "文献库", "去重", "doi", "找几篇"),
    "C": ("研究设计", "方法设计", "设计方法", "技术路线", "研究方法", "实验设计",
          "统计分析", "显著性", "对照组", "基线", "数据分析", "怎么做实验"),
    "D": ("写作", "大纲", "章节", "正文", "撰写", "图表", "图号", "题注", "引用来源", "引用真实"),
    "E": ("格式", "排版", "字体", "字号", "行距", "页边距", "页眉", "页脚", "著录",
          "参考文献格式", "导出", "latex", "bibtex", "模板", "规范"),
    "F": ("降重", "查重", "重复率", "重复", "ai率", "ai检测", "aigc", "改写", "降ai", "去ai"),
}

#: 阶段内部的节点触发词。命中则**只跑命中那些节点**；全都不命中就跑整阶段。
NODE_CUES: dict[str, dict[str, tuple[str, ...]]] = {
    "A": {
        "A1": ("研究空白", "关键词", "趋势", "热点"),
        "A2": ("可行性", "做得动", "数据拿得到", "难不难", "工作量"),
        "A3": ("开题", "开题报告"),
    },
    "B": {
        "B1": ("找文献", "查文献", "文献库"),
        "B2": ("综述", "文献综述"),
        "B3": ("去重", "doi", "参考文献", "引用管理", "bibtex"),
    },
    "C": {
        "C1": ("方法设计", "设计方法", "研究设计", "实验设计", "技术路线", "路线", "基线"),
        "C2": ("统计", "显著", "数据", "检验", "对照组"),
    },
    "D": {
        "D1": ("大纲", "目录", "章节安排"),
        "D2": ("正文", "撰写", "写一节"),
        "D3": ("图表", "图号", "题注", "插图", "表格编号"),
        "D4": ("引用来源", "引用真实", "引用核对", "引的文献"),
    },
    "E": {
        "E1": ("排版", "字体", "字号", "行距", "页边距", "页眉", "页脚", "规整"),
        "E2": ("术语", "关键词覆盖", "篇幅", "一致性"),
        "E3": ("导出", "latex", "bibtex", "参考文献格式", "著录"),
    },
    "F": {
        "F1": ("定位", "报告", "查重报告", "ai报告"),
        "F2": ("改写", "降ai", "去ai", "重复"),
    },
}

#: **硬依赖**：选了右边没选左边，右边拿不到输入。
#:
#: 这不是"顺序偏好"，是真实的数据依赖：
#: ``D2`` 读 ``state["outline"]``（D1 写的），``F2`` 读 ``state["located"]``（F1 写的）。
NODE_DEPS: dict[str, tuple[str, ...]] = {
    # A2/A3 读 A1 写的 signals
    "A2": ("A1",),
    "A3": ("A1",),
    # B2 读 B1 检索出来的文献
    "B2": ("B1",),
    # D2 读 D1 写的 outline
    "D2": ("D1",),
    # F2 读 F1 写的 located 清单
    "F2": ("F1",),
}

#: **软依赖**：有前者更好（能用上前者的产物），没有也能跑（回退到原始输入）。
#: 软依赖**不自动补齐** —— 因为多跑一个节点就是多一份成本，
#: 用户没要求时不该替他跑。
SOFT_DEPS: dict[str, tuple[str, ...]] = {
    "E2": ("E1",),   # 检查"规整后"的文本；没有就检查原稿
    "E3": ("E1",),   # 导出"规整后"的文本；没有就导出原稿
}


def close_dependencies(stage: str, nodes: Sequence[str]) -> tuple[str, ...]:
    """把依赖闭包补全，并保持本阶段的声明顺序。

    ``("F2",)`` → ``("F1", "F2")``；``("D2",)`` → ``("D1", "D2")``。
    """
    wanted = set(nodes)
    changed = True
    while changed:
        changed = False
        for code in list(wanted):
            for dep in NODE_DEPS.get(code, ()):
                if dep not in wanted:
                    wanted.add(dep)
                    changed = True
    order = [n.code for n in STAGES[stage].nodes]
    return tuple(c for c in order if c in wanted)


class Planner(Protocol):
    """规划器协议：给请求与上下文，产出 Plan。"""

    def __call__(self, request: str, context: Mapping[str, Any]) -> Plan: ...


def _hits(text: str, cues: Sequence[str]) -> list[str]:
    low = text.lower()
    return [c for c in cues if c.lower() in low]


@dataclass
class RulePlanner:
    """关键词规划器。**确定性、免费、可单测。**"""

    #: 把某句话强制映射到某些阶段（用于演示与测试的确定性注入）
    forced_stages: tuple[str, ...] = ()

    def __call__(self, request: str, context: Mapping[str, Any]) -> Plan:
        text = request or ""
        plan = Plan(request=request)

        stages: list[str] = list(self.forced_stages)
        reasons: dict[str, str] = {}
        if not stages:
            for key, cues in STAGE_CUES.items():
                hit = _hits(text, cues)
                if hit:
                    stages.append(key)
                    reasons[key] = "命中关键词：" + "、".join(hit[:3])

        if not stages:
            plan.notes.append(
                "没有匹配到任何阶段 —— 请说得具体一点（如「帮我检查格式」「这段降重」）"
            )
            return plan

        for key in sorted(stages, key="ABCDEF".index):
            stage = STAGES[key]
            node_cues = NODE_CUES.get(key, {})
            picked: list[str] = []
            for node in stage.nodes:
                if _hits(text, node_cues.get(node.code, ())):
                    picked.append(node.code)

            if picked and len(picked) < len(stage.nodes):
                nodes: tuple[str, ...] | None = close_dependencies(key, picked)
                note = "收窄到节点：" + "、".join(nodes)
            else:
                nodes = None          # 全跑
                note = "本阶段全部节点"

            plan.steps.append(
                PlanStep(
                    stage=key,
                    nodes=nodes,
                    reason=(
                        (reasons.get(key, "") + "；" + note).strip("；")
                        if reasons.get(key)
                        else note
                    ),
                )
            )

        # 缺输入检查：缺了就把阶段标出来，让执行器跳过并说明原因
        for key in (s.stage for s in plan.steps):
            for need in REQUIRED_INPUTS.get(key, ()):
                if not str(context.get(need) or "").strip():
                    if need not in plan.missing:
                        plan.missing.append(need)

        if plan.missing:
            plan.notes.append("缺少输入：" + "、".join(plan.missing))
        plan.notes.append(
            "规划规则版：关键词驱动。**它只能决定调什么，不能判断做得对不对。**"
        )
        return plan


@dataclass
class LlmPlanner:
    """用模型规划，**规则版兜底**。

    ``complete`` 是一个 ``(prompt) -> JSON 字符串`` 的可调用对象（接你自己的模型）。
    模型给出非法结果时**不问第二次**，直接回退到规则版 ——
    规划失败应该是廉价且可预期的，不是重试到成功。
    """

    complete: Callable[[str], str]
    fallback: Planner = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.fallback is None:
            object.__setattr__(self, "fallback", RulePlanner())

    def __call__(self, request: str, context: Mapping[str, Any]) -> Plan:
        import json

        prompt = _build_prompt(request, context)
        try:
            raw = self.complete(prompt)
            payload = json.loads(raw)
            steps = [
                PlanStep(
                    stage=str(s["stage"]).upper(),
                    nodes=tuple(s["nodes"]) if s.get("nodes") else None,
                    reason=str(s.get("reason") or "模型规划"),
                )
                for s in payload.get("steps", [])
            ]
        except Exception as exc:  # noqa: BLE001 - 规划失败一律回退
            plan = self.fallback(request, context)
            plan.notes.append(f"模型规划失败（{type(exc).__name__}），已回退到规则版")
            return plan

        legal = [s for s in steps if s.stage in STAGES]
        if not legal:
            plan = self.fallback(request, context)
            plan.notes.append("模型没有给出合法阶段，已回退到规则版")
            return plan

        # 节点合法性 + 依赖闭包：模型的输出也要过校验，不能直接采信
        fixed: list[PlanStep] = []
        for s in legal:
            known = {n.code for n in STAGES[s.stage].nodes}
            nodes = None
            if s.nodes:
                keep = [c for c in s.nodes if c in known]
                nodes = close_dependencies(s.stage, keep) if keep else None
            fixed.append(PlanStep(stage=s.stage, nodes=nodes, reason=s.reason))

        plan = Plan(request=request, steps=fixed)
        plan.notes.append("规划由模型给出，已校验阶段/节点合法性")
        for key in plan.stages:
            for need in REQUIRED_INPUTS.get(key, ()):
                if not str(context.get(need) or "").strip() and need not in plan.missing:
                    plan.missing.append(need)
        return plan


def _build_prompt(request: str, context: Mapping[str, Any]) -> str:
    stages = "\n".join(
        f"- {s.key} {s.name}：节点 " + "、".join(n.code for n in s.nodes)
        for s in STAGES.values()
    )
    available = {
        k: ("有" if str(v or "").strip() else "无")
        for k, v in context.items()
        if k in {"title", "draft", "question", "report", "requirement"}
    }
    return (
        "你是织论（Zhilun）的主 Agent，需要把用户请求拆成执行计划。\n"
        "可用阶段与节点：\n"
        f"{stages}\n\n"
        f"当前已有输入：{available}\n"
        f"用户请求：{request}\n\n"
        "只输出 JSON：{\"steps\": [{\"stage\": \"E\", \"nodes\": [\"E1\"], \"reason\": \"...\"}]}\n"
        "nodes 为 null 表示该阶段全部节点。只选真正需要的阶段。"
    )
