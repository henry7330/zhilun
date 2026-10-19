"""跨阶段 / 跨节点的统一契约。

★ 术语（重要：这里曾经写错两次，已纠正）
----------------------------------------
**一个阶段 = 一个 Agent**，共 6 个：

```
A 选题分析   B 文献检索   C 研究路线   D 论文写作   E 格式规整   F 降重降AI
```

**Agent 内部的工作内容叫「节点」（node），不叫 Agent。**

```
阶段 E（格式规整 Agent）
├─ 节点 E1 格式规整      LLM 读格式要求直接改文件
├─ 节点 E2 文本一致性    术语 / 关键词 / 篇幅 / 图表引用
└─ 节点 E3 导出          BibTeX / GB-T 7714 / LaTeX + 回读校验
```

两次纠错的历史：

1. 早期把 20 个节点都叫 "Agent" —— **概念错误**。
   「Agent」意味着**一个能独立对外提供服务的单位**，那正是"阶段"；
   节点只是它内部的工作步骤。用户不会单独调用"格式校验"这个服务，
   他调用的是"格式规整"这个服务，校验是其中一步。
2. 后来链路被简化成 6 个环节（选题-文献-研究-写作-格式规整-降重降AI），
   原来的「质检」拆掉了、「交付」取消了、「降重/降AI」合并了。
   现在的 STAGES 就是这条链路。

## 阶段之间彼此独立

```
用户需要哪个阶段，就调用哪个阶段
        │
        ├──▶ 阶段 A 的入口  run_topic(...)         内部节点自动串起来
        ├──▶ 阶段 B 的入口  run_literature(...)    内部节点自动串起来
        ├──▶ 阶段 C 的入口  run_research(...)      内部节点自动串起来
        └──▶ …
```

**没有"全自动链路"** —— 不做"一份论文进去、六个阶段全跑完"这种东西。
理由：每个阶段的产出都需要人来判断（选题要不要做、综述写得行不行），
串成一条自动链只会把判断点挤到末尾，让人面对一堆无法追溯的产物。

**但阶段内部是全自动的**：一旦你调用阶段 E，E1→E2→E6→[E3→E4/E5]→合并
会按条件自动跑完，不需要你逐个节点调。

## 本模块提供什么

======================  ============================================================
``NodeSpec``            一个节点的声明（编号 / 名称 / 是否需模型 / 是否已实现）
``StageSpec``           **一个 Agent 的声明**（阶段键 / 目录 / 节点清单 / 入口）
``STAGES``              **6 个 Agent 的唯一事实来源**
``Finding``             一条统一的发现（跨节点可比、可合并）
``NodeResult``          一个节点跑完之后交回来的东西（**失败也是一种结果**）
``StageReport``         一个 Agent 的合并报告（**不认识任何具体节点**）
======================  ============================================================

**为什么清单是数据而不是代码分支**：文档、入口、检查脚本都从 `STAGES` 读，
避免"三处各写一份然后不一致"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from copilot.quality.format.report import Issue, Severity

__all__ = [
    # 数据
    "Finding",
    "NodeResult",
    "StageReport",
    # 声明
    "NodeSpec",
    "StageSpec",
    "STAGES",
    "stage_specs",
    "implemented_stages",
    "describe_stages",
    # 工具
    "finding_from_issue",
    "build_report",
]


# ─────────────────────────── 单条发现 ───────────────────────────


@dataclass(frozen=True)
class Finding:
    """一条跨节点统一格式的发现。

    ``node`` 是**报出它的节点编号**（E1 / E3 / C2 …），不是 Agent 编号。

    ``payload`` 是**刻意保留的逃生舱**：各节点的结构化细节（置信度、
    改写候选、保真结论……）原样放在这里，合并层不解释、不丢弃。

    为什么用 ``payload`` 而不是给 ``Finding`` 建子类：
    合并层处理的是**异质列表**（一次遍历 E1 的、E3 的、E6 的）。
    子类会让下游到处写 ``isinstance`` 分支；``payload`` 让下游
    「不认识的就原样透传」—— 合并报告因此不需要知道任何节点的细节。
    """

    node: str
    code: str
    severity: Severity
    message: str
    location: str = ""
    fix: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    #: 兼容旧字段名（早期写作 ``agent``）
    @property
    def agent(self) -> str:
        return self.node

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "code": self.code,
            "severity": self.severity.value,
            "location": self.location,
            "message": self.message,
            "fix": self.fix,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Finding":
        """从 ``to_dict()`` 还原。**checkpoint 能被读回来，靠的就是这一对。**"""
        return cls(
            node=str(data.get("node") or ""),
            code=str(data.get("code") or ""),
            severity=_severity_of(data.get("severity")),
            message=str(data.get("message") or ""),
            location=str(data.get("location") or ""),
            fix=str(data.get("fix") or ""),
            payload=dict(data.get("payload") or {}),
        )

    def __str__(self) -> str:
        where = f"（{self.location}）" if self.location else ""
        return f"[{self.code}] {self.message}{where}"


def _severity_of(value: Any) -> Severity:
    """把 ``"error"`` / ``Severity.ERROR`` 统一成 ``Severity``。

    反序列化时拿到的可能是字符串，也可能是枚举本身 —— 两种都要能转。
    """
    if isinstance(value, Severity):
        return value
    try:
        return Severity(str(value))
    except ValueError:
        return Severity.INFO


def finding_from_issue(node: str, issue: Issue, **payload: Any) -> Finding:
    """把 E1 / E6 的 ``Issue`` 转成 ``Finding``。

    这是"适配器只写一次"的体现：两个节点共用同一个 ``Issue`` 词汇
    （E6 刻意复用 E1 的），所以只需要一个转换函数。
    """
    return Finding(
        node=node,
        code=issue.code,
        severity=issue.severity,
        message=issue.message,
        location=issue.location,
        fix=issue.fix,
        payload=payload,
    )


# ─────────────────────────── 单个节点的结果 ───────────────────────────


@dataclass
class NodeResult:
    """一个节点跑完之后交回来的东西。

    **失败也是一种结果**：``ok=False`` 且带 ``error``，
    而不是抛异常把整个阶段带崩。
    """

    node: str
    ok: bool = True
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def agent(self) -> str:
        """兼容旧字段名。"""
        return self.node

    def count(self, severity: Severity | None = None) -> int:
        if severity is None:
            return len(self.findings)
        return sum(1 for f in self.findings if f.severity is severity)

    def codes(self) -> list[str]:
        return [f.code for f in self.findings]

    def summary_line(self) -> str:
        if not self.ok:
            return f"{self.node}：失败 —— {self.error}"
        return (
            f"{self.node}：{self.count(Severity.ERROR)} 错误 / "
            f"{self.count(Severity.WARNING)} 警告 / "
            f"{self.count(Severity.INFO)} 提示"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "ok": self.ok,
            "error": self.error,
            "stats": self.stats,
            "notes": self.notes,
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeResult":
        return cls(
            node=str(data.get("node") or ""),
            ok=bool(data.get("ok", True)),
            error=data.get("error"),
            stats=dict(data.get("stats") or {}),
            notes=list(data.get("notes") or []),
            findings=[Finding.from_dict(f) for f in (data.get("findings") or [])],
        )



# ─────────────────────────── 声明：节点 与 Agent（阶段） ───────────────────────────


@dataclass(frozen=True)
class NodeSpec:
    """Agent 内部的一个**工作内容**（节点）。"""

    code: str
    name: str
    requires_model: bool = False
    implemented: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "requires_model": self.requires_model,
            "implemented": self.implemented,
            "description": self.description,
        }


@dataclass(frozen=True)
class StageSpec:
    """**一个 Agent** —— 一个可以独立提供给用户的服务。"""

    key: str
    """阶段键：``A`` / ``B`` / ``C`` / ``D`` / ``E`` / ``F``。"""

    code: str
    """目录名：``01-topic`` … ``06-revise``。"""

    name: str
    """中文名：选题分析 / 文献检索 / 研究路线 / 论文写作 / 格式规整 / 降重降AI。"""

    nodes: tuple[NodeSpec, ...] = ()
    """这个 Agent 内部的工作内容。"""

    entry: str = ""
    """对外入口：``模块.函数``，用户只调这一个。"""

    implemented: bool = False
    description: str = ""

    # ── 派生的便利属性 ──

    @property
    def node_codes(self) -> tuple[str, ...]:
        return tuple(n.code for n in self.nodes)

    @property
    def implemented_nodes(self) -> tuple[str, ...]:
        return tuple(n.code for n in self.nodes if n.implemented)

    @property
    def requires_model(self) -> bool:
        """**任一个**节点需要模型，这个 Agent 就需要"可选的模型接入"。"""
        return any(n.requires_model for n in self.nodes)

    def node(self, code: str) -> NodeSpec:
        for n in self.nodes:
            if n.code == code:
                return n
        raise KeyError(f"{self.key} 阶段没有节点 {code}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "code": self.code,
            "name": self.name,
            "entry": self.entry,
            "implemented": self.implemented,
            "requires_model": self.requires_model,
            "description": self.description,
            "nodes": [n.to_dict() for n in self.nodes],
        }


def _n(code: str, name: str, *, model: bool = False, done: bool = False, desc: str = "") -> NodeSpec:
    return NodeSpec(code=code, name=name, requires_model=model, implemented=done, description=desc)


#: **6 个 Agent（阶段）的唯一事实来源。**
#:
#: 注意 ``implemented`` 指的是**这个阶段的入口可用**，
#: 而 ``NodeSpec.implemented`` 指**单个节点可用** ——
#: 两者可以不一致（阶段可用但某个节点只是骨架）。
STAGES: dict[str, StageSpec] = {
    "A": StageSpec(
        key="A", code="01-topic", name="选题分析",
        entry="copilot.topic.run_topic",
        implemented=True,
        description="从领域语料挖信号 → 评估可行性 → 出开题报告骨架",
        nodes=(
            _n("A1", "选题分析", done=True,
               desc="关键词共现 / 时间趋势 / 研究空白候选（纯统计）"),
            _n("A2", "可行性评估", model=True, done=True,
               desc="数据可得性 / 工作量 / 阻断性风险（缺省给骨架）"),
            _n("A3", "开题报告", model=True, done=True,
               desc="结构化开题报告（缺省给骨架 + 待填标记）"),
        ),
    ),
    "B": StageSpec(
        key="B", code="02-literature", name="文献检索",
        entry="copilot.literature.run_literature",
        implemented=True,
        description="检索文献 → 管理引用（含真实性核实）→ 出综述骨架",
        nodes=(
            _n("B1", "文献检索", done=True, desc="多路检索 + RRF 融合（复用检索层）"),
            _n("B2", "综述写作", model=True, done=True, desc="按组写作 + 逐句证据绑定（缺省给骨架）"),
            _n("B3", "引用管理", done=True,
               desc="去重 / DOI 核实 / BibTeX + GB-T 7714 —— **真实性的唯一关口**"),
        ),
    ),
    "C": StageSpec(
        key="C", code="03-research", name="研究路线",
        entry="copilot.research.run_research",
        implemented=True,
        description="设计研究方案 → 做统计分析（配对检验 + BH 校正）",
        nodes=(
            _n("C1", "方法设计", model=True, done=True, desc="路线比选 / 基线 / 指标（缺省给骨架）"),
            _n("C2", "数据分析", done=True, desc="配对检验 / 差值 CI / 效应量 / 多重比较校正"),
        ),
    ),
    "D": StageSpec(
        key="D", code="04-writing", name="论文写作",
        entry="copilot.writing.run_writing",
        implemented=True,
        description="出大纲 → 规范图表 → 逐节写作 → **校验引用来源**",
        nodes=(
            _n("D1", "大纲生成", done=True, desc="模板骨架 + 要点分配 + 篇幅分配"),
            _n("D2", "分章写作", model=True, done=True, desc="逐节生成 + 逐句证据绑定（缺省给骨架）"),
            _n("D3", "图表规范", done=True, desc="编号 / 交叉引用 / 题注（复用 E1 规则，反向用作生成）"),
            _n("D4", "文献引用", done=True,
               desc="正文引用必须来自 B 阶段的文献库 —— 从源头保证真实性"),
        ),
    ),
    "E": StageSpec(
        key="E", code="05-format", name="格式规整",
        entry="copilot.format_stage.run_format",
        implemented=True,
        description="按格式要求规整文件 → 文本一致性 → 导出（含回读校验）",
        nodes=(
            _n("E1", "格式规整", model=True, done=True,
               desc="LLM 读格式要求直接改文件 + diff + 未覆盖项清单"),
            _n("E2", "文本一致性", done=True,
               desc="术语一致 / 关键词覆盖 / 章节篇幅 / 图表引用"),
            _n("E3", "导出", done=True, desc="BibTeX / GB-T 7714 / LaTeX + 回读一致性校验"),
        ),
    ),
    "F": StageSpec(
        key="F", code="06-revise", name="降重降AI",
        entry="copilot.revise.run_revise",
        implemented=True,
        description="按外部报告定位问题片段 → **一次改写同时满足两个目标**",
        nodes=(
            _n("F1", "报告解析定位", done=True,
               desc="解析查重/AI 报告；**只保留有定位的片段**，未定位的交回用户"),
            _n("F2", "合并改写", model=True, done=True,
               desc="一次改写同时降重 + 降AI，保真 → 重合 → 风格 三道校验"),
        ),
    ),
}


def stage_specs(*, implemented_only: bool = False) -> list[StageSpec]:
    """按阶段键顺序返回 Agent 清单。"""
    items = [STAGES[k] for k in sorted(STAGES)]
    if implemented_only:
        items = [s for s in items if s.implemented]
    return items


def implemented_stages() -> list[str]:
    return [s.key for s in stage_specs(implemented_only=True)]


def describe_stages() -> str:
    """渲染成一张 Markdown 表（文档、演示、检查脚本复用同一份数据）。"""
    lines = [
        "| 阶段 | Agent | 目录 | 入口 | 节点 | 需模型 | 状态 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in stage_specs():
        lines.append(
            f"| **{s.key}** | {s.name} | `{s.code}` | `{s.entry}` | "
            f"{'、'.join(s.node_codes)} | {'是' if s.requires_model else '否'} | "
            f"{'✅' if s.implemented else '⬜'} |"
        )
    return "\n".join(lines)


# ─────────────────────────── 合并报告 ───────────────────────────


@dataclass
class StageReport:
    """一个阶段的**合并报告**。

    它只依赖 ``NodeResult`` / ``Finding``，**不认识任何具体节点** ——
    所以新增一个节点时，这个类一行都不用改。
    """

    stage: str = ""
    label: str = "<inline>"
    results: list[NodeResult] = field(default_factory=list)

    # ── 聚合 ──

    def findings(self, severity: Severity | None = None) -> list[Finding]:
        out = [f for r in self.results for f in r.findings]
        if severity is not None:
            out = [f for f in out if f.severity is severity]
        return sorted(out, key=lambda f: f.severity.rank)

    def by_node(self) -> dict[str, NodeResult]:
        return {r.node: r for r in self.results}

    #: 兼容旧方法名（早期叫 ``by_agent``）
    by_agent = by_node

    def counts(self) -> dict[str, int]:
        return {
            "nodes": len(self.results),
            "error": len(self.findings(Severity.ERROR)),
            "warning": len(self.findings(Severity.WARNING)),
            "info": len(self.findings(Severity.INFO)),
        }

    def failed_nodes(self) -> list[NodeResult]:
        return [r for r in self.results if not r.ok]

    #: 兼容旧方法名
    failed_agents = failed_nodes

    @property
    def can_submit(self) -> bool:
        """有 ERROR 或有节点失败，都不能算通过。

        **节点失败按阻断处理**：一个跑挂的检查不等于"这项没问题"。
        """
        return not self.findings(Severity.ERROR) and not self.failed_nodes()

    def verdict(self) -> str:
        if self.can_submit:
            return "✅ 通过（无阻断项）"
        if self.failed_nodes():
            names = "、".join(r.node for r in self.failed_nodes())
            return f"❌ 未通过：{names} 执行失败"
        return f"❌ 未通过：{len(self.findings(Severity.ERROR))} 个阻断项"

    def summary_line(self) -> str:
        c = self.counts()
        head = f"阶段 {self.stage}：" if self.stage else ""
        return (
            f"{head}{self.verdict()}（{c['nodes']} 个节点 / "
            f"错误 {c['error']} / 警告 {c['warning']} / 提示 {c['info']}）"
        )

    # ── 输出 ──

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "label": self.label,
            "can_submit": self.can_submit,
            "verdict": self.verdict(),
            "counts": self.counts(),
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageReport":
        """从 ``to_dict()`` 还原。

        ``can_submit`` / ``verdict`` / ``counts`` **不还原** ——
        它们是算出来的，读回后重新计算，避免"缓存值"与"实际 findings"不一致。
        """
        return cls(
            stage=str(data.get("stage") or ""),
            label=str(data.get("label") or ""),
            results=[NodeResult.from_dict(r) for r in (data.get("results") or [])],
        )


    def to_markdown(self) -> str:
        c = self.counts()
        title = f"# 阶段 {self.stage} 报告" if self.stage else "# 阶段报告"
        lines = [
            title,
            "",
            f"- 对象：`{self.label}`",
            f"- 结论：**{self.verdict()}**",
            f"- 规模：{c['nodes']} 个节点参与"
            f"（错误 {c['error']} / 警告 {c['warning']} / 提示 {c['info']}）",
            "",
            "## 各节点执行情况",
            "",
            "| 节点 | 结果 | 错误 | 警告 | 提示 | 备注 |",
            "|---|---|---|---|---|---|",
        ]
        for r in self.results:
            note = "；".join(r.notes)[:60] if r.notes else ("—" if r.ok else (r.error or ""))
            lines.append(
                f"| `{r.node}` | {'✅' if r.ok else '❌ 失败'} | "
                f"{r.count(Severity.ERROR)} | {r.count(Severity.WARNING)} | "
                f"{r.count(Severity.INFO)} | {note} |"
            )

        for level, section in (
            (Severity.ERROR, "## 🚫 阻断项"),
            (Severity.WARNING, "## ⚠️ 警告"),
            (Severity.INFO, "## ℹ️ 提示"),
        ):
            items = self.findings(level)
            if not items:
                continue
            lines += ["", section, "", "| 节点 | 规则码 | 位置 | 问题 | 建议 |", "|---|---|---|---|---|"]
            for f in items:
                msg = " ".join(f.message.split())[:80].replace("|", "/")
                fix = " ".join((f.fix or "").split())[:60].replace("|", "/")
                lines.append(f"| `{f.node}` | `{f.code}` | {f.location} | {msg} | {fix} |")

        lines += [
            "",
            "---",
            "",
            "> 本报告由各节点的结构化输出**合并**而来，",
            "> 未做二次判断，也未隐藏任何节点的失败。",
            "",
        ]
        return "\n".join(lines)


#: 兼容旧名（本项目早期把阶段报告叫 ``QualityReport``）
QualityReport = StageReport


def build_report(
    label: str, results: Iterable[NodeResult], *, stage: str = ""
) -> StageReport:
    return StageReport(stage=stage, label=label, results=list(results))
