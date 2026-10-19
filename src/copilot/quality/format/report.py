"""子部分 ⑥ 问题报告：统一的问题词汇 + 汇总与呈现。

为什么先写这一层
----------------
它定义了**规则输出的统一格式**（`Issue`）。三条规则的产物必须是同一种东西，
否则汇总、分级、排序、下游消费都要各写一套。所以虽然它在设计路线里排第六，
代码上要先立好这个「词汇表」。

为什么分级而不是只报"有问题"
-----------------------------
**校验的价值取决于问题好不好修。** 一份列出 200 条同权重告警的报告，
用户会直接关掉。必须区分：

- ``ERROR``：明确不合规，不改会被退回（如缺「参考文献」章节）；
- ``WARNING``：很可能有问题，建议核对（如作者数超限未用「等」）；
- ``INFO``：提示性，可能是风格选择（如某图未被正文引用）。

这个分级不是装饰 —— 用户就是靠它决定先修哪些。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from copilot.quality.format.docmodel import DocModel
from copilot.quality.format.template import Template

__all__ = ["Severity", "Issue", "FormatReport"]


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    @property
    def rank(self) -> int:
        return {"error": 0, "warning": 1, "info": 2}[self.value]

    @property
    def label(self) -> str:
        return {"error": "❌ 错误", "warning": "⚠️ 警告", "info": "ℹ️ 提示"}[self.value]


@dataclass(frozen=True)
class Issue:
    """一条格式问题。

    ``code`` 是**稳定编号**（如 ``SEC001``）。为什么不用自然语言做标识：
    编号能被程序稳定地统计、忽略、断言 —— 报告里说"这次比上次少了 3 个
    REF002"，前提就是编号稳定。
    """

    code: str
    severity: Severity
    message: str
    rule: str
    line: int | None = None
    fix: str = ""

    @property
    def location(self) -> str:
        return f"第 {self.line + 1} 行" if self.line is not None else "全文"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "rule": self.rule,
            "line": self.line,
            "fix": self.fix,
        }


@dataclass
class FormatReport:
    """一次格式校验的完整结果。"""

    doc: DocModel
    template: Template
    issues: list[Issue] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 排序：先按严重级，再按行号 —— 用户是从上往下改的
        self.issues.sort(key=lambda i: (i.severity.rank, i.line if i.line is not None else 10**9))

    @property
    def has_error(self) -> bool:
        return any(i.severity is Severity.ERROR for i in self.issues)

    @property
    def passed(self) -> bool:
        """无 ERROR 即视为通过格式检查（WARNING/INFO 不阻塞）。"""
        return not self.has_error

    def by_severity(self, severity: Severity) -> list[Issue]:
        return [i for i in self.issues if i.severity is severity]

    def counts(self) -> dict[str, int]:
        return {
            "error": len(self.by_severity(Severity.ERROR)),
            "warning": len(self.by_severity(Severity.WARNING)),
            "info": len(self.by_severity(Severity.INFO)),
            "total": len(self.issues),
        }

    def codes(self) -> list[str]:
        """命中的规则码（去重、保序），便于回归测试与统计。"""
        seen: list[str] = []
        for i in self.issues:
            if i.code not in seen:
                seen.append(i.code)
        return seen

    def summary_line(self) -> str:
        c = self.counts()
        verdict = "通过" if self.passed else "未通过"
        return (
            f"格式校验{verdict}：{c['error']} 个错误 / {c['warning']} 个警告 / "
            f"{c['info']} 个提示（共 {c['total']} 项）"
        )

    def to_dict(self) -> dict:
        return {
            "file": str(self.doc.path),
            "template": self.template.name,
            "reference_style": self.template.reference_style,
            "passed": self.passed,
            "counts": self.counts(),
            "stats": {
                "body_words": self.doc.body_words,
                "sections": len(self.doc.flat_sections()),
                "references": len(self.doc.references),
                "captions": len(self.doc.captions),
                "cross_refs": len(self.doc.cross_refs),
            },
            "issues": [i.to_dict() for i in self.issues],
        }

    def to_markdown(self) -> str:
        c = self.counts()
        lines = [
            "# 格式校验报告",
            "",
            f"- 文档：`{self.doc.path}`",
            f"- 模板：**{self.template.name}**（参考文献风格 `{self.template.reference_style}`）",
            f"- 统计：正文 {self.doc.body_words} 字 / "
            f"{len(self.doc.flat_sections())} 个章节 / "
            f"{len(self.doc.references)} 条参考文献 / "
            f"{len(self.doc.captions)} 个图表",
            "",
            f"## 结论：{'✅ 通过' if self.passed else '❌ 未通过'}",
            "",
            f"错误 **{c['error']}** / 警告 **{c['warning']}** / 提示 **{c['info']}**",
            "",
        ]

        if not self.issues:
            lines += ["没有问题。", ""]
            return "\n".join(lines)

        lines += [
            "| 严重级 | 规则码 | 位置 | 问题 | 建议 |",
            "|---|---|---|---|---|",
        ]
        for i in self.issues:
            msg = i.message.replace("|", "/")
            fix = (i.fix or "—").replace("|", "/")
            lines.append(
                f"| {i.severity.label} | `{i.code}` | {i.location} | {msg} | {fix} |"
            )

        lines += ["", "## 规则码对照", ""]
        rules: dict[str, int] = {}
        for i in self.issues:
            rules[i.code] = rules.get(i.code, 0) + 1
        for code in sorted(rules):
            lines.append(f"- `{code}`：{rules[code]} 处")
        lines.append("")
        return "\n".join(lines)
