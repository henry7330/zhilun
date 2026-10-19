"""F2 · 门禁判定与报告。

为什么要给一个明确的 yes/no
---------------------------
只说"有 12 个问题"没有决策价值。用户需要的是：

1. **能不能提交**（yes / no）；
2. 如果不能，**至少修哪几条**（阻断项）；
3. 哪些只是建议（警告）。

所以报告的第一行必须是结论，而不是问题列表。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from copilot.quality.format.report import Issue, Severity

from copilot.delivery.prcheck.manifest import Manifest

__all__ = ["PreSubmissionReport", "build_report"]


@dataclass
class PreSubmissionReport:
    """一次提交前检查的完整结果。"""

    doc_path: str
    manifest: Manifest
    issues: list[Issue] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.issues.sort(
            key=lambda i: (i.severity.rank, i.line if i.line is not None else 10**9)
        )

    # ── 判定 ──

    def blockers(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.INFO]

    @property
    def can_submit(self) -> bool:
        """**有任意阻断项就不能提交。**"""
        return not self.blockers()

    def counts(self) -> dict[str, int]:
        return {
            "blocker": len(self.blockers()),
            "warning": len(self.warnings()),
            "info": len(self.infos()),
            "total": len(self.issues),
        }

    def verdict(self) -> str:
        if self.can_submit:
            return "✅ 可以提交（无阻断项）"
        return f"❌ 不能提交：还有 {len(self.blockers())} 个阻断项必须处理"

    def summary_line(self) -> str:
        c = self.counts()
        return f"{self.verdict()}（阻断 {c['blocker']} / 警告 {c['warning']} / 提示 {c['info']}）"

    # ── 输出 ──

    def to_dict(self) -> dict:
        return {
            "file": self.doc_path,
            "can_submit": self.can_submit,
            "counts": self.counts(),
            "issues": [i.to_dict() for i in self.issues],
        }

    def to_markdown(self) -> str:
        c = self.counts()
        lines = [
            "# 提交前检查报告",
            "",
            f"- 文档：`{self.doc_path}`",
            f"- 匿名要求：{'是（双盲）' if self.manifest.anonymity_required else '否'}",
            "",
            "## 结论",
            "",
            f"### {self.verdict()}",
            "",
            f"阻断 **{c['blocker']}** / 警告 **{c['warning']}** / 提示 **{c['info']}**",
            "",
        ]

        if not self.issues:
            lines += ["没有发现问题。", ""]
            return "\n".join(lines)

        blockers = self.blockers()
        if blockers:
            lines += [
                "## 🚫 阻断项（必须修完才能提交）",
                "",
                "| 规则码 | 位置 | 问题 | 建议 |",
                "|---|---|---|---|",
            ]
            for i in blockers:
                lines.append(
                    f"| `{i.code}` | {i.location} | {_q(i.message)} | {_q(i.fix)} |"
                )
            lines.append("")

        warnings = self.warnings()
        if warnings:
            lines += [
                "## ⚠️ 警告（建议修）",
                "",
                "| 规则码 | 位置 | 问题 |",
                "|---|---|---|",
            ]
            for i in warnings:
                lines.append(f"| `{i.code}` | {i.location} | {_q(i.message)} |")
            lines.append("")

        infos = self.infos()
        if infos:
            lines += [
                "## ℹ️ 提示",
                "",
                "| 规则码 | 位置 | 问题 |",
                "|---|---|---|",
            ]
            for i in infos:
                lines.append(f"| `{i.code}` | {i.location} | {_q(i.message)} |")
            lines.append("")

        lines += [
            "## 检查项说明",
            "",
            "| 码 | 含义 |",
            "|---|---|",
            "| `PRE001` | 占位符残留（TODO / 待补 / 空占位） |",
            "| `PRE002` | 敏感信息（邮箱 / 手机号 / 身份证 / 路径 / 密钥） |",
            "| `PRE003` | 匿名性（致谢 / 基金号 / 机构名 / 仓库链接） |",
            "| `PRE004` | 结构完整性（**复用 E1 规则，按「能不能交」重新定级**） |",
            "| `PRE005` | 字数 |",
            "",
            "> 结构类问题的规则码沿用 E1（`SEC` / `REF` / `FIG`），",
            "> 因为**检测逻辑只有一份**，F2 只重新定级。",
            "",
        ]
        return "\n".join(lines)


def _q(text: str, limit: int = 90) -> str:
    t = " ".join((text or "").split()).replace("|", "/")
    if len(t) > limit:
        t = t[:limit] + "…"
    return t


def build_report(doc_path: str, manifest: Manifest, issues: Iterable[Issue]) -> PreSubmissionReport:
    return PreSubmissionReport(doc_path=doc_path, manifest=manifest, issues=list(issues))
