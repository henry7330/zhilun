"""F2 · 检查清单定义（声明式）。

为什么是声明式
--------------
与 E1 的模板同一条理由：**不同学校/期刊的提交要求不同**，
硬编码进代码就换一个要求改一次代码。

更重要的是：**"哪些项是阻断项"本身是配置，不是逻辑。**
有的会议允许附录，有的要求匿名；把"能不能提交"的判据写成数据，
才能让用户按自己的投稿要求调整。

与 E1 复用同一套 ``Severity``
-----------------------------
E1 的 ERROR / WARNING / INFO 在 F2 语境下正好对应
**阻断项 / 建议修 / 仅提示** —— 同一套词汇，两种用途。
这样 E1 与 F2 的产物可以直接合并成一份报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from copilot.quality.format.report import Severity

__all__ = [
    "CheckSpec",
    "Manifest",
    "SEVERITY_REGRADE",
    "default_manifest",
]


@dataclass(frozen=True)
class CheckSpec:
    """一项检查的声明。"""

    code: str
    title: str
    description: str
    severity: Severity = Severity.WARNING
    enabled: bool = True


@dataclass
class Manifest:
    """一次提交前检查的配置。"""

    checks: list[CheckSpec] = field(default_factory=list)
    #: 是否要求匿名（双盲投稿）。关闭时不跑匿名性检查。
    anonymity_required: bool = False
    #: 正文字数下限；``None`` 表示不查
    min_body_words: int | None = None
    #: 是否把 E1 的结构/格式问题也纳入本次检查
    include_format_checks: bool = True

    def enabled(self) -> list[CheckSpec]:
        return [c for c in self.checks if c.enabled]

    def severity_of(self, code: str) -> Severity:
        for c in self.checks:
            if c.code == code:
                return c.severity
        return Severity.WARNING


#: **把 E1 的问题重新定级**成"能不能提交"的判据。
#:
#: 为什么需要重新定级：E1 的严重级是**格式规范的严重程度**
#: （空章节只是"提示"，因为格式上没问题）；而提交前检查关心的是
#: **"带着这个能交吗"** —— 空章节显然不能。
#:
#: 同一次检测，两种判决。检测只做一遍，判决按用途各定各的。
SEVERITY_REGRADE: dict[str, Severity] = {
    # E1 的「章节结构」
    "SEC001": Severity.ERROR,     # 缺必需章节
    "SEC002": Severity.ERROR,     # 章节顺序错
    "SEC003": Severity.WARNING,   # 层级过深
    "SEC004": Severity.WARNING,   # 字数不足
    "SEC005": Severity.ERROR,     # 空章节  ← 从 INFO 升为阻断
    # E1 的「参考文献」
    "REF001": Severity.WARNING,
    "REF002": Severity.WARNING,
    "REF003": Severity.WARNING,
    "REF004": Severity.ERROR,     # 序号不连续
    "REF005": Severity.ERROR,     # 无参考文献
    "REF006": Severity.WARNING,
    # E1 的「图表」
    "FIG001": Severity.ERROR,     # 编号不连续
    "FIG002": Severity.ERROR,     # 引用了不存在的图表
    "FIG003": Severity.INFO,      # 图表未被引用
}


def default_manifest(*, anonymity_required: bool = False, min_body_words: int | None = None) -> Manifest:
    """默认清单：五项检查全部开启。"""
    return Manifest(
        checks=[
            CheckSpec(
                code="PRE001",
                title="占位符残留",
                description="正文里还留着 TODO / 待补 / 【】 一类未完成标记",
                severity=Severity.ERROR,
            ),
            CheckSpec(
                code="PRE002",
                title="敏感信息",
                description="正文里出现了邮箱 / 手机号 / 身份证 / 绝对路径 / 密钥形态",
                severity=Severity.ERROR,
            ),
            CheckSpec(
                code="PRE003",
                title="匿名性",
                description="双盲投稿时，致谢 / 基金号 / 机构名 / 个人主页会暴露身份",
                severity=Severity.ERROR,
                enabled=anonymity_required,
            ),
            CheckSpec(
                code="PRE004",
                title="结构完整性",
                description="复用 E1 的格式/结构规则，按「能不能交」重新定级",
                severity=Severity.ERROR,
            ),
            CheckSpec(
                code="PRE005",
                title="字数",
                description="正文（不含参考文献）是否达到下限",
                severity=Severity.WARNING,
            ),
        ],
        anonymity_required=anonymity_required,
        min_body_words=min_body_words,
    )
