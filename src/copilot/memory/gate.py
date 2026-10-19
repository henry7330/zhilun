"""写入策略门禁：决定一条候选记忆能否进入长期记忆。

威胁模型
--------
Agent 会读不可信内容（网页、PDF、工具返回），处理结果又会被写进长期记忆。
于是：

    不可信内容 → 抽取为「记忆」→ 写入 → 跨会话存活
    → 未来会话检索到它，却以「用户自己的记忆」身份高信任度注入提示词

**记忆是一条跨越信任边界、然后以「可信」身份重新注入的通道。**
它比单轮提示注入严重得多：单轮注入只影响一轮，记忆投毒影响之后所有会话；
而且注入时的信任标记更高（"记忆" vs "外部文档"），用户也完全看不见。

本模块的作用：在这条通道上装一道闸门。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from copilot.evidence.models import DocStore
from copilot.evidence.verify import verify_span
from copilot.memory.models import MemoryItem, TrustLevel

__all__ = ["WriteDecision", "GateResult", "PolicyRules", "WriteGate", "PermissiveGate"]


class WriteDecision(str, Enum):
    ALLOW = "allow"
    """直接写入长期记忆。"""

    QUARANTINE = "quarantine"
    """存下来但**默认召回不到**，等补充证据后再提升。

    为什么不直接拒绝：拒绝会丢掉审计线索，也失去了"该内容后来被
    独立来源证实"时转正的可能。隔离是"暂缓"，不是"否定"。
    """

    REJECT = "reject"
    """直接丢弃（跨会话/跨用户内容，本就不该进入本用户记忆）。"""


@dataclass(frozen=True)
class GateResult:
    decision: WriteDecision
    reason: str

    @property
    def allowed(self) -> bool:
        return self.decision is WriteDecision.ALLOW


@dataclass
class PolicyRules:
    """可配置的策略规则。

    指令式措辞被单独拎出来，是因为**记忆本应是陈述性事实**。
    一段写在记忆里的祈使句（"用户已授权自动转账"）不是事实，而是
    试图改变未来行为的植入指令 —— 即使来源标称可信也值得复核。
    """

    instruction_patterns: tuple[str, ...] = (
        r"用户(已)?授权",
        r"忽略(之前|上述|以上|前面)",
        r"不要(引用|提及|告诉)",
        r"(使用|用)这个?(密钥|key|token)",
        r"(转账|付款|付钱|打款)",
        r"(always|never)\s+(do|use|reveal)",
        r"ignore\s+(all\s+)?(previous|above)",
        r"system\s*:",
    )
    _compiled: list[re.Pattern[str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.instruction_patterns]

    def looks_like_instruction(self, text: str) -> str | None:
        """返回命中的模式（便于诊断），未命中返回 None。"""
        for pattern in self._compiled:
            if pattern.search(text):
                return pattern.pattern
        return None


class WriteGate:
    """写入策略门禁。

    ================================  ====  ====  ====
    来源信任级别                        有证据  无证据
    ================================  ====  ====  ====
    TRUSTED_USER / SYSTEM             放行   放行
    DERIVED_USER                      放行   放行
    UNTRUSTED_EXTERNAL / TOOL         **放行**  隔离
    UNTRUSTED_CROSS                   拒绝   拒绝
    ================================  ====  ====  ====

    任何来源命中「指令式措辞」→ 隔离（需人工复核）。

    注意 "UNTRUSTED + 有证据 → 放行" 这一行：外部内容**可以**变成可信记忆，
    但前提是它绑定了通过三重校验的原文片段。
    这就是 P1 的证据层在这里的作用：**它是本门禁的验证器**。
    """

    def __init__(
        self,
        *,
        doc_store: DocStore | None = None,
        rules: PolicyRules | None = None,
    ) -> None:
        self._doc_store = doc_store
        self.rules = rules or PolicyRules()

    # ── 证据核验 ────────────────────────────────────────────

    def evidence_status(self, item: MemoryItem) -> tuple[bool, str]:
        """检查该记忆是否被可验证的原文支撑。

        复用 P1 的 ``verify_span``：文档存在 → hash 未被篡改 → 偏移处文本一致。
        三道都过才算"有证据"，任一失败即视为不可信 —— 因为
        "引用了被替换过的文档"和"没有引用"在可靠性上是一回事。
        """
        if not item.evidence:
            return False, "没有绑定任何原文片段"
        if self._doc_store is None:
            return False, "未配置文档库，无法校验证据"
        for span in item.evidence:
            check = verify_span(span, self._doc_store)
            if not check.ok:
                return False, f"证据校验失败（{check.problem.value}）：{check.detail}"
        return True, f"{len(item.evidence)} 个片段均通过三重校验"

    # ── 判定 ────────────────────────────────────────────────

    def evaluate(self, item: MemoryItem) -> GateResult:
        trust = item.provenance.trust

        # 1) 跨会话/跨用户：直接拒绝，与证据无关
        if trust is TrustLevel.UNTRUSTED_CROSS:
            return GateResult(
                WriteDecision.REJECT,
                "来源为其他会话/用户，不应写入本用户记忆",
            )

        # 2) 指令式措辞：任何来源都隔离待复核
        hit = self.rules.looks_like_instruction(item.content)
        if hit is not None:
            return GateResult(
                WriteDecision.QUARANTINE,
                f"内容呈指令式措辞（命中 {hit!r}）；记忆应为陈述性事实，需复核",
            )

        # 3) 不可信来源：要看有没有通过校验的证据
        if not trust.is_trusted:
            ok, detail = self.evidence_status(item)
            if ok:
                return GateResult(
                    WriteDecision.ALLOW,
                    f"不可信来源，但证据已通过校验，可放行（{detail}）",
                )
            return GateResult(
                WriteDecision.QUARANTINE,
                f"不可信来源（{trust.value}）且证据不足：{detail}",
            )

        # 4) 可信来源
        return GateResult(WriteDecision.ALLOW, f"来源可信（{trust.value}）")


class PermissiveGate:
    """基线门禁：**不做任何检查，全部放行**。

    存在的唯一目的是**消融对照** —— 要回答"门禁到底挡住了多少投毒"，
    就必须有一个"没有门禁"的对照组。没有对照组的防护效果是无法证明的。

    ⚠️ **不要用于生产。** 它等价于"Agent 会读外部内容、又把结果写进长期记忆、
    且不做任何信任检查"这一常见但危险的默认形态。
    """

    name = "permissive"

    def evaluate(self, item: MemoryItem) -> GateResult:  # noqa: ARG002 - 基线刻意忽略输入
        return GateResult(WriteDecision.ALLOW, "基线：不做检查，全部放行")

