"""E4 · 改写器接口 —— 模型接入点（可插拔）。

为什么必须做成接口
------------------
这是整个 E 阶段**唯一必须要模型**的一环（E1/E2/E3/E6 都能确定性实现）。
做成可插拔接口有三个实际好处：

1. **没有模型时全链路仍可跑通** —— 用 :class:`StubRewriter`，
   测试与联调不被阻塞；
2. **换模型不改其它代码**；
3. **可以做对照** —— 不同模型/提示词的降重效果可以量化比较。

内置的三个实现
--------------
===============  ==========================================
实现              用途
===============  ==========================================
StubRewriter     原样返回。用于测试"没有改善"这条路径
CannedRewriter   按预设映射返回人手写的改写。用于演示
CallableRewriter 包一个函数。用于测试与快速接入
===============  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from copilot.quality.rewrite.strategy import RewriteInstruction
from copilot.textnorm import normalize

__all__ = [
    "RewriteRequest",
    "Rewriter",
    "StubRewriter",
    "CannedRewriter",
    "CallableRewriter",
]


@dataclass(frozen=True)
class RewriteRequest:
    """一次改写请求。"""

    text: str
    instruction: RewriteInstruction
    #: 上下文（所在段落的前后文），帮助模型保持连贯
    context: str = ""

    def to_prompt_hint(self) -> str:
        """拼成给模型的指令文本。

        为什么写在代码里而不是散在提示词文件里：
        **指令与策略是同一个决策**，分开维护迟早会不一致。
        """
        return (
            f"【改写要求】{self.instruction.hint}\n"
            f"【内容类型】{self.instruction.kind.label}\n"
            f"【原文】{self.text}"
        )


class Rewriter(Protocol):
    name: str

    def rewrite(self, request: RewriteRequest) -> list[str]: ...


class StubRewriter:
    """**无模型时的替身**：原样返回，不产生任何改写。

    它的用途不是"假装能改写"，而是**让全链路可跑通**：
    切分 → 策略 → 生成 → 度量 → 保真 → 报告，每一环都能被测试到。

    由于它原样返回，效果度量会显示"没有改善"，于是走"未达标"分支 ——
    **这正好把失败路径也测到了**，比返回一个"看起来更好"的假改写有用得多。
    """

    name = "stub"

    def rewrite(self, request: RewriteRequest) -> list[str]:
        return [request.text]


class CannedRewriter:
    """按预设映射返回改写候选。

    **用于演示**：演示里那几句"好改写"是人手写的，
    用它来展示"如果模型给出好的改写，整条链路会怎样"。

    ⚠️ 演示时必须标明这是人手写的候选，**不能当成模型能力的证据**。

    匹配方式是**归一化后的精确匹配**（去空白、去句末标点），
    而不是原始字符串匹配 —— 因为改写单元会按句切分、
    句末标点常被裁掉，用原始串匹配会莫名对不上。
    """

    def __init__(self, mapping: dict[str, Sequence[str]], *, name: str = "canned") -> None:
        self.mapping = {_key(k): list(v) for k, v in mapping.items()}
        self.name = name

    def rewrite(self, request: RewriteRequest) -> list[str]:
        return list(self.mapping.get(_key(request.text), []))


def _key(text: str) -> str:
    """归一化匹配键：去空白、去首尾标点。

    只用于 :class:`CannedRewriter` 的查表，不参与任何判定逻辑。
    """
    t = normalize(text)
    compact = "".join(ch for ch in t if not ch.isspace())
    return compact.strip(".,;:!?()[]\"'")


class CallableRewriter:
    """把一个函数包成改写器。测试与快速接入用。"""

    def __init__(
        self,
        fn: Callable[[RewriteRequest], list[str]],
        *,
        name: str = "callable",
    ) -> None:
        self._fn = fn
        self.name = name

    def rewrite(self, request: RewriteRequest) -> list[str]:
        return list(self._fn(request))
