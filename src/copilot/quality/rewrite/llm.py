"""用真实模型做改写（E4 降重 / E5 降AI 共用）。

为什么放在 E4 的包里而不是 ``00-core``
--------------------------------------
``00-core/copilot/llm.py`` 负责"**怎么调模型**"（HTTP、重试、密钥、配置）；
本文件负责"**怎么让模型改写**"（提示词、清洗、候选数）。

E5 通过 ``from copilot.quality.rewrite import LlmRewriter`` 直接复用，
不需要自己再写一套提示词 —— **同一件事只能有一份实现**。

★ 提示词里的硬约束来自项目的四条铁律
------------------------------------
改写环节最容易出的问题是"为了改得像而把事实改了"。所以系统提示里
把不可动的部分逐条列死：

1. **不新增信息** —— 原文没有的数字 / 实验条件 / 文献一律不得出现；
2. **不动数字与专名** —— 数字、单位、缩写、引用标注原样保留；
3. **不改变结论方向** —— 提升不能写成下降；
4. **长度受控** —— 0.8~1.2 倍，太长太短都会被下游的保真校验拒掉。

这些约束和 ``fidelity.py`` 的检查项**一一对应**：
提示词说"不要改数字"，保真校验就查数字是否保留。
**提示词与校验器说同一件事**，模型才知道边界在哪。

⚠️ 提示词是软约束，校验器才是硬闸门
------------------------------------
即使模型不听话，``fidelity.check_fidelity`` 也会把它拦下来 ——
所以本文件**不负责保证质量**，只负责**尽量给出好的候选**。
这个分工很重要：把"质量"寄托在提示词上是不设防的。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from copilot.llm import LlmConfig, LlmError, Transport, chat
from copilot.quality.rewrite.rewriter import RewriteRequest, Rewriter

__all__ = [
    "LlmRewriter",
    "SYSTEM_PROMPT",
    "STRATEGY_GUIDANCE",
    "build_messages",
    "clean_candidate",
]


#: 系统提示。**这是整条链路上唯一"教模型守规矩"的地方。**
SYSTEM_PROMPT = """你是中文学术论文的改写助手。你的任务是改写给定文本的表达方式，同时保持事实完全不变。

必须遵守（违反任何一条，改写结果都会被丢弃）：
1. 不得新增原文没有的信息。特别是：数字、百分比、实验条件、数据集名称、文献引用，一个都不能凭空出现。
2. 保留所有数字、单位、专有名词、方法名、缩写、以及引用标注（如 [1]、（张三, 2020））原样不变。
3. 不得改变原意。如果原文说的是"提升"，改写后不能变成"下降"；原文的限定条件（如"在低资源场景下"）不能丢。
4. 改写后的字数控制在原文的 0.8 到 1.2 倍之间。
5. 只输出改写后的正文本身。不要解释改写思路，不要加"改写："之类的前缀，不要用引号包裹，不要用代码块。

如果无法在不改变事实的前提下改写，就原样返回原文。"""


#: ★ v2 追加的结构要求 —— 按策略给出**可核对**的目标。
#:
#: 为什么需要它（实测）：v1 提示词只说"改写表达方式"，模型的典型行为是
#: **只替换几个词、保留原句骨架** —— 结果是连续重叠 12~21 字、n-gram 包含 43%~53%，
#: 全部超过达标线（≤12 字 / ≤35%）。
#:
#: 根因：模型把"改写"理解成了"同义替换"。要拿到低重合，
#: 必须让它**改结构**（语序 / 视角 / 分句 / 连接方式）。
#:
#: ★ 注意最后那句量化目标：**提示词里说的数字与 `effect.py` 的达标线一致**。
#: 提示词与度量说同一件事，模型才有明确的靶子。
STRATEGY_GUIDANCE: dict[str, str] = {
    "rephrase": """【结构要求 · 重要】
不要只替换同义词 —— 那只算润色，不算改写。必须改变句子结构：
- 调整语序（把条件/时间状语挪到句首或句尾）
- 换主语视角（「本文」→「我们的方法」→「该方案」）
- 拆分长句，或把两个短句合并
- 改变分句之间的连接方式（因果、并列、递进换一种说法）

**目标：改写后不要有任何连续 10 个字以上与原文相同。**
但仍然不得改动任何数字、专有名词与引用标注。""",
    "restructure": """【结构要求 · 重要】
在保留全部事实的前提下重组表达：
- 可以调整句子之间的顺序，只要逻辑关系不变
- 可以把「先结论后依据」改成「先依据后结论」
- 主动句与被动句互换

**目标：改写后不要有任何连续 10 个字以上与原文相同。**""",
    "cite_instead": """【说明】
这一段更适合直接引用而不是改写。如果原文本身是他人观点的精确表述，
请原样返回，不要改写。""",
}


def clean_candidate(text: str) -> str:
    """清洗模型输出里常见的装饰。

    模型经常会自作主张加这些东西，而下游按句比对，多一个前缀就对不上：

    - ```` ```text ... ``` ```` 代码围栏
    - ``改写：`` / ``改写后：`` 前缀
    - 整体包裹的引号或书名号
    """
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*\n?", "", t)
    t = re.sub(r"\n?```\s*$", "", t)
    t = re.sub(r"^(改写后|改写|修改后|结果|输出)\s*[:：]\s*", "", t)
    if len(t) >= 2 and t[0] in "“\"'「『" and t[-1] in "”\"'」』":
        t = t[1:-1].strip()
    return t.strip()


def build_messages(request: RewriteRequest, *, structured_guidance: bool = True) -> list[dict[str, str]]:
    """把一个 ``RewriteRequest`` 组装成对话消息。

    只带**这一段**的上下文（``request.context`` 由 E4 给出，通常是前后各一句），
    不塞全文 —— 上下文越长，模型越容易抓到别处的内容来"帮忙"。

    ``structured_guidance=False`` 时退化为 v1 提示词（不追加结构要求）。
    这个开关只为**做对照实验**而存在：评估"提示词迭代到底有没有用"。
    """
    instruction = request.instruction
    hint = getattr(instruction, "hint", "") or ""
    rationale = getattr(instruction, "rationale", "") or ""
    strategy = getattr(getattr(instruction, "strategy", None), "value", "")

    parts = ["【改写要求】"]
    if hint:
        parts.append(hint)
    if strategy:
        parts.append(f"改写方式：{strategy}")
    if rationale:
        parts.append(f"（这样要求的原因：{rationale}）")

    if structured_guidance:
        guidance = STRATEGY_GUIDANCE.get(strategy)
        if guidance:
            parts.append("\n" + guidance)

    if request.context.strip():
        parts.append("\n【上下文，仅供理解，不要改写】")
        parts.append(request.context.strip())

    parts.append("\n【待改写内容】")
    parts.append(request.text.strip())

    parts.append(
        "\n请直接输出改写后的这一段，保持事实与数字不变，字数接近原文。"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


@dataclass
class LlmRewriter:
    """调用真实模型的改写器。

    ``name`` 里带模型名 —— 报告里因此能看出"这份建议是哪个模型给的"。
    可追溯性不该只覆盖证据，也要覆盖生成。

    ★ ``n_candidates > 1`` 未必生效
    ---------------------------------
    OpenAI 兼容接口的 ``n`` 参数**服务端可能不支持** —— 实测某个端点
    请求 3 个候选只返回 1 个，而且**不报错**。

    静默失效比报错更糟：调用方以为"我拿 3 个候选挑最好的"，
    实际上一直在拿 1 个。所以这里用 ``n_ignored`` 计数，
    让调用方能如实报告"多候选没生效，请改用重试"。
    """

    config: LlmConfig
    #: 一次请求要几个候选（E4 会挑最好的那个）
    n_candidates: int = 1
    #: 测试注入用；``None`` 走真实 HTTP
    transport: Transport | None = None
    #: 是否追加策略专属的「结构要求」（v2 提示词）。
    #: 置 ``False`` 即退化为 v1 —— **只为对照实验存在**。
    structured_guidance: bool = True

    #: 累计调用次数（成功返回才算）
    calls: int = field(default=0, init=False)
    #: 请求了多个候选、但服务端只回了 1 个的次数
    n_ignored: int = field(default=0, init=False)

    @property
    def name(self) -> str:
        prompt = "v2" if self.structured_guidance else "v1"
        return f"llm:{self.config.model}/{prompt}"

    @property
    def n_supported(self) -> bool:
        """多候选在这个端点/模型上是否真的生效过。"""
        return self.n_ignored == 0

    def rewrite(self, request: RewriteRequest) -> list[str]:
        try:
            raw = chat(
                build_messages(request, structured_guidance=self.structured_guidance),
                config=self.config,
                n=max(1, self.n_candidates),
                transport=self.transport,
            )
        except LlmError:
            # **不吞异常**：让上层的"需人工/失败"分支如实反映模型没跑通。
            # 静默返回空列表会被误读成"模型认为不需要改"。
            raise

        self.calls += 1
        if self.n_candidates > 1 and len(raw) <= 1:
            self.n_ignored += 1

        cleaned = [c for c in (clean_candidate(t) for t in raw) if c]
        # 与原文完全相同的候选没有意义（度量会显示"没改善"），直接丢掉，
        # 让上层走"需人工"分支而不是"改了个寂寞"。
        return [c for c in cleaned if c.strip() != request.text.strip()]

    def stats(self) -> dict[str, object]:
        """给报告用的运行统计 —— 让"多候选有没有生效"可见。"""
        return {
            "calls": self.calls,
            "n_candidates": self.n_candidates,
            "n_ignored": self.n_ignored,
            "n_supported": self.n_supported,
        }
