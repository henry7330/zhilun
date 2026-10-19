"""E4 · 保真校验 —— 改写之后，意思还是原来那个意思吗？

为什么这是最容易被忽略、也最关键的一环
--------------------------------------
改写最容易出的问题是「**改完之后意思变了**」。

而一个改变了原意的"降重"**比不降更糟** —— 它制造了一个与事实不符的表述，
正好是本项目要防的事。所以：

> **保真校验不通过，改写一律弃用 —— 即使降重效果再好。**

本模块做两类校验
----------------
1. **确定性检查（始终执行）** —— 数字、否定、长度、专名。
   这几项是"改写最容易出错、而且完全可以机器判"的地方。
2. **对照来源校验（有来源时执行）** —— 复用 E2 的规则判断，
   检查改写后的句子是否仍被来源原文支持。

**为什么第 1 类就足够抓出大部分问题**：模型改写时最常犯的错是
**吞掉数字**（"提升了 8.3%" → "有提升"）和**翻转否定**
（"并不优于" → "优于"）。这两类都能 100% 确定性判出，不需要模型。

**如实标注局限**：本模块**不做完整的语义等价判断**。
"改写后意思变了但数字和否定都没变"这种情况，这里抓不出来 ——
那需要真正的语义理解。这一点在报告里必须写明。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from copilot.evidence.text import normalize_number, numbers_in

__all__ = [
    "FidelityIssue",
    "FidelityResult",
    "check_fidelity",
    "DEFAULT_MIN_LENGTH_RATIO",
    "DEFAULT_MAX_LENGTH_RATIO",
]

#: 改写后长度与原句的比值下限/上限。
#: 低于 0.5 通常意味着"删掉了内容"；高于 2.0 通常意味着"加进了原文没有的东西"。
DEFAULT_MIN_LENGTH_RATIO = 0.5
DEFAULT_MAX_LENGTH_RATIO = 2.0

#: 需要原样保留的"专名"：英文缩写（如 LLM、BERT）、书名号内的内容
_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")
_BOOK_TITLE = re.compile(r"《([^》]{1,30})》")

#: "结论级"否定 —— 会让判断翻转的那一类。
#:
#: ★ 为什么不用通用的 ``has_negation``（实测踩过）：
#: 它匹配任何"不/没/未"，于是「不到一百行」「不少」这类普通表述
#: 也会被当成"否定被翻转"，把**正常的改写误判成失真**，
#: 导致大量可用候选被白白拒掉。
#:
#: 真正危险的是**判断层面的否定**（"并不优于" → "优于"），
#: 所以这里只认那类词。
_VERDICT_NEGATION = re.compile(
    r"并不|并未|未能|不能|不会|不可|无法|不足以|不属于|不存在|尚无|缺乏|没有(?:提升|改善|差异|效果|优势)"
)


@dataclass(frozen=True)
class FidelityIssue:
    """一条保真问题。"""

    kind: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.detail}"


@dataclass
class FidelityResult:
    """保真校验结果。"""

    ok: bool
    issues: list[FidelityIssue] = field(default_factory=list)
    #: 实际跑了哪些检查 —— 让用户知道"通过"是基于什么
    checks_run: list[str] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        """是否有致命问题（数字/否定）。这两类一旦出错必须弃用。"""
        return any(i.kind in ("number", "negation") for i in self.issues)

    def summary(self) -> str:
        if self.ok:
            return f"通过（已检查：{'、'.join(self.checks_run)}）"
        return "未通过：" + "；".join(str(i) for i in self.issues)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks_run": list(self.checks_run),
            "issues": [{"kind": i.kind, "detail": i.detail} for i in self.issues],
        }


def _latin_tokens(text: str) -> set[str]:
    """英文缩写/专名。这类词改写时必须原样保留。"""
    return {t for t in _LATIN_TOKEN.findall(text) if len(t) >= 2}


def _book_titles(text: str) -> set[str]:
    return set(_BOOK_TITLE.findall(text))


def check_fidelity(
    rewritten: str,
    original: str,
    *,
    source_text: str | None = None,
    min_length_ratio: float = DEFAULT_MIN_LENGTH_RATIO,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
) -> FidelityResult:
    """检查改写是否保真。

    ``source_text`` 有值时，额外把来源原文作为对照基准。
    """
    issues: list[FidelityIssue] = []
    checks: list[str] = []

    if not rewritten.strip():
        return FidelityResult(
            ok=False,
            issues=[FidelityIssue("empty", "改写结果为空")],
            checks_run=["empty"],
        )

    # ── 1) 数字必须保留 ──
    checks.append("数字保留")
    orig_numbers = {normalize_number(n) for n in numbers_in(original)}
    new_numbers = {normalize_number(n) for n in numbers_in(rewritten)}
    lost = orig_numbers - new_numbers
    if lost:
        issues.append(
            FidelityIssue(
                "number",
                f"改写后丢失了数字：{', '.join(sorted(lost))}"
                "（数字是事实，不能因为改写而消失）",
            )
        )
    added = new_numbers - orig_numbers
    if added:
        issues.append(
            FidelityIssue(
                "number",
                f"改写后出现了原文没有的数字：{', '.join(sorted(added))}"
                "（可能是编造，必须人工核对）",
            )
        )

    # ── 2) 结论级否定不能翻转 ──
    checks.append("否定保留")
    orig_neg = bool(_VERDICT_NEGATION.search(original))
    new_neg = bool(_VERDICT_NEGATION.search(rewritten))
    if orig_neg != new_neg:
        issues.append(
            FidelityIssue(
                "negation",
                f"结论层面的否定被改变了：原文{'有' if orig_neg else '没有'}"
                f"否定表述，改写后{'有' if new_neg else '没有'}"
                "（这是把结论说反了）",
            )
        )

    # ── 3) 长度要合理 ──
    checks.append("长度合理")
    if original.strip():
        ratio = len(rewritten) / len(original)
        if ratio < min_length_ratio:
            issues.append(
                FidelityIssue(
                    "length",
                    f"改写后长度只有原文的 {ratio:.0%}，可能删掉了内容"
                    f"（下限 {min_length_ratio:.0%}）",
                )
            )
        elif ratio > max_length_ratio:
            issues.append(
                FidelityIssue(
                    "length",
                    f"改写后长度是原文的 {ratio:.0%}，可能加进了原文没有的内容"
                    f"（上限 {max_length_ratio:.0%}）",
                )
            )

    # ── 4) 专名必须保留 ──
    checks.append("专名保留")
    for name, tokens in (
        ("英文缩写/专名", _latin_tokens(original)),
        ("书名", _book_titles(original)),
    ):
        missing = {t for t in tokens if t not in rewritten}
        if missing:
            issues.append(
                FidelityIssue(
                    "term",
                    f"改写后丢失了{name}：{', '.join(sorted(missing))}"
                    "（专名有确定的所指，不能替换）",
                )
            )

    # ── 5) 有来源时，对照来源做规则判断（复用 E2 的极性与数字检查）──
    if source_text and source_text.strip():
        checks.append("对照来源")
        src_numbers = {normalize_number(n) for n in numbers_in(source_text)}
        claimed = new_numbers - src_numbers - orig_numbers
        if claimed:
            issues.append(
                FidelityIssue(
                    "source",
                    f"改写后出现来源原文里没有的数字：{', '.join(sorted(claimed))}",
                )
            )
        if _VERDICT_NEGATION.search(source_text) and not _VERDICT_NEGATION.search(rewritten):
            issues.append(
                FidelityIssue(
                    "source",
                    "来源原文是否定表述，但改写后变成了肯定 —— 结论方向被改反了",
                )
            )

    # 致命问题（数字/否定）直接判不通过；其余问题也算不通过，
    # 因为"长度异常""专名丢失"同样会让表述失真
    return FidelityResult(ok=not issues, issues=issues, checks_run=checks)
