"""F 阶段（降重降AI）测试。

核心验收：
1. **没有外部报告就拒绝运行**（不是"跳过"）—— 跳过会被误读为"没问题"；
2. **只处理有定位的片段**，未定位的如实交回用户；
3. 一次改写**同时**满足两个目标，两道校验都过才采纳；
4. 保真是否决条件：改动原意的候选一律弃用。
"""

from __future__ import annotations

import pytest

from copilot.revise import ReviseStatus, run_revise
from copilot.revise.combined import HINT_OVERLAP, HINT_STYLE, build_instruction, revise_unit
from copilot.quality.rewrite.units import RewriteUnit

DRAFT = """# 方法

本文提出了一种基于图神经网络的推荐方法，并在五个数据集上验证了效果。

这项工作具有十分重要的理论意义和现实价值。
"""

#: 与正文完全一致 → 能精确定位
PLANTED = "本文提出了一种基于图神经网络的推荐方法，并在五个数据集上验证了效果。"

#: 正文里根本没有 → 定位不到
MISSING = "这段文字完全不在正文里，纯属另一篇论文的内容。"


class FakeRewriter:
    """按给定候选返回的假改写器。"""

    name = "fake/v1"

    def __init__(self, *candidates: str) -> None:
        self._candidates = list(candidates)
        self.calls = 0

    def rewrite(self, request) -> list[str]:
        self.calls += 1
        return list(self._candidates)


# ─────────────────── 入口门禁 ───────────────────


def test_report_is_mandatory() -> None:
    """★ 没有报告就不做 —— 而且要说出来，不能静默跳过。"""
    with pytest.raises(ValueError, match="必须提供外部检测报告"):
        run_revise(draft=DRAFT, report="   ")


def test_report_type_is_validated() -> None:
    with pytest.raises(ValueError, match="report_type"):
        run_revise(draft=DRAFT, report=PLANTED, report_type="whatever")


def test_empty_draft_is_rejected() -> None:
    with pytest.raises(ValueError, match="draft"):
        run_revise(draft="", report=PLANTED)


# ─────────────────── F1 定位 ───────────────────


def test_unlocated_snippet_is_returned_to_user() -> None:
    """★ 定位不到就提示用户，**不猜位置**。"""
    report = run_revise(draft=DRAFT, report=MISSING)
    f1 = report.by_node()["F1"]
    assert f1.stats["unlocated"] == 1
    warned = [f for f in f1.findings if "定位不到" in f.message]
    assert warned, "未定位必须显式提示"
    assert warned[0].severity.value == "warning"


def test_located_snippet_is_counted() -> None:
    report = run_revise(draft=DRAFT, report=PLANTED)
    f1 = report.by_node()["F1"]
    assert f1.stats["located"] >= 1
    assert f1.stats["unlocated"] == 0


# ─────────────────── F2 改写 ───────────────────


def test_without_rewriter_nothing_is_silently_claimed() -> None:
    """没接模型 → 一个字都不该采纳，更不能报"已降重"。

    缺省 `StubRewriter` 会把原文原样返回（重合 100%），
    所以这里的状态是"未达标"而不是"无候选" —— 两者都不能算成功。
    """
    report = run_revise(draft=DRAFT, report=PLANTED)
    f2 = report.by_node()["F2"]
    assert f2.stats["accepted"] == 0
    assert any("未达标" in f.message or "没有候选" in f.message for f in f2.findings)


def test_rewritten_sentence_is_accepted_when_targets_met() -> None:
    better = "作者设计了一个两阶段框架，把图结构信息引入推荐建模，测试覆盖五个数据集。"
    report = run_revise(
        draft=DRAFT, report=PLANTED, rewriter=FakeRewriter(better), max_attempts=1
    )
    f2 = report.by_node()["F2"]
    assert f2.stats["accepted"] >= 1
    assert any("可采纳改写" in f.message for f in f2.findings)


def test_fidelity_violation_is_rejected() -> None:
    """★ 改动原意的候选必须弃用 —— 保真是否决条件。

    这里改的是**阿拉伯数字**（5 → 8）。中文数字（五个→十个）不在数字检查范围内，
    那是已知局限，不能拿它当测试用例。
    """
    draft = "# 方法\n\n该方法在 5 个数据集上验证，覆盖 30 个基线。\n"
    planted = "该方法在 5 个数据集上验证，覆盖 30 个基线。"
    tampered = "该方法在 8 个数据集上验证，覆盖 30 个基线。"

    report = run_revise(
        draft=draft, report=planted, rewriter=FakeRewriter(tampered), max_attempts=1
    )
    f2 = report.by_node()["F2"]
    assert f2.stats["accepted"] == 0
    assert any("改动原意" in f.message for f in f2.findings)


def test_retries_until_attempts_exhausted() -> None:
    bad = "完全相同的一句话。"
    rw = FakeRewriter(bad)
    run_revise(draft=DRAFT, report=PLANTED, rewriter=rw, max_attempts=2)
    assert rw.calls >= 2, "不达标应当重试"


# ─────────────────── 合并策略 ───────────────────


def _unit(text: str) -> RewriteUnit:
    return RewriteUnit(
        unit_id="u1", text=text, char_start=0, char_end=len(text), line=3
    )


def test_both_targets_produce_one_combined_instruction() -> None:
    """★ 一次改写同时满足两个目标 —— 指令里两段要求都要在。"""
    ins = build_instruction(_unit("这是一段需要同时降重和降AI的文字。"), ("overlap", "style"))
    assert HINT_OVERLAP in ins.hint
    assert HINT_STYLE in ins.hint


def test_single_target_only_includes_its_hint() -> None:
    ins = build_instruction(_unit("只降重。"), ("overlap",))
    assert HINT_OVERLAP in ins.hint
    assert HINT_STYLE not in ins.hint


def test_protected_unit_is_never_rewritten() -> None:
    u = _unit("张三等人提出了类似方法[1]。")
    u.protected = True
    u.protected_reason = "所在句含引用标注 [1]"
    s = revise_unit(u, rewriter=FakeRewriter("随便改改"), targets=("overlap",))
    assert s.status == ReviseStatus.SKIPPED_PROTECTED


def test_report_type_both_sets_two_targets() -> None:
    report = run_revise(draft=DRAFT, report=PLANTED, report_type="both")
    assert report.by_node()["F2"].stats["targets"] == ["overlap", "style"]


def test_report_type_ai_sets_style_only() -> None:
    report = run_revise(draft=DRAFT, report=PLANTED, report_type="ai")
    assert report.by_node()["F2"].stats["targets"] == ["style"]


def test_stage_reports_two_nodes_in_order() -> None:
    order: list[str] = []
    run_revise(draft=DRAFT, report=PLANTED, on_node=lambda c, _r: order.append(c))
    assert order == ["F1", "F2"]
