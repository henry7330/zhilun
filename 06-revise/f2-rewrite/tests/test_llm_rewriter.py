"""``LlmRewriter`` 测试（**不联网**）。

核心验收：
1. **提示词把不可动的部分列死**（数字 / 专名 / 引用 / 结论方向）；
2. 模型输出里的装饰（代码围栏、``改写：`` 前缀、包裹引号）被清掉；
3. **与原文完全相同的候选被丢弃** —— 否则会走"改了但没改善"的分支，
   而不是如实的"需人工"；
4. 模型调用失败**向上抛**，不被吞成空列表（否则会被误读成"模型认为不用改"）。
"""

from __future__ import annotations

import pytest

from copilot.llm import LlmConfig, LlmError
from copilot.quality.rewrite import (
    ContentKind,
    LlmRewriter,
    RewriteInstruction,
    RewriteRequest,
    Strategy,
    build_messages,
    clean_candidate,
)
from copilot.quality.rewrite.llm import SYSTEM_PROMPT

# ─────────────────────── 夹具 ───────────────────────

ORIGINAL = "本文提出了一种基于图神经网络的推荐方法，并在三个数据集上验证了效果。"


def _cfg() -> LlmConfig:
    return LlmConfig(base_url="https://example.test/v1", api_key="k", model="m1")


def _request(text: str = ORIGINAL) -> RewriteRequest:
    return RewriteRequest(
        text=text,
        instruction=RewriteInstruction(
            strategy=Strategy.REPHRASE,
            kind=ContentKind.METHOD,
            hint="改句式，不要换术语",
            rationale="方法名不能替换",
        ),
        context="前一句是本文的动机说明。",
    )


def _reply(*texts: str):
    return lambda url, headers, body, timeout: (
        200,
        {"choices": [{"message": {"content": t}} for t in texts]},
    )


# ─────────────────── 提示词 ───────────────────


@pytest.mark.parametrize(
    "must_mention",
    ["数字", "专有名词", "引用", "原意", "0.8 到 1.2 倍"],
)
def test_system_prompt_states_hard_rules(must_mention: str) -> None:
    """★ 提示词必须与 `fidelity.py` 的检查项说同一件事。"""
    assert must_mention in SYSTEM_PROMPT


def test_system_prompt_forbids_adding_facts() -> None:
    assert "不得新增" in SYSTEM_PROMPT
    assert "不得改变原意" in SYSTEM_PROMPT


def test_build_messages_contains_instruction_and_text() -> None:
    messages = build_messages(_request())
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "改句式，不要换术语" in user          # hint
    assert "方法名不能替换" in user               # rationale
    assert "rephrase" in user                     # strategy
    assert ORIGINAL in user                       # 待改写内容
    assert "前一句是本文的动机说明" in user        # 上下文


def test_build_messages_marks_context_as_readonly() -> None:
    """上下文必须明确标"不要改写"，否则模型会把上下文也改了。"""
    user = build_messages(_request())[1]["content"]
    assert "不要改写" in user


def test_build_messages_without_context() -> None:
    req = RewriteRequest(
        text="一句话。",
        instruction=RewriteInstruction(
            strategy=Strategy.REPHRASE, kind=ContentKind.OTHER, hint="改", rationale=""
        ),
    )
    user = build_messages(req)[1]["content"]
    assert "上下文" not in user
    assert "一句话。" in user


# ─────────────────── 输出清洗 ───────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("```text\n改写结果\n```", "改写结果"),
        ("```\n改写结果\n```", "改写结果"),
        ("改写：这是结果", "这是结果"),
        ("改写后：这是结果", "这是结果"),
        ("“这是结果”", "这是结果"),
        ("「这是结果」", "这是结果"),
        ('"这是结果"', "这是结果"),
        ("  这是结果  ", "这是结果"),
    ],
)
def test_clean_candidate_strips_decorations(raw: str, expected: str) -> None:
    assert clean_candidate(raw) == expected


def test_clean_candidate_keeps_inner_quotes() -> None:
    """只剥整体包裹的引号，不碰正文内部的引号。"""
    text = '他说「这个方法有效」，因此值得推广'
    assert clean_candidate(text) == text


# ─────────────────── rewrite ───────────────────


def test_rewrite_returns_cleaned_candidate() -> None:
    rewriter = LlmRewriter(_cfg(), transport=_reply("```\n我们的方案借助图结构建模。\n```"))
    got = rewriter.rewrite(_request())
    assert got == ["我们的方案借助图结构建模。"]


def test_rewrite_returns_multiple_candidates() -> None:
    rewriter = LlmRewriter(_cfg(), n_candidates=2, transport=_reply("候选一。", "候选二。"))
    assert rewriter.rewrite(_request()) == ["候选一。", "候选二。"]


def test_rewrite_drops_candidate_identical_to_original() -> None:
    """★ 与原文一样的候选 = "改了但没改"，应丢弃让上层走"需人工"分支。"""
    rewriter = LlmRewriter(_cfg(), transport=_reply(ORIGINAL))
    assert rewriter.rewrite(_request()) == []


def test_rewrite_keeps_different_candidates_among_identical() -> None:
    rewriter = LlmRewriter(_cfg(), n_candidates=2, transport=_reply(ORIGINAL, "真正改过的句子。"))
    assert rewriter.rewrite(_request()) == ["真正改过的句子。"]


def test_rewrite_drops_blank_candidates() -> None:
    rewriter = LlmRewriter(_cfg(), n_candidates=2, transport=_reply("   ", "有效候选。"))
    assert rewriter.rewrite(_request()) == ["有效候选。"]


def test_rewrite_propagates_llm_error() -> None:
    """★ 失败必须向上抛 —— 吞成空列表会被误读成"模型认为不用改"。"""
    rewriter = LlmRewriter(_cfg(), transport=lambda *a: (500, {"error": "boom"}))
    with pytest.raises(LlmError):
        rewriter.rewrite(_request())


def test_rewriter_name_includes_model_and_prompt_version() -> None:
    """报告里要能看出这份建议**是哪个模型、哪个提示词版本**给的。

    可追溯性不该只覆盖证据，也要覆盖生成 ——
    否则"换了提示词以后变好了"这种结论就无法归因。
    """
    assert LlmRewriter(_cfg()).name == "llm:m1/v2"
    assert LlmRewriter(_cfg(), structured_guidance=False).name == "llm:m1/v1"


def test_strategy_guidance_only_in_v2() -> None:
    """v2 追加「结构要求」；v1 不追加 —— 这是对照实验的自变量。"""
    from copilot.quality.rewrite import STRATEGY_GUIDANCE

    v2_user = build_messages(_request())[1]["content"]
    v1_user = build_messages(_request(), structured_guidance=False)[1]["content"]
    assert STRATEGY_GUIDANCE["rephrase"] in v2_user
    assert STRATEGY_GUIDANCE["rephrase"] not in v1_user
    # 共同部分两版都有
    assert "改句式，不要换术语" in v2_user and "改句式，不要换术语" in v1_user


def test_v2_target_is_consistent_with_effect_threshold() -> None:
    """★ 提示词里的「连续 10 字」必须与 effect.py 的达标线（12 字）方向一致。

    提示词与度量说同一件事，模型才有明确靶子；
    如果提示词说"尽量改"、度量却卡 12 字，模型永远在盲猜。
    """
    from copilot.quality.rewrite import DEFAULT_MAX_RUN, STRATEGY_GUIDANCE

    assert "10 个字" in STRATEGY_GUIDANCE["rephrase"]
    assert DEFAULT_MAX_RUN >= 10, "提示词目标比达标线更严，才不会卡在边界"


def test_rewriter_satisfies_protocol_shape() -> None:
    rw = LlmRewriter(_cfg(), transport=_reply("x"))
    assert hasattr(rw, "name")
    assert callable(rw.rewrite)


# ─────────────────── 与 E4 流水线对齐 ───────────────────


def test_rewrite_result_can_enter_effect_measurement() -> None:
    """候选产出后能直接进 `measure_effect` —— 契约对齐的验证。"""
    from copilot.quality.rewrite import measure_effect

    candidate = "我们的方案借助图结构对关联关系建模。"
    effect = measure_effect(rewritten=candidate, original=ORIGINAL)
    assert effect.after_run >= 0
    assert 0.0 <= effect.after_containment <= 1.0


def test_rewrite_result_can_enter_fidelity_check() -> None:
    from copilot.quality.rewrite import check_fidelity

    candidate = "我们的方案借助图结构对关联关系建模。"
    result = check_fidelity(rewritten=candidate, original=ORIGINAL)
    assert result.ok, result.summary()

# ─────────────── ★ 多候选可能被服务端静默忽略 ───────────────


def test_tracks_ignored_n_candidates() -> None:
    """★ 请求 3 个候选、只回 1 个 —— **必须记下来，不能静默**。

    实测某个 OpenAI 兼容端点不支持 ``n``，请求 3 个只返回 1 个且不报错。
    调用方以为"我拿 3 个挑最好的"，其实一直只拿 1 个。
    """
    rewriter = LlmRewriter(_cfg(), n_candidates=3, transport=_reply("只有一个候选。"))
    rewriter.rewrite(_request())
    assert rewriter.n_ignored == 1
    assert rewriter.n_supported is False


def test_n_effective_when_server_honours_it() -> None:
    rewriter = LlmRewriter(_cfg(), n_candidates=3, transport=_reply("候选一。", "候选二。", "候选三。"))
    rewriter.rewrite(_request())
    assert rewriter.n_ignored == 0
    assert rewriter.n_supported is True


def test_n_candidates_1_never_counts_as_ignored() -> None:
    rewriter = LlmRewriter(_cfg(), n_candidates=1, transport=_reply("候选。"))
    rewriter.rewrite(_request())
    assert rewriter.n_ignored == 0
    assert rewriter.calls == 1


def test_stats_exposes_run_info() -> None:
    rewriter = LlmRewriter(_cfg(), n_candidates=2, transport=_reply("候选。"))
    rewriter.rewrite(_request())
    s = rewriter.stats()
    assert s["calls"] == 1
    assert s["n_candidates"] == 2
    assert s["n_supported"] is False
