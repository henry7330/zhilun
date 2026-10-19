"""主 Agent（编排层）测试。

## 核心验收

1. **三层路由各自可测**：选阶段、选节点、选模型
2. **节点收窄只减不增**，且**必须补齐依赖**（否则下游静默出空报告）
3. **缺输入跳过，不硬跑** —— 并且要能说清缺什么
4. **跳过 ≠ 通过**：`can_submit` 只在真有阶段跑过且都通过时为真
5. 模型规划器失败要**回退到规则版**，而不是重试或抛错
"""

from __future__ import annotations

import pytest

from copilot.orchestrator import (
    NODE_DEPS,
    ModelRouter,
    Plan,
    PlanStep,
    RulePlanner,
    LlmPlanner,
    close_dependencies,
    run_request,
)

DRAFT = """# 摘要

本文研究长尾推荐问题。

# 方法

本文设计了两阶段策略。

# 结论

方法有效。
"""

CTX = {"title": "图神经网络推荐", "draft": DRAFT, "question": "A 比 B 好吗"}


def plan_of(text: str, **ctx) -> Plan:
    return RulePlanner()(text, {**CTX, **ctx})


# ─────────────────── ① 选阶段 ───────────────────


@pytest.mark.parametrize(
    "text,stages",
    [
        ("帮我做个选题", ["A"]),
        ("找几篇文献", ["B"]),
        ("帮我设计方法", ["C"]),
        ("帮我看看格式", ["E"]),
        ("帮我降重", ["F"]),
    ],
)
def test_single_stage_requests(text: str, stages: list[str]) -> None:
    assert plan_of(text).stages == stages


def test_multi_stage_request() -> None:
    """一句话可以触发多个子 Agent —— 这正是「不是固定流程」的体现。"""
    plan = plan_of("把格式整理一下，再看看有没有重复的段落")
    assert plan.stages == ["E", "F"]


def test_pure_format_request_does_not_touch_other_stages() -> None:
    """只问格式，就不该顺手跑降重、写作。"""
    assert plan_of("帮我看看格式").stages == ["E"]


def test_no_match_is_reported_not_guessed() -> None:
    plan = plan_of("随便说点什么")
    assert plan.steps == []
    assert any("没有匹配到任何阶段" in n for n in plan.notes)


# ─────────────────── ② 选节点 ───────────────────


def test_stage_name_does_not_narrow() -> None:
    """★ "做个选题"里的「选题」是**阶段名**，不是子方面 → 跑整个阶段。"""
    step = plan_of("帮我做个选题").steps[0]
    assert step.nodes is None


def test_sub_aspect_narrows() -> None:
    """★ 点名了子方面才收窄。"""
    assert plan_of("看一下研究空白和趋势").steps[0].nodes == ("A1",)
    assert plan_of("字体字号对不对").steps[0].nodes == ("E1",)
    assert plan_of("导出 latex").steps[0].nodes == ("E3",)


def test_narrowing_never_adds_nodes() -> None:
    """收窄只能减少，不能增加 —— 多跑用户没要求的东西是错。"""
    for text in ("研究空白", "导出 latex", "引用来源靠谱吗"):
        for step in plan_of(text).steps:
            if step.nodes is None:
                continue
            from copilot.contracts import STAGES

            all_codes = {n.code for n in STAGES[step.stage].nodes}
            assert set(step.nodes) <= all_codes


def test_dependencies_are_closed() -> None:
    """★ 选了 F2 必须带上 F1，选了 A3 必须带上 A1。"""
    assert close_dependencies("F", ("F2",)) == ("F1", "F2")
    assert close_dependencies("A", ("A3",)) == ("A1", "A3")
    assert close_dependencies("D", ("D2",)) == ("D1", "D2")
    assert close_dependencies("B", ("B2",)) == ("B1", "B2")


def test_dependency_closure_keeps_declaration_order() -> None:
    """补齐后顺序必须与 STAGES 里一致 —— 反过来跑就取不到输入。"""
    assert close_dependencies("A", ("A2",)) == ("A1", "A2")


def test_dependency_closure_is_transitive() -> None:
    """多级依赖也要补齐（A3 → A1；如果将来 A3 → A2 → A1 也成立）。"""
    got = close_dependencies("A", ("A3",))
    assert "A1" in got


def test_every_declared_dep_exists() -> None:
    from copilot.contracts import STAGES

    for code, deps in NODE_DEPS.items():
        stage = STAGES[code[0]]
        known = {n.code for n in stage.nodes}
        assert code in known, f"依赖表里的 {code} 不是任何阶段的节点"
        for d in deps:
            assert d in known, f"{code} 依赖的 {d} 不存在"


def test_rewrite_request_narrows_and_closes() -> None:
    """「这段改写一下」→ F2 收窄，但必须带上 F1 才能定位。"""
    step = plan_of("这段改写一下").steps[0]
    assert step.nodes == ("F1", "F2")


def test_generic_reweight_runs_whole_stage() -> None:
    """★ 「帮我降重」是阶段名 → 整个 F 阶段（定位 + 改写）。"""
    step = plan_of("帮我降重").steps[0]
    assert step.nodes is None


# ─────────────────── ③ 选模型 ───────────────────


def test_deterministic_only_nodes_need_no_model() -> None:
    d = ModelRouter(default="x").resolve("E", ("E2", "E3"))
    assert d.model is None
    assert d.needs_model is False


def test_node_level_model_wins_over_stage_and_default() -> None:
    r = ModelRouter(default="d", by_stage={"A": "s"}, by_node={"A2": "n"})
    assert r.resolve("A", ("A2",)).model == "n"


def test_stage_model_used_when_no_node_override() -> None:
    r = ModelRouter(default="d", by_stage={"E": "s"})
    assert r.resolve("E", ("E1",)).model == "s"


def test_default_applies_when_nothing_registered() -> None:
    assert ModelRouter(default="d").resolve("F", ("F2",)).model == "d"


def test_no_model_registered_is_reported() -> None:
    """有节点需要模型但没登记 —— 要说出来，不能假装有模型。"""
    d = ModelRouter().resolve("E", ("E1",))
    assert d.model is None
    assert d.needs_model is True
    assert "骨架" in d.note or "没登记" in d.note


def test_mixed_models_are_disclosed() -> None:
    """一个阶段一次只能接一个模型 —— 混合时要如实说明。"""
    r = ModelRouter(default="a", by_node={"A2": "b"})
    d = r.resolve("A", None)      # A 阶段里 A2/A3 需要模型
    assert d.note and "混合" in d.note


def test_models_are_attached_to_steps() -> None:
    plan = RulePlanner()("帮我看看格式", CTX)
    assert plan.steps[0].model is None       # E1 需要模型但没登记


def test_router_reaches_the_plan() -> None:
    r = run_request(
        "帮我看看格式", context=CTX, router=ModelRouter(default="m"), plan_only=True
    )
    assert r.plan.steps[0].model == "m"


# ─────────────────── ④ 缺输入 ───────────────────


def test_missing_report_skips_revise_stage() -> None:
    """★ 要降重但没给报告 → 跳过，并说人话。"""
    r = run_request("帮我降重", context={"draft": DRAFT})
    assert r.steps[0].ran is False
    assert "外部查重" in r.steps[0].skipped_reason


def test_missing_draft_skips_format_stage() -> None:
    r = run_request("帮我看看格式", context={"title": "t"})
    assert r.steps[0].ran is False
    assert "论文草稿" in r.steps[0].skipped_reason


def test_missing_inputs_are_listed_in_plan() -> None:
    plan = RulePlanner()("帮我降重", {"draft": DRAFT})
    assert "report" in plan.missing


def test_stage_runs_when_inputs_present() -> None:
    r = run_request("帮我看看格式", context={"draft": DRAFT})
    assert r.steps[0].ran is True
    assert r.steps[0].report.stage == "E"


# ─────────────────── ⑤ 汇总语义 ───────────────────


def test_skipped_is_not_a_pass() -> None:
    """★ 跳过 ≠ 通过。一个都没跑成时不能算过。"""
    r = run_request("帮我降重", context={"draft": DRAFT})
    assert r.can_submit is False
    assert "没有任何阶段被执行" in r.summary_line() or "跳过" in r.summary_line()


def test_partial_run_still_reports_skips() -> None:
    r = run_request(
        "把格式整理一下，再看看有没有重复的段落", context={"draft": DRAFT}
    )
    assert len(r.ran_steps) == 1        # E 跑了
    assert len(r.skipped_steps) == 1    # F 缺报告
    assert r.can_submit is True         # E 通过


def test_pass_requires_all_ran_stages_to_pass() -> None:
    """D4 报 ERROR 时（引用的文献不在库里）→ 整体不可提交。"""
    from copilot.bib import BibEntry

    draft = DRAFT + "\n# 参考文献\n\n[1] 李四. 来路不明[J]. 某刊, 2019.\n"
    library = [
        BibEntry(key="z", authors=["张三"], title="另一种方法",
                 venue="某刊", year=2020, cn_type="J")
    ]
    r = run_request(
        "引用来源靠谱吗",
        context={"title": "t", "draft": draft, "entries": library},
    )
    assert r.ran_steps
    assert r.can_submit is False


def test_reference_count_is_reported_without_library() -> None:
    """★ 回归：没给文献库时也要报出"正文有几条参考文献"。

    之前提前 return，报告写「参考文献 0 条」，而实际有 1 条 —— 会误导用户。
    """
    draft = DRAFT + "\n# 参考文献\n\n[1] 李四. 来路不明[J]. 某刊, 2019.\n"
    r = run_request("引用来源靠谱吗", context={"title": "t", "draft": draft})
    d4 = r.ran_steps[0].report.by_node()["D4"]
    assert d4.stats["references"] == 1
    assert d4.stats["library"] == 0


def test_plan_only_does_not_execute() -> None:
    r = run_request("帮我看看格式", context=CTX, plan_only=True)
    assert r.plan.steps
    assert r.steps == []


def test_on_step_callback_fires() -> None:
    seen: list[str] = []
    run_request("帮我看看格式", context=CTX, on_step=lambda s, r: seen.append(s.stage))
    assert seen == ["E"]


def test_report_serializes() -> None:
    import json

    r = run_request("帮我看看格式", context=CTX)
    payload = json.loads(json.dumps(r.to_dict(), ensure_ascii=False))
    assert payload["plan"]["stages"] == ["E"]
    assert payload["steps"][0]["ran"] is True


def test_markdown_contains_plan_and_result() -> None:
    md = run_request("帮我看看格式", context=CTX).to_markdown()
    assert "主 Agent 执行报告" in md
    assert "规划明细" in md


# ─────────────────── ⑥ 模型规划器兜底 ───────────────────


def test_llm_planner_falls_back_on_bad_json() -> None:
    planner = LlmPlanner(complete=lambda _p: "这不是 JSON")
    plan = planner("帮我看看格式", CTX)
    assert plan.stages == ["E"]
    assert any("回退" in n for n in plan.notes)


def test_llm_planner_accepts_valid_json() -> None:
    planner = LlmPlanner(complete=lambda _p: '{"steps":[{"stage":"F","nodes":["F2"],"reason":"x"}]}')
    plan = planner("随便", CTX)
    assert plan.stages == ["F"]
    assert plan.steps[0].nodes == ("F1", "F2")   # 依赖被补齐


def test_llm_planner_drops_illegal_stage() -> None:
    planner = LlmPlanner(complete=lambda _p: '{"steps":[{"stage":"Z","nodes":null}]}')
    plan = planner("帮我看看格式", CTX)
    assert plan.stages == ["E"]                  # 回退到规则版


def test_llm_planner_filters_illegal_nodes() -> None:
    planner = LlmPlanner(
        complete=lambda _p: '{"steps":[{"stage":"E","nodes":["E1","NOPE"]}]}'
    )
    plan = planner("随便", CTX)
    assert plan.steps[0].nodes == ("E1",)


# ─────────────────── ⑦ 阶段入口的 only ───────────────────


def test_only_selects_a_subset_of_nodes() -> None:
    from copilot.format_stage import run_format

    report = run_format(draft=DRAFT, only=["E3"])
    assert [r.node for r in report.results] == ["E3"]


def test_only_rejects_unknown_node() -> None:
    from copilot.format_stage import run_format

    with pytest.raises(KeyError, match="E9"):
        run_format(draft=DRAFT, only=["E9"])


def test_only_empty_runs_nothing() -> None:
    from copilot.format_stage import run_format

    report = run_format(draft=DRAFT, only=[])
    assert report.results == []
