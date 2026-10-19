"""织论 Zhilun —— 本地交互演示（Streamlit）。

    streamlit run app.py

## 这个页面演示什么

1. **描述需求 → 主 Agent 出计划**：调哪几个阶段、阶段里哪几个节点、用什么模型
2. **确认后执行**：逐阶段跑，结果按报告展开
3. **单阶段直调**：跳过主 Agent，只做一件事

**不需要 API key 也能跑** —— 需要模型的节点会退化成骨架并标注。
想接模型：在仓库根建 `.env`（见 README）。
"""

from __future__ import annotations

import json

import streamlit as st

SAMPLE_DRAFT = """# 摘要

本文研究推荐系统中的长尾问题。

# 方法

我们设计了两阶段检索策略，并在 5 个数据集上验证了效果。

# 结论

方法有效。

# 参考文献

[1] 张三. 一种推荐方法[J]. 计算机学报, 2020.
"""

SAMPLE_REPORT = """我们设计了两阶段检索策略，并在 5 个数据集上验证了效果。
"""

SAMPLE_REQUIREMENT = """标题：黑体三号居中
正文：宋体小四，行距 1.5 倍
参考文献：按 GB/T 7714 著录
"""


def _plan_rows(plan):
    rows = []
    for s in plan.steps:
        rows.append(
            {
                "阶段": f"{s.stage}",
                "节点": "全部" if s.nodes is None else "、".join(s.nodes),
                "模型": s.model or "—（确定性）",
                "为什么调它": s.reason,
            }
        )
    return rows


def main() -> None:
    # 页面启动时读一次 .env（仓库根或当前目录），让需要模型的节点真能调到模型。
    from copilot.llm import load_env

    load_env()

    st.set_page_config(page_title="织论 Zhilun", page_icon="📄", layout="wide")

    st.title("📄 织论 Zhilun")
    st.caption("基于 LangGraph 的多智能体论文辅助系统 —— 六个阶段，按需调度")

    # ── 侧边栏：输入 ──
    with st.sidebar:
        st.header("输入")
        if st.button("载入示例数据", use_container_width=True):
            st.session_state["draft"] = SAMPLE_DRAFT
            st.session_state["report"] = SAMPLE_REPORT
            st.session_state["requirement"] = SAMPLE_REQUIREMENT

        draft = st.text_area(
            "论文草稿（Markdown）", value=st.session_state.get("draft", SAMPLE_DRAFT), height=220
        )
        report = st.text_area(
            "外部查重 / AI 报告（F 阶段必填）",
            value=st.session_state.get("report", ""), height=120,
        )
        requirement = st.text_area(
            "学校格式要求（E 阶段用）",
            value=st.session_state.get("requirement", ""), height=100,
        )
        st.session_state.update({"draft": draft, "report": report, "requirement": requirement})

        st.divider()
        st.caption(
            "不填报告就调用降重阶段时，系统会**跳过并说明原因**，"
            "而不是假装检查过。"
        )

    tab_request, tab_stage, tab_about = st.tabs(["按请求执行", "单阶段直调", "关于"])

    # ── Tab 1：主 Agent ──
    with tab_request:
        request = st.text_input(
            "用一句话描述你想做什么",
            value="把格式整理一下，再看看有没有重复的段落",
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            only_plan = st.checkbox("只看计划", value=True)
        with col2:
            st.caption("先看计划是最便宜的用法 —— 它不执行任何阶段，也不花 token。")

        if st.button("开始", type="primary", key="run_request"):
            from copilot.orchestrator import run_request

            context = {
                "title": "（未命名）",
                "draft": draft,
                "report": report,
                "requirement": requirement,
                "question": "（未提供）",
            }
            out = run_request(request, context=context, plan_only=only_plan)

            st.subheader("① 主 Agent 的计划")
            if not out.plan.steps:
                st.warning("没有匹配到任何阶段 —— 请说得具体一点（如「帮我检查格式」）。")
                for note in out.plan.notes:
                    st.caption(f"· {note}")
            else:
                st.write(f"**{out.plan.describe()}**")
                st.dataframe(_plan_rows(out.plan), use_container_width=True, hide_index=True)
                if out.plan.missing:
                    st.info("缺少输入：" + "、".join(out.plan.missing) + "（缺了会跳过对应阶段）")
                for note in out.plan.notes:
                    st.caption(f"· {note}")

            if not only_plan and out.plan.steps:
                st.subheader("② 执行结果")
                st.write(out.summary_line())
                for step in out.steps:
                    label = f"{step.step.stage} · " + (
                        "未执行" if not step.ran else step.report.verdict()
                    )
                    with st.expander(label, expanded=not step.ran):
                        if step.ran:
                            st.markdown(step.report.to_markdown())
                        else:
                            st.warning(f"跳过原因：{step.skipped_reason}")

                st.download_button(
                    "下载完整报告（Markdown）",
                    data=out.to_markdown(),
                    file_name="zhilun-report.md",
                    mime="text/markdown",
                )
                with st.expander("JSON（便于脚本消费）"):
                    st.code(json.dumps(out.to_dict(), ensure_ascii=False, indent=2), language="json")

    # ── Tab 2：单阶段 ──
    with tab_stage:
        st.caption("跳过主 Agent，直接调用某一个阶段 —— 这就是「六个阶段是独立服务」。")
        stage = st.selectbox(
            "选阶段",
            ["E 格式规整", "F 降重降AI", "D 论文写作", "C 研究路线", "B 文献检索", "A 选题分析"],
        )
        key = stage.split()[0]

        if st.button("运行这个阶段", key="run_stage"):
            from copilot.contracts import STAGES

            spec = STAGES[key]
            st.write(f"**{spec.key} {spec.name}** — 入口 `{spec.entry}`")
            st.write("节点：" + "、".join(f"{n.code} {n.name}" for n in spec.nodes))

            try:
                if key == "E":
                    from copilot.format_stage import run_format

                    rep = run_format(draft=draft, requirement=requirement)
                elif key == "F":
                    from copilot.revise import run_revise

                    if not report.strip():
                        st.error("F 阶段必须提供外部检测报告 —— 这是设计上刻意的拒绝，不是 bug。")
                        return
                    rep = run_revise(draft=draft, report=report)
                elif key == "D":
                    from copilot.writing import run_writing

                    rep = run_writing(title="（未命名）", draft=draft)
                elif key == "C":
                    from copilot.research import run_research

                    rep = run_research("（未提供研究问题）")
                elif key == "B":
                    from copilot.literature import run_literature

                    rep = run_literature()
                else:
                    from copilot.topic import run_topic

                    rep = run_topic(title="（未命名）")
            except Exception as exc:  # noqa: BLE001 - 页面不该因单次失败白屏
                st.error(f"{type(exc).__name__}: {exc}")
                return

            st.success(rep.summary_line())
            st.markdown(rep.to_markdown())
            st.download_button(
                "下载这个阶段的报告",
                data=rep.to_markdown(),
                file_name=f"zhilun-{key}.md",
                mime="text/markdown",
            )

    # ── Tab 3：关于 ──
    with tab_about:
        st.markdown(
            """
### 这是什么

一个**可验证**的论文辅助系统：六个阶段（选题 → 文献 → 研究 → 写作 → 格式 → 降重），
由主 Agent 按需调度。

### 几个刻意的设计

| 决策 | 为什么 |
|---|---|
| **不是固定流程** | 只问格式就不跑降重。用户没要求的事不做 |
| **缺输入就跳过** | 没给查重报告就不跑降重，**跳过 ≠ 通过** |
| **失败也是一种结果** | 节点崩了不算"这项没问题"，报告会判为不可通过 |
| **降重降AI 合并** | 分两次改会互相抵消：第一次改结构，第二次改回去 |
| **引用必须来自文献库** | 真实性在 B 阶段核实过，D 阶段只认那个库 |

### 已知局限（如实说）

- 评测用例是**自造的 16 条**，P=R=F1=1.0 只说明"实现符合设计"，不代表真实语料够用
- 改写通过率 42%→100% 是 **n=36** 的小样本
- 「降重」用的是字面重合度**代理指标**，不等于外部查重结果
- 没有 Word/PDF 导出（只有 BibTeX / LaTeX）
            """
        )
        st.caption("接模型：在仓库根建 `.env`，填 COPILOT_LLM_API_KEY / _BASE_URL / _MODEL。")


if __name__ == "__main__":
    main()
