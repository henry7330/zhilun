"""标注评测集：E1 / E3 / E6 / F2 的**带标准答案**用例。

为什么用 Python 模块而不是 JSON
-------------------------------
用例里全是中文、多行、带 Markdown 结构。写成 JSON 需要大量 ``\\n`` 转义，
改一条用例要数半天引号 —— **可读性直接决定这个数据集会不会被维护**。

用 Python 的元组 + 三引号，改起来像写文档。

## 每个用例的字段

====================  =========================================================
``id``                用例标识
``agent``             测哪个 Agent：``E1`` / ``E3`` / ``E6`` / ``F2``
``desc``              **这条用例在测什么**（报告里会展示）
``input``             输入（键名与各 Agent 的接口一致）
``options``           可选参数（模板、匿名要求、章节阈值…）
``codes_all``         必须**全部**出现的规则码
``codes_none``        必须**一个都不**出现的规则码
``located``           期望定位成功的条数（E3）
``unlocated``         期望未定位的条数（E3）
``can_submit``        期望的提交判定（F2）
====================  =========================================================

## 判定口径

- ``codes_all`` 是**召回**的来源：漏了就扣 recall；
- 实际报出但不在 ``codes_all`` 里的码算**误报**，扣 precision；
- 干净用例（``codes_all`` 为空）专门用来量**误报率** —— 没有它，
  precision 永远是 1.0，那是自欺。

## ⚠️ 这套用例的边界

用例是**我按各 Agent 的设计写的**，不是从真实论文里抽的。
所以它验证的是"实现符合设计"，**不是"在真实论文上够用"**。
要后者，需要真实的、他人标注的论文集 —— 本项目没有，也不假装有。
"""

from __future__ import annotations

#: 评测统一用的模板。
#:
#: 为什么不用 `default_template()`：它要求 9 个章节 + 8000 字，
#: 几乎所有小样例会同时触发一堆 SEC001，**淹没真正想测的那一条**。
#: 评测要的是"这条规则准不准"，不是"整篇论文合不合格"。
EVAL_TEMPLATE = {
    "name": "评测模板",
    "required_sections": ["摘要", "绪论", "方法", "结论", "参考文献"],
    "min_body_words": None,
}


def _refs(n: int, *, start: int = 1) -> str:
    """生成 n 条格式合法、序号连续的参考文献。

    刻意做得**干净**：这样 E1 的 REF 类规则不会误报，
    用例才能只测它想测的那一条。
    """
    lines = []
    for i in range(start, start + n):
        lines.append(f"[{i}] 张三. 一种用于实验的方法与系统[J]. 计算机学报, 2020, 43(2): 1-10.")
    return "\n".join(lines)


def _paper(
    *,
    conclusion: str = "本文的方法在多个数据集上取得了稳定效果，在低资源场景下仍存在进一步提升的空间，后续将围绕这一方向继续展开。",
    conclusion_title: str = "## 结论",
    body_extra: str = "",
    refs: int = 2,
    ref_start: int = 1,
) -> str:
    """拼一份结构完整的草稿，只让调用方改想改的地方。"""
    return f"""# 一个用于实验的研究题目

## 摘要

本文研究了一种用于文献检索的增强方法，并在三个公开数据集上完成了验证工作。

## 绪论

检索增强生成在近年受到广泛关注。已有工作讨论了检索质量对最终效果的影响，
本文在此基础上进一步分析了长尾查询上的表现差异。

{body_extra}

## 方法

我们采用两阶段流程：先做粗筛，再做精排。粗筛用词面信号召回候选，
精排用语义信号重新排序，两阶段之间用一个小型重排模块连接起来。

{conclusion_title}

{conclusion}

## 参考文献

{_refs(refs, start=ref_start)}
"""


# ─────────────────────────── E1 格式校验 ───────────────────────────

_E1 = [
    {
        "id": "e1-missing-conclusion",
        "agent": "E1",
        "desc": "缺「结论」章 —— 应报 SEC001",
        "input": {
            "draft": _paper().replace("## 结论", "## 小结")
        },
        "options": {"template": EVAL_TEMPLATE},
        "codes_all": ["SEC001"],
    },
    {
        "id": "e1-empty-section",
        "agent": "E1",
        "desc": "「结论」只有标题没有正文 —— 应报 SEC005",
        "input": {"draft": _paper(conclusion="")},
        "options": {"template": EVAL_TEMPLATE},
        "codes_all": ["SEC005"],
    },
    {
        "id": "e1-ref-gap",
        "agent": "E1",
        "desc": "参考文献序号从 1 跳到 3 —— 应报 REF004",
        "input": {
            "draft": _paper(refs=1, ref_start=1).replace(
                "[1] 张三. 一种用于实验的方法与系统[J]. 计算机学报, 2020, 43(2): 1-10.",
                "[1] 张三. 一种用于实验的方法与系统[J]. 计算机学报, 2020, 43(2): 1-10.\n"
                "[3] 李四. 另一种方法[J]. 软件学报, 2021, 32(4): 5-20.",
            )
        },
        "options": {"template": EVAL_TEMPLATE},
        "codes_all": ["REF004"],
    },
    {
        "id": "e1-fig-ref-missing",
        "agent": "E1",
        "desc": "正文引用「图 5」但全文没有图 5 —— 应报 FIG002",
        "input": {"draft": _paper(body_extra="整体框架如 图 5 所示。")},
        "options": {"template": EVAL_TEMPLATE},
        "codes_all": ["FIG002"],
    },
    {
        "id": "e1-clean",
        "agent": "E1",
        "desc": "结构完整、引用规范的草稿 —— **不应报任何结构/引用/图表问题**",
        "input": {"draft": _paper()},
        "options": {"template": EVAL_TEMPLATE},
        "codes_all": [],
        "codes_none": ["SEC001", "SEC005", "REF004", "FIG002"],
    },
]


# ─────────────────────────── E6 审稿挑错 ───────────────────────────

_E6 = [
    {
        "id": "e6-term-inconsistent",
        "agent": "E2",
        "desc": "同一概念混用「召回率」与「查全率」 —— 应报 REV001",
        "input": {
            "draft": _paper(
                body_extra="已有工作讨论了召回率对最终效果的影响，另一些报告里又称查全率。"
            )
        },
        "options": {"min_section_chars": 1},
        "codes_all": ["REV001"],
    },
    {
        "id": "e6-keyword-gap",
        "agent": "E2",
        "desc": "摘要把「鲁棒性」列为关键词但正文从未出现 —— 应报 REV002",
        "input": {
            "draft": _paper().replace(
                "本文研究了一种用于文献检索的增强方法，并在三个公开数据集上完成了验证工作。",
                "本文研究了一种用于文献检索的增强方法。\n\n**关键词**：检索增强；鲁棒性",
            )
        },
        "options": {"min_section_chars": 1},
        "codes_all": ["REV002"],
    },
    {
        "id": "e6-short-section",
        "agent": "E2",
        "desc": "「结论」只有 10 个字 —— 应报 REV003",
        "input": {"draft": _paper(conclusion="方法有效。")},
        "options": {"min_section_chars": 120},
        "codes_all": ["REV003"],
    },
    {
        "id": "e6-clean",
        "agent": "E2",
        "desc": "术语一致、关键词都在正文出现、章节篇幅足够 —— **不应报 REV 类问题**",
        "input": {
            "draft": _paper().replace(
                "本文研究了一种用于文献检索的增强方法，并在三个公开数据集上完成了验证工作。",
                "本文研究了一种用于文献检索的增强方法，并在三个公开数据集上完成了验证工作。\n\n"
                "**关键词**：检索增强",
            )
        },
        "options": {"min_section_chars": 1},
        "codes_all": [],
        "codes_none": ["REV001", "REV002", "REV003"],
    },
]


# ─────────────────────────── F2 提交前检查 ───────────────────────────

_F2 = [
    {
        "id": "f2-placeholder",
        "agent": "E1H",
        "desc": "正文残留 TODO —— 应报 PRE001（阻断项）",
        "input": {"draft": _paper(body_extra="TODO: 这里还要补充相关工作。")},
        "options": {"template": EVAL_TEMPLATE},
        "codes_all": ["PRE001"],
        "can_submit": False,
    },
    {
        "id": "f2-email",
        "agent": "E1H",
        "desc": "正文出现邮箱 —— 应报 PRE002，且**报告里要脱敏**",
        "input": {"draft": _paper(body_extra="有疑问请联系 zhangsan@example.com。")},
        "options": {"template": EVAL_TEMPLATE},
        "codes_all": ["PRE002"],
        "can_submit": False,
    },
    {
        "id": "f2-acknowledgement",
        "agent": "E1H",
        "desc": "双盲投稿但致谢非空 —— 应报 PRE003",
        "input": {
            "draft": _paper().replace(
                "## 参考文献", "## 致谢\n\n感谢清华大学李教授的指导。\n\n## 参考文献"
            )
        },
        "options": {"template": EVAL_TEMPLATE, "anonymity_required": True},
        "codes_all": ["PRE003"],
        "can_submit": False,
    },
    {
        "id": "f2-clean",
        "agent": "E1H",
        "desc": "无占位符、无敏感信息、无致谢 —— **应判为可提交**",
        "input": {"draft": _paper()},
        "options": {"template": EVAL_TEMPLATE, "anonymity_required": True},
        "codes_all": [],
        "codes_none": ["PRE001", "PRE002", "PRE003"],
        "can_submit": True,
    },
]


# ─────────────────────────── E3 报告解析与定位 ───────────────────────────

_E3_SNIPPET = "本文的方法在多个数据集上取得了稳定效果"

_E3 = [
    {
        "id": "e3-exact",
        "agent": "F1",
        "desc": "报告片段与正文逐字一致 —— 应精确定位到 1 条，0 条未定位",
        "input": {"draft": _paper(), "report": _E3_SNIPPET},
        "located": 1,
        "unlocated": 0,
    },
    {
        "id": "e3-unlocated",
        "agent": "F1",
        "desc": "报告片段在正文里根本不存在（旧版本内容）—— 应 1 条未定位，**不硬指位置**",
        "input": {
            "draft": _paper(),
            "report": "这段文字完全来自另一篇论文，本文里一个字都没有。",
        },
        "located": 0,
        "unlocated": 1,
    },
    {
        "id": "e3-whitespace",
        "agent": "F1",
        "desc": "报告片段带多余空格（PDF 复制常见）—— 仍应精确定位",
        "input": {
            "draft": _paper(),
            "report": "本文的方法在  多个数据集上  取得了稳定效果",
        },
        "located": 1,
        "unlocated": 0,
    },
]


#: 全部用例
CASES: list[dict] = [*_E1, *_E6, *_F2, *_E3]

#: 覆盖的 Agent
AGENTS_COVERED: tuple[str, ...] = ("E1", "E1H", "E2", "F1")
