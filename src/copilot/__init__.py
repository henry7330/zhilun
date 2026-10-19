"""copilot：织论 Zhilun 的核心包 —— 6 个阶段 / 17 个节点。

## 目录

源码统一在 ``src/copilot/``（单一布局），按职责分三块：

**共享层**::

    contracts.py    6 个 Agent + 17 个节点的唯一事实来源
    stage.py        阶段图：把节点编成 LangGraph StateGraph
    bib.py          文献模型 + 身份判定（B3 与 D4 共用）
    llm.py          模型接入（OpenAI 兼容，密钥 repr 脱敏）
    evidence/       证据层：四态判定 / 绑定 / 三重校验
    retrieval/      检索层：保偏移切分 / 三路信号 / RRF
    memory/         记忆层：信任分级 / 写入门禁 / 双时态
    guard/          风控层：风险信号 / 处置动作 / 审计
    graph/          能力流水线（LangGraph，验证骨架不依赖模型）

**编排层**::

    orchestrator/   主 Agent：三层路由（选阶段 / 选节点 / 选模型）

**阶段（子 Agent）**::

    topic/          A 选题分析
    literature/     B 文献检索
    research/       C 研究路线
    writing/        D 论文写作
    format_stage/   E 格式规整
    revise/         F 降重降AI

**阶段内节点实现**（被上面两个阶段复用，本身不对外作为服务）::

    quality/format/          E1 的 14 条格式规则
    quality/review/          E2 文本一致性
    quality/report_parse/    F1 报告解析定位
    quality/rewrite/         F2 改写原语
    quality/deai/            F2 降AI 原语
    delivery/export/         E3 导出
    delivery/prcheck/        E1 内容卫生

## 为什么 import 名是 copilot，而项目名是 zhilun

``zhilun`` 是发行名（pip 包名 / 仓库名），``copilot`` 是 import 名。
改 import 名要动几百处，收益只是好看 —— 不值。两者不一致是**刻意的**。
"""
