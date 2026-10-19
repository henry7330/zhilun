# 02 · 文献

> **状态：⬜ 仅设计，未实现**（B2 需模型；B1/B3 的能力层已就绪，缺 Agent 外壳）
>
> 本文件是该阶段的**设计文档**：接口契约 + 各 Agent 的作用与「为什么需要」。

## 阶段目标

从「我要研究 X」到「我掌握了 X 的文献全貌，并能写出**有据可查**的综述」。

## 为什么这个阶段最关键

这是**证据密度最高**的一环：综述里每一句话都应当能指向具体文献。

也是**最容易出事**的一环 —— "编造一篇不存在的文献"就发生在这里。
所以本阶段是全场唯一 **100% 强制证据绑定**的阶段。

## 输入 / 输出契约

| | 内容 |
|---|---|
| **输入** | `TopicSpec`（题目 + 关键词 + 研究问题） |
| **输出** | `Library`：文献集合（带 `content_hash`）+ 综述 + 参考文献表 |
| **不做** | 不下载全文（只处理已获取的语料）；不做查重（那是 E3） |

### 向下游传递的数据

```python
Library = {
    "documents": list[Document],     # 每篇都有 doc_id + content_hash
    "survey": str,                   # 综述正文
    "claims": list[Claim],           # 综述里每条断言 + 它的证据
    "references": list[BibEntry],    # 规范化参考文献（F1 的类型）
}

Claim = {                           # 复用 copilot.evidence.models.Claim
    "claim_id": str, "text": str,
    "status": SupportStatus,        # 四态
    "evidence": list[EvidenceSpan], # 带 hash 的原文片段
    "citations": list[CitationMeta],
}
```

**★ `content_hash` 是跨阶段契约的核心** ——
有了它，D 阶段写作时才能确认"这条证据来自我当时读的那一版文献"，
而不是"某篇同名但内容已变的文章"。

## 三个 Agent

### B1 文献检索 `RETRIEVAL`

| | 内容 |
|---|---|
| **作用** | 多路检索 + 引文图扩展 + 精排 |
| **输入** | `keywords` + 研究问题 |
| **输出** | `list[Document]`（带 hash）+ 检索指标 |
| **需模型** | ❌ 不需要 |

#### 子部分

**① 多路检索** —— `BM25 ∪ 字符 trigram`，RRF 融合。

**为什么这个组合**：这是本项目**实测出来的**最优配置 ——
`reports/retrieval_ablation.md` 里唯一在配对检验下显著胜出的方案
（MRR +0.041，p(BH)=0.007）。而 dense 融合**显著有害**（MRR −0.091），
所以默认不启用。

> 这条结论有数据支撑，不是偏好。

**② 引文图扩展** —— 沿参考文献 / 被引关系扩散。

**为什么需要**：词面检索找得到"用同样术语的论文"，
找不到"用了不同术语但其实是同一问题的论文"。引文图补的正是这块。

**⚠️ 当前**：需外部数据源（Semantic Scholar / OpenAlex API），
本项目不做网络依赖，因此**标为未实现**。

**③ 精排** —— 用覆盖率信号重排。

**为什么实测里它没显著效果**（覆盖率重排点估计为负）：
在小语料上 BM25 已经够好，重排带来的噪声大于收益。
**保留接口但不默认开启** —— 这是"有数据支撑的不做"。

#### 接口

```python
def search_literature(
    keywords: Sequence[str],
    *,
    corpus: Sequence[Document],
    k: int = 20,
    enable_citation_expansion: bool = False,   # 需外部源
) -> Library: ...
```

---

### B2 综述写作 `SURVEY`

| | 内容 |
|---|---|
| **作用** | 产出有据可查的综述 |
| **输入** | `Library.documents` + 研究问题 |
| **输出** | `survey` 文本 + `list[Claim]`（每条带证据） |
| **需模型** | ✅ 必须，**且必须走证据绑定** |

#### 为什么这是全场最严格的一环

综述的一句话可能是"近年来图神经网络在推荐系统上取得了显著进展"——
这种"行业公论"最容易被模型自由发挥。

**做法**：先把文献**分组**（按方法/年份/问题），
再让模型**只在给定片段内**写，每句都要能落到具体 `doc_id` 上。

#### 三个子部分

**① 文献分组** —— 按主题聚类，每组给一段"共同点 / 分歧点"。
**为什么需要**：综述的价值在于**组织**，不在于罗列。不分组就是流水账。

**② 逐组写作 + 逐句绑定** —— 每组写完后立刻跑证据校验。
**为什么边写边校验**：写完再校验，一旦大面积 `unsupported`，
整段都要重写 —— 边写边验能及时纠正 prompt。

**③ 冲突标注** —— 不同文献结论矛盾时**并列写出**，不做仲裁。

**为什么**：领域内的真实分歧是综述最有价值的部分。
模型"综合"掉矛盾，等于抹掉了最该被讨论的东西。

---

### B3 引用管理 `REF`

| | 内容 |
|---|---|
| **作用** | 格式转换、去重、DOI 校验 |
| **输入** | `list[BibEntry]` |
| **输出** | 规范化参考文献表（GB/T 7714 / BibTeX） + 问题清单 |
| **需模型** | ❌ **完全确定性** |

#### 复用而非重写

**这个 Agent 的核心能力已经实现了**：

| 能力 | 现成实现 |
|---|---|
| 解析参考文献 → `BibEntry` | `copilot.delivery.export.from_reference` |
| BibTeX 生成（含转义） | `to_bibtex` / `render_bibtex_file` |
| GB/T 7714 生成 | `to_gbt7714` / `render_gbt7714_list` |
| DOI / 文献库核实 | `copilot.evidence.verify_citation` |
| 去重 | **需新增**（按 DOI 或规范化标题） |

**★ 这是本阶段唯一"已实现"的部分** —— B3 的 Agent 外壳只需要：
① 调 F1 的导出函数；② 调 E2 的 `verify_citation`；③ 加一个去重步骤。

#### 接口

```python
def manage_references(
    entries: Sequence[BibEntry],
    *,
    library: CitationLibrary | None = None,
    resolver: DoiResolver | None = None,
) -> ReferenceReport: ...
```

## 框架图

```
TopicSpec
    │
    ▼
B1 文献检索 ──▶ Library（documents 带 hash）
    │                 │
    │                 ├──▶ B2 综述写作 ──▶ survey + claims
    │                 │        ↑
    │                 │   copilot.evidence（逐句绑定 + 四态判定）
    │                 │
    │                 └──▶ B3 引用管理 ──▶ references
    │                              ↑
    │                     copilot.delivery.export（已实现）
    ▼
下游 C / D 阶段
```

## 设计路线

| 步 | 内容 | 产出 | 状态 |
|---|---|---|---|
| 1 | 检索 Agent 外壳（包 `RrfFusion`） | `literature/search.py` | 能力已有 |
| 2 | 引用管理（去重 + 复用导出） | `literature/references.py` | **最易先做** |
| 3 | 文献分组（聚类） | `literature/grouping.py` | 需设计 |
| 4 | 综述写作 + 逐句绑定 | `literature/survey.py` | 需模型 |
| 5 | 引文图扩展 | — | 需外部源，暂不做 |

**建议先做第 2 步**：它是唯一纯确定性、且能立刻复用现成代码的部分。

## 验收标准

1. **综述里每条断言都有 `EvidenceSpan`**；`unsupported` 的句子**不得进入正文**；
2. **`content_hash` 贯穿全程**，下游能反查"这条证据来自哪一版"；
3. **冲突并列而非仲裁**：两份文献结论相反时，综述里两个方向都要出现；
4. **参考文献去重按 DOI / 规范化标题**，不能按字符串相等；
5. **B3 的格式生成与 F1 完全一致**（复用同一实现，不另写一份）；
6. **检索默认配置与消融报告一致**（BM25 ∪ trigram，不启 dense）。

## 关键设计判断

| 判断 | 理由 |
|---|---|
| 综述 100% 强制绑定 | 这是编造文献最集中的地方 |
| 边写边校验 | 写完再验一旦大面积失败，整段要重写 |
| 冲突并列不仲裁 | 领域分歧是综述最有价值的部分 |
| 默认不启 dense | **实测**显著有害（MRR −0.091），不是偏好 |
| B3 复用 F1 的生成器 | 同一件事只能有一份实现 |
| 不做引文图扩展 | 需网络依赖，与"可复现、零外部依赖"冲突 |

## 已知局限

| 局限 | 说明 |
|---|---|
| **无引文图扩展** | 找不到"术语不同但问题相同"的论文 |
| 语料规模受限 | 60 篇语料的结论不能外推到大库 |
| 分组依赖模型 | 聚类质量无法自动验证 |
| 综述质量难以自动评测 | 目前只能查"是否绑定证据"，查不了"写得好不好" |
| 去重对预印本/正式版不敏感 | 同一工作的两个版本可能被当成两篇 |

---

# 第二部分：实现细节与实测

> 代码：`02-literature/src/copilot/literature/` · 测试：`02-literature/tests/test_literature.py`（14 项）
> 入口：`from copilot.literature import run_literature`

## 文件划分

```
src/copilot/literature/
├── models.py   LibraryItem / Library / ReferenceIssue / ReferenceReport
├── search.py   B1 检索 + B3 引用管理
├── survey.py   B2 综述分组与写作计划
└── __init__.py 节点装配 + run_literature
```

## 一次必须做的重构：`BibEntry` 上移到共享层

B3 要用 `BibEntry`，而它原本在 `05-format/e3-export/` 里。
**跨阶段 import 会破坏"阶段独立"** —— 于是按既有原则
「被多方复用的地基放 00-core」上移到 `copilot/bib.py`。

原位置改成一层转发 shim，所有旧 import 照旧可用：

```
00-core/src/copilot/bib.py       ← 真正的实现
05-format/e3-export/.../export/bib.py     ← from copilot.bib import *（转发）
```

**判断标准**：一个东西被两个阶段用到，它就不属于任何单一阶段。

## 各节点的实现要点

### B1 文献检索 `search.py`

`Bm25Retriever ∪ CharTrigramRetriever` → RRF 融合 → 文档级 max-pool。

**不启用 dense**：`reports/retrieval_ablation.md` 实测 dense 融合
**显著有害**（MRR −0.091，p(BH)=0.010）。这不是偏好，是有数据支撑的决定。

每篇文献都带 `content_hash` —— 没有 hash 的会被报 ERROR，
因为没有它就无法确认后续引用的是同一版本。

### B3 引用管理 `search.py`

**去重不能按字符串相等**：

```
张三. 一种方法[J]. 计算机学报, 2020.
张三 . 一种方法 [J]. 计算机学报 , 2020.
```

按字符串比等于"全都不重复"。正确做法是**按标识**：

```python
def dedupe_key(entry):
    if entry.doi: return f"doi:{entry.doi.lower()}"      # DOI 优先
    return f"title:{normalize_title(entry.title)}"        # 其次规范化标题
```

**为什么 DOI 优先**：同名论文真实存在（不同年份、不同团队撞名），
只有 DOI 能唯一确定一篇文献。

生成部分**复用 `copilot.bib`**（BibTeX + GB/T 7714），核实部分**复用
`copilot.evidence.verify_citation`** —— 同一件事只有一份实现。

**缺文献库时**：跳过核实，但 note 里明确写
「**跳过引用核实**（这不等于「引用都没问题」）」。
静默跳过会让人以为查过了 —— 那是最危险的情况。

### B2 综述分组 `survey.py`

按「来源 + 年代」分组，给出写作要点与**需要并列呈现的分歧**。

**★ 实测踩过的 bug：篇数重复计数**

分组是**多维**的（来源一组、年代一组），同一篇会出现在多个组里。
早期用 `sum(len(g.entries))` 当总篇数 —— **4 篇文献被算成 8 篇**。

修法：单独存 `papers = len(entries)`（去重后的真实篇数）。

## 实测结果

```
阶段 B：✅ 通过（无阻断项）（3 个节点 / 错误 0 / 警告 0 / 提示 2）
输入 3 条参考文献（其中 2 条同 DOI）
  → B3 去重：in=3 out=2 removed=1
  → B2 综述骨架：3 组 / 2 篇
```

## 已知局限（补充）

| 局限 | 说明 |
|---|---|
| 无引文图扩展 | 需外部数据源（Semantic Scholar / OpenAlex），与零依赖冲突 |
| 分组维度只有来源/年代 | 按方法流派分组需要读全文，不在本节点假装能做 |
| B2 只给骨架 | 正文写作需要模型 + 逐句证据绑定，未接入 |
| 去重对预印本/正式版不敏感 | 同一工作的两个版本 DOI 不同，会被当成两篇 |
