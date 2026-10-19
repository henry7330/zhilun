"""子部分 ④ 参考文献格式规则（GB/T 7714 等）。

为什么参考文献值得单独做一个子部分
----------------------------------
它是**格式要求最琐碎、人工核对最痛苦**的部分：30 条文献，每条都要看
文献类型标识、作者数是否该用「等」、年份有没有、标点对不对。
而它又**天然是结构化的**（``作者. 题名[J]. 刊名, 年, 卷(期): 页码.``），
非常适合规则化 —— 投入产出比在本模块里最高。

设计取向：**宁可漏报，不可误报**
--------------------------------
参考文献的写法变体极多（中英文混排、专著带版本、会议论文带 //、
电子资源带 URL 和引用日期）。规则写得激进会产出大量**假错误**，
用户看两次就不信这个工具了。

因此本模块的策略是：
- 只检查**几乎没有歧义**的问题（缺类型标识、序号断裂、缺年份）；
- 拿不准的一律不报；
- 把"作者是否该用等"这类判断的**阈值放进模板**，让用户按学校要求调。
"""

from __future__ import annotations

from copilot.quality.format.docmodel import DocModel, Reference
from copilot.quality.format.report import Issue, Severity
from copilot.quality.format.template import Template

__all__ = ["check_references", "VALID_REF_TYPES", "ETAL_MARKERS"]

RULE = "reference"

#: GB/T 7714 认可的文献类型标识
VALID_REF_TYPES = frozenset(
    {
        "J",       # 期刊
        "M",       # 专著
        "C",       # 论文集
        "D",       # 学位论文
        "N",       # 报纸
        "P",       # 专利
        "S",       # 标准
        "R",       # 报告
        "Z",       # 其他
        "DB",      # 数据库
        "CP",      # 计算机程序
        "EB",      # 电子公告
        "EB/OL",   # 电子公告（联机）
        "DB/OL",
        "J/OL",
        "M/OL",
        "C/OL",
        "A",       # 析出文献
    }
)

#: 「等」/「et al.」的多种写法
ETAL_MARKERS = ("等", "et al", "etc")


def _has_etal(ref: Reference) -> bool:
    blob = (ref.author_text + " " + ref.raw).lower()
    return any(m.lower() in blob for m in ETAL_MARKERS)


def check_references(doc: DocModel, template: Template) -> list[Issue]:
    issues: list[Issue] = []

    # ── REF005 没有参考文献条目 ──
    if not doc.references:
        issues.append(
            Issue(
                code="REF005",
                severity=Severity.ERROR,
                rule=RULE,
                message="未找到任何参考文献条目",
                fix="确认存在「参考文献」章节，且条目格式为「[1] 作者. 题名[J]. ...」",
            )
        )
        return issues

    # ── REF004 序号不连续 ──
    indices = [r.index for r in doc.references if r.index is not None]
    if indices and indices != list(range(1, len(indices) + 1)):
        issues.append(
            Issue(
                code="REF004",
                severity=Severity.ERROR,
                rule=RULE,
                line=doc.references[0].line,
                message=(
                    f"参考文献序号不连续或有重复：实际为 {indices}，"
                    f"应为 1..{len(indices)}"
                ),
                fix="重新编号；正文中的引用编号也要同步修改",
            )
        )

    for ref in doc.references:
        label = f"[{ref.index}]" if ref.index is not None else "（无序号）"

        # ── REF001 缺文献类型标识 ──
        if ref.ref_type is None:
            issues.append(
                Issue(
                    code="REF001",
                    severity=Severity.WARNING,
                    rule=RULE,
                    line=ref.line,
                    message=f"{label} 缺少文献类型标识（如 [J] [M] [D]）",
                    fix="在题名后补上方括号类型标识，例如「...[J]. 计算机学报, 2020」",
                )
            )
        elif ref.ref_type not in VALID_REF_TYPES:
            issues.append(
                Issue(
                    code="REF002",
                    severity=Severity.WARNING,
                    rule=RULE,
                    line=ref.line,
                    message=f"{label} 文献类型标识「[{ref.ref_type}]」不是 GB/T 7714 认可的类型",
                    fix=f"改为：{'、'.join(sorted(VALID_REF_TYPES)[:8])} …",
                )
            )

        # ── REF003 缺年份 ──
        if ref.year is None:
            issues.append(
                Issue(
                    code="REF003",
                    severity=Severity.WARNING,
                    rule=RULE,
                    line=ref.line,
                    message=f"{label} 未识别到出版年份",
                    fix="补充 4 位年份，例如「, 2020, 43(2): 1-10.」",
                )
            )

        # ── REF006 作者数超阈值未用「等」 ──
        # 只在**作者确实抽出来了**且超过阈值时判断，避免抽取失败导致误报
        n = len(ref.authors)
        if n > template.max_authors_before_etal and not _has_etal(ref):
            issues.append(
                Issue(
                    code="REF006",
                    severity=Severity.WARNING,
                    rule=RULE,
                    line=ref.line,
                    message=(
                        f"{label} 列出 {n} 位作者，超过 {template.max_authors_before_etal} 位"
                        f"但未使用「等」/「et al.」"
                    ),
                    fix=f"保留前 {template.max_authors_before_etal} 位，其余用「等」代替",
                )
            )

    return issues
