"""子部分 ⑤ 图表与交叉引用规则。

为什么这一类问题值得单独检查
----------------------------
「正文写着『见图 3』，但全文只有图 1 和图 2」—— 这种问题
**人工必须通读全文才能发现**，而作者在反复增删图表之后最容易漏掉。
它属于「全局一致性」问题：任何**单点阅读**都看不出错，只有把
「题注清单」与「引用清单」**放在一起比对**才暴露。

这正是机器擅长、人容易累的场景，也是本子部分存在的核心理由。

依赖跨清单比对，说明为什么文档模型必须先行
------------------------------------------
本规则需要同时拿到 `doc.captions` 与 `doc.cross_refs`。
如果按"每条规则各自扫原文"的写法，这条规则要自己重新解析一遍全文
（而且还得自己判断哪一行是题注、哪一行是引用）——
文档模型把这些脏活做完了，规则才能只表达**业务判断**。
"""

from __future__ import annotations

import re
from collections import defaultdict

from copilot.quality.format.docmodel import Caption, DocModel
from copilot.quality.format.report import Issue, Severity
from copilot.quality.format.template import Template

__all__ = ["check_figures"]

RULE = "figures"

_KIND_CN = {"figure": "图", "table": "表"}


def _parts(number: str) -> list[int]:
    """把「2-1」「2.1」「3」拆成数字序列。"""
    return [int(p) for p in re.split(r"[-.]", number) if p.isdigit()]


def _group_key(number: str) -> tuple[int, ...]:
    """分组键：取编号的**前缀部分**。

    「图 2-1 / 2-2 / 2-3」应各自在"第 2 章"这一组里连续，
    而不是要求全文的图编号从 1 连续到 N。
    中文论文按章编号（2-1）与全文连续编号（1,2,3）都常见，
    所以两种都要支持。
    """
    parts = _parts(number)
    return tuple(parts[:-1])


def _check_continuity(captions: list[Caption]) -> list[Issue]:
    """检查某一类图表的编号是否连续。"""
    issues: list[Issue] = []
    groups: dict[tuple[int, ...], list[Caption]] = defaultdict(list)
    for c in captions:
        groups[_group_key(c.number)].append(c)

    for _key, items in groups.items():
        # 按编号排序后再比对，避免"文档里顺序就是对的"这种假设
        items = sorted(items, key=lambda c: _parts(c.number))
        seq = [_parts(c.number)[-1] for c in items]
        expected = list(range(1, len(seq) + 1))
        if seq != expected:
            kind_cn = _KIND_CN.get(items[0].kind, items[0].kind)
            issues.append(
                Issue(
                    code="FIG001",
                    severity=Severity.WARNING,
                    rule=RULE,
                    line=items[0].line,
                    message=(
                        f"{kind_cn}编号不连续：实际为 {seq}，应为 {expected}"
                    ),
                    fix="统一重新编号（题注与正文引用都要改）",
                )
            )
    return issues


def check_figures(doc: DocModel, template: Template) -> list[Issue]:
    issues: list[Issue] = []

    if template.require_continuous_figure_numbering:
        for kind in ("figure", "table"):
            issues.extend(_check_continuity(doc.captions_of(kind)))

    declared = {c.key for c in doc.captions}
    referenced = {r.key for r in doc.cross_refs}

    # ── FIG002 正文引用了不存在的图表 ──
    # 严重级为 ERROR：这是**实打实的错误**，读者按图索骥会找不到东西
    for ref in doc.cross_refs:
        if ref.key not in declared:
            kind_cn = _KIND_CN.get(ref.kind, ref.kind)
            issues.append(
                Issue(
                    code="FIG002",
                    severity=Severity.ERROR,
                    rule=RULE,
                    line=ref.line,
                    message=f"正文引用了不存在的{kind_cn} {ref.number}（原文：{ref.raw}）",
                    fix=f"确认{kind_cn} {ref.number} 是否存在；若已删除请同步修改正文引用",
                )
            )

    # ── FIG003 图表存在但正文从未引用 ──
    # 严重级为 INFO 而不是 WARNING：确实可能是疏漏，也可能是排版需要，
    # 由作者判断。规则不该替用户做这种决定，只提示。
    for cap in doc.captions:
        if cap.key not in referenced:
            kind_cn = _KIND_CN.get(cap.kind, cap.kind)
            issues.append(
                Issue(
                    code="FIG003",
                    severity=Severity.INFO,
                    rule=RULE,
                    line=cap.line,
                    message=f"{kind_cn} {cap.number} 未被正文引用",
                    fix=f"在正文中引用一次（如「如{kind_cn} {cap.number} 所示」）",
                )
            )

    return issues
