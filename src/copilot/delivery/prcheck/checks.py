"""F2 · 具体的检查实现。

五项检查分两类
--------------
**F2 独有的**（E 阶段不做）：
- 占位符残留
- 敏感信息
- 匿名性

**复用 E1 的**（只重新定级）：
- 结构完整性
- 字数

为什么敏感信息/匿名性放在 F2 而不放 E1
----------------------------------------
按**判据的性质**归属，而不是按检测手段归属。

"正文里有没有你的手机号"不是格式问题，是**交付风险**问题。
E1 的视角是"格式合不合规"，把它塞进 E1 会让 E1 的规则集失去内聚性。
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from copilot.quality.format.docmodel import DocModel
from copilot.quality.format.report import Issue, Severity
from copilot.quality.format.template import Template

__all__ = [
    "check_placeholders",
    "check_sensitive",
    "check_anonymity",
    "check_structure",
    "check_word_count",
    "PLACEHOLDER_PATTERNS",
    "SENSITIVE_PATTERNS",
    "ANONYMITY_PATTERNS",
]

RULE = "prcheck"

#: 占位符模式。**按"交付事故"的常见程度排序**。
PLACEHOLDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("TODO 标记", re.compile(r"\b(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)),
    ("中文占位", re.compile(r"待补|待完善|此处补充|待填入|此处省略")),
    ("空占位符", re.compile(r"【\s*】|\[\s*\]|〔\s*〕|（\s*）")),
    ("疑问占位", re.compile(r"？？？|\?\?\?")),
    ("待定标记", re.compile(r"待定|TBD", re.IGNORECASE)),
)

#: 敏感信息模式。``(说明, 正则, 严重级)``
SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str], Severity], ...] = (
    (
        "邮箱地址",
        re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}"),
        Severity.ERROR,
    ),
    (
        "手机号",
        re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        Severity.ERROR,
    ),
    (
        "身份证号",
        re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
        Severity.ERROR,
    ),
    (
        "密钥形态",
        re.compile(r"\b(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"),
        Severity.ERROR,
    ),
    (
        "绝对路径",
        re.compile(r"[A-Za-z]:\\[\\\w\-./ ]{3,}|/(?:home|Users)/[\w.-]+"),
        Severity.WARNING,
    ),
    (
        "内网地址",
        re.compile(r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
        Severity.WARNING,
    ),
)

#: 匿名性模式（仅在 ``anonymity_required=True`` 时启用）
ANONYMITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("基金资助号", re.compile(r"(国家|省|市)?(自然科学基金|社科基金|重点研发计划)[^。\n]{0,20}(No\.?|编号|批准号)\s*[:：]?\s*[\w-]{4,}")),
    ("个人主页/仓库", re.compile(r"(github\.com|gitee\.com|gitlab\.com|https?://[^\s]*(?:homepage|~)[\w/]*)", re.IGNORECASE)),
    ("机构名称", re.compile(r"[\u4e00-\u9fff]{2,12}(大学|学院|研究院|研究所|实验室)")),
    ("第一人称自引", re.compile(r"我们(?:在|之前|此前)(?:的)?(?:工作|研究|论文|文章)")),
)

#: 致谢/鸣谢章节标题
_ACK_TITLES = ("致谢", "鸣谢", "acknowledgements", "acknowledgments")


def _locate_line(doc: DocModel, pattern: re.Pattern[str], *, start_line: int = 0) -> int | None:
    for i in range(start_line, len(doc.lines)):
        if pattern.search(doc.lines[i]):
            return i
    return None


def check_placeholders(doc: DocModel) -> list[Issue]:
    """PRE001 占位符残留。

    **最高频、最低级、最丢人的交付事故。**

    作者自己读一遍很难发现 —— 因为大脑会自动跳过它认为
    "我知道这里要写什么"的地方。机器不会。
    """
    issues: list[Issue] = []
    for label, pattern in PLACEHOLDER_PATTERNS:
        for i, line in enumerate(doc.lines):
            m = pattern.search(line)
            if not m:
                continue
            issues.append(
                Issue(
                    code="PRE001",
                    severity=Severity.ERROR,
                    rule=RULE,
                    line=i,
                    message=f"发现{label}：{m.group(0)!r}",
                    fix="补完内容或删掉标记；提交前务必清空",
                )
            )
            break  # 同一类只报第一处，避免刷屏
    return issues


def check_sensitive(doc: DocModel) -> list[Issue]:
    """PRE002 敏感信息。

    这类事故**一旦提交就收不回**，而且**完全可规则检测** —— 没有理由不做。
    """
    issues: list[Issue] = []
    for label, pattern, severity in SENSITIVE_PATTERNS:
        for i, line in enumerate(doc.lines):
            m = pattern.search(line)
            if not m:
                continue
            issues.append(
                Issue(
                    code="PRE002",
                    severity=severity,
                    rule=RULE,
                    line=i,
                    message=f"正文中出现{label}：{_mask(m.group(0))}",
                    fix=f"删除或脱敏这条{label}；正文不应含个人信息与本地环境信息",
                )
            )
            break
    return issues


def _mask(text: str) -> str:
    """脱敏后再展示 —— **报告本身也不能泄露**。

    一条"发现手机号 13800138000"的报告，如果把号码原样打出来，
    等于把泄露面从正文扩大到了报告。
    """
    t = text.strip()
    if len(t) <= 4:
        return "*" * len(t)
    return t[:2] + "*" * max(1, len(t) - 4) + t[-2:]


def check_anonymity(doc: DocModel) -> list[Issue]:
    """PRE003 匿名性（双盲投稿）。

    为什么按**章节位置**判定致谢：全文搜"感谢"会误报
    （「感谢审稿人的意见」在正文里是正常的）。
    """
    issues: list[Issue] = []

    for section in doc.flat_sections():
        title = section.title.strip().lower()
        if not any(t in title for t in _ACK_TITLES):
            continue
        start, end = section.span
        body = "\n".join(doc.lines[start + 1 : end]).strip()
        if body:
            issues.append(
                Issue(
                    code="PRE003",
                    severity=Severity.ERROR,
                    rule=RULE,
                    line=section.line,
                    message=f"双盲投稿时「{section.title}」章节非空，会暴露作者身份",
                    fix="双盲评审请删除致谢，或改为终稿版本再补回",
                )
            )

    for label, pattern in ANONYMITY_PATTERNS:
        i = _locate_line(doc, pattern)
        if i is None:
            continue
        m = pattern.search(doc.lines[i])
        issues.append(
            Issue(
                code="PRE003",
                severity=Severity.ERROR if label == "基金资助号" else Severity.WARNING,
                rule=RULE,
                line=i,
                message=f"双盲投稿风险：正文出现{label}（{m.group(0)[:24] if m else ''}）",
                fix="双盲版本中删除或替换为占位表述",
            )
        )
    return issues


def check_structure(doc: DocModel, template: Template) -> list[Issue]:
    """PRE004 结构完整性 —— **复用 E1 的规则，重新定级**。

    同一次检测，两种判决：E1 关心"格式合不合规"，
    F2 关心"带着这个能不能交"。
    """
    from copilot.quality.format import check_document

    from copilot.delivery.prcheck.manifest import SEVERITY_REGRADE

    e1_report = check_document(doc, template)
    out: list[Issue] = []
    for issue in e1_report.issues:
        new_severity = SEVERITY_REGRADE.get(issue.code, issue.severity)
        out.append(
            Issue(
                code=issue.code,
                severity=new_severity,
                rule=f"{issue.rule}→prcheck",
                line=issue.line,
                message=issue.message,
                fix=issue.fix,
            )
        )
    return out


def check_word_count(doc: DocModel, min_words: int | None) -> list[Issue]:
    """PRE005 字数。"""
    if min_words is None:
        return []
    words = doc.body_words
    if words >= min_words:
        return []
    return [
        Issue(
            code="PRE005",
            severity=Severity.WARNING,
            rule=RULE,
            message=f"正文 {words} 字，低于要求的 {min_words} 字",
            fix="补充论述；注意统计已排除参考文献",
        )
    ]
