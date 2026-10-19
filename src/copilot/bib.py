"""参考文献模型与两种导出格式（BibTeX / GB/T 7714）—— **共享层**。

> 本文件原在 ``06-delivery/f1-export/``（F1 文档导出）下。
> B 阶段的 B3 引用管理也要用它 —— **跨阶段 import 会破坏"阶段独立"**，
> 所以按「被多方复用的地基放 00-core」这条既有原则上移到共享层。
> F1 的 ``export/bib.py`` 现在只是一层转发，旧 import 全部照旧可用。

为什么这里要新建一个模型，而不是直接用 E1 的 ``Reference``
----------------------------------------------------------
E1 的 ``Reference`` 是**为格式校验设计的**：只抽了够判断"格式对不对"的字段
（有没有类型标识、有没有年份、作者数超没超）。

导出需要**更全的字段**（卷 / 期 / 页码 / DOI / 出版者）。

**为什么不给 E1 的模型加字段**：那会让 E1 的校验规则依赖它根本不需要的信息。
按用途建模型，而不是建一个大而全的模型。

★ 为什么 LaTeX 转义是这个文件里最重要的部分
--------------------------------------------
标题里一个 ``&`` 就能让整个 ``.bib`` 编译失败，
而 LaTeX 的报错会指向**生成后几百行的 .tex**，极难定位到源头。

更麻烦的是：**这个错误在生成时完全不报错** —— 文件看着"生成成功"了。
所以转义必须做，而且必须做在**生成器**里（而不是写文件时），
这样任何调用路径都得到正确转义。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from copilot.quality.format.docmodel import Reference

__all__ = [
    "BibEntry",
    "from_reference",
    "to_bibtex",
    "to_gbt7714",
    "render_bibtex_file",
    "render_gbt7714_list",
    "escape_latex",
    "LATEX_ESCAPES",
    "normalize_title",
    "dedupe_key",
]

#: LaTeX 特殊字符 → 转义形式。
LATEX_ESCAPES: tuple[tuple[str, str], ...] = (
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
)

#: 逐字符映射表 —— **单遍替换**用。
#:
#: ★ 为什么不能顺序 ``str.replace``（实测踩过）：
#: 逐个替换会让**后一次替换命中前一次产生的文本**。
#: 例如先把 ``\\`` 换成 ``\\textbackslash{}``，随后的 ``{`` → ``\\{``
#: 会把刚插入的那对花括号又转义掉，得到 ``\\textbackslash\\{\\}`` —— 编译失败。
#:
#: 单遍逐字符映射根本不存在"命中自己产物"的问题：每个原字符只看一次。
_LATEX_CHAR_MAP: dict[str, str] = dict(LATEX_ESCAPES)

#: E1 的文献类型标识 → BibTeX 条目类型
_TYPE_TO_BIBTEX: dict[str, str] = {
    "J": "article",
    "M": "book",
    "C": "inproceedings",
    "D": "phdthesis",
    "R": "techreport",
    "S": "standard",
    "P": "patent",
    "N": "newspaper",
    "EB": "misc",
    "EB/OL": "misc",
    "DB/OL": "misc",
    "J/OL": "article",
    "M/OL": "book",
    "C/OL": "inproceedings",
}

#: 中文文献类型标识（GB/T 7714 用）
_GB_MARK: dict[str, str] = {
    "J": "[J]",
    "M": "[M]",
    "C": "[C]",
    "D": "[D]",
    "R": "[R]",
    "S": "[S]",
    "P": "[P]",
    "N": "[N]",
    "EB": "[EB]",
    "EB/OL": "[EB/OL]",
    "DB/OL": "[DB/OL]",
    "J/OL": "[J/OL]",
    "M/OL": "[M/OL]",
    "C/OL": "[C/OL]",
}


def escape_latex(text: str) -> str:
    """转义 LaTeX 特殊字符。

    **单遍逐字符映射**：每个原字符只被看一次，因此不存在
    "后一次替换命中前一次产物"的问题（顺序 ``replace`` 会踩这个坑）。

    >>> escape_latex("A & B")
    'A \\\\& B'
    >>> escape_latex("\\\\")
    '\\\\textbackslash{}'
    """
    return "".join(_LATEX_CHAR_MAP.get(ch, ch) for ch in text)


@dataclass
class BibEntry:
    """一条结构化的参考文献。"""

    key: str
    entry_type: str = "article"
    authors: list[str] = field(default_factory=list)
    title: str = ""
    #: 期刊名（article）或论文集名（inproceedings）或出版社（book）
    venue: str = ""
    year: int | None = None
    volume: str | None = None
    number: str | None = None
    pages: str | None = None
    doi: str | None = None
    publisher: str | None = None
    #: 中文 GB/T 7714 的类型标识（J / M / C / D ...）
    cn_type: str = "J"
    #: 原始的 E1 Reference，便于追溯
    source_index: int | None = None

    @property
    def first_author(self) -> str:
        return self.authors[0] if self.authors else ""


def _make_key(authors: Sequence[str], year: int | None, title: str) -> str:
    """生成 BibTeX 引用键：``首作者姓 + 年份 + 题名首个实词``。

    为什么不用序号：LaTeX 里 ``\\cite{key}`` 是可读的，
    而 ``\\cite{ref3}`` 在正文里完全看不出指的是哪篇。
    """
    surname = "anonymous"
    if authors:
        first = authors[0].strip()
        # 中文姓名取首字；西文取第一个词
        if re.fullmatch(r"[\u4e00-\u9fff]+", first):
            surname = first[:2]
        else:
            surname = re.split(r"[\s,]+", first)[0] or "anonymous"

    year_part = str(year) if year else "nd"

    word = ""
    for token in re.split(r"[\s\-_]+", title):
        cleaned = re.sub(r"[^A-Za-z\u4e00-\u9fff]", "", token)
        if len(cleaned) >= 3:
            # 中文题名没有空格，整句就是一个 token，所以取短一些
            word = cleaned[:6]
            break

    key = f"{surname}{year_part}{word}".lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", key) or "ref"


def _split_pages(text: str | None) -> tuple[str | None, str | None]:
    """从 "43(2): 1-10" 一类文本里尽力抽出 卷 / 期 / 页。

    只做**保守抽取**：抽不准就留 None。导出一个错字段比缺字段更难查。
    """
    if not text:
        return (None, None)
    m = re.search(r"(\d+)\s*\(\s*(\d+)\s*\)", text)
    if m:
        return (m.group(1), m.group(2))
    return (None, None)


def from_reference(ref: Reference, *, used_keys: set[str] | None = None) -> BibEntry:
    """把 E1 的 ``Reference`` 转成 ``BibEntry``。

    ``used_keys`` 用于**避免键冲突**（两个同姓同年的作者 + 同题名首词）：
    冲突时追加 ``-2`` / ``-3``。不处理冲突会让 ``\\cite`` 指错文献。
    """
    cn_type = (ref.ref_type or "J").upper()
    entry_type = _TYPE_TO_BIBTEX.get(cn_type, "misc")

    volume, number = _split_pages(ref.raw)

    title = ref.title or ""
    entry = BibEntry(
        key="",
        entry_type=entry_type,
        authors=list(ref.authors),
        title=title.strip(" .。"),
        venue=(ref.venue or "").strip(" .。,，"),
        year=ref.year,
        volume=volume,
        number=number,
        cn_type=cn_type,
        source_index=ref.index,
    )
    entry.key = _make_key(entry.authors, entry.year, entry.title)

    if used_keys is not None:
        base = entry.key
        suffix = 2
        while entry.key in used_keys:
            entry.key = f"{base}-{suffix}"
            suffix += 1
        used_keys.add(entry.key)

    return entry


def _author_field(authors: Sequence[str]) -> str:
    """BibTeX 用 ``and`` 连接作者。"""
    return " and ".join(a.strip() for a in authors if a.strip())


def to_bibtex(entry: BibEntry, *, indent: str = "  ") -> str:
    """生成一条 BibTeX 记录。**所有文本字段都做 LaTeX 转义。**"""
    lines: list[str] = [f"@{entry.entry_type}{{{entry.key},"]
    fields: list[tuple[str, str]] = []

    if entry.authors:
        fields.append(("author", _author_field(entry.authors)))
    if entry.title:
        fields.append(("title", entry.title))

    if entry.entry_type == "article":
        if entry.venue:
            fields.append(("journal", entry.venue))
    elif entry.entry_type in ("inproceedings", "incollection"):
        if entry.venue:
            fields.append(("booktitle", entry.venue))
    elif entry.entry_type in ("book", "phdthesis", "techreport"):
        if entry.publisher:
            fields.append(("publisher", entry.publisher))
        elif entry.venue:
            fields.append(("institution" if entry.entry_type == "techreport" else "publisher", entry.venue))
    else:
        if entry.venue:
            fields.append(("howpublished", entry.venue))

    if entry.year is not None:
        fields.append(("year", str(entry.year)))
    if entry.volume:
        fields.append(("volume", entry.volume))
    if entry.number:
        fields.append(("number", entry.number))
    if entry.pages:
        fields.append(("pages", entry.pages.replace("-", "--")))
    if entry.doi:
        fields.append(("doi", entry.doi))

    for name, value in fields:
        lines.append(f"{indent}{name} = {{{escape_latex(value)}}},")

    lines.append("}")
    return "\n".join(lines)


def to_gbt7714(entry: BibEntry) -> str:
    """生成 GB/T 7714 格式的参考文献条目（不含序号）。

    策略：**能拼的字段都拼，缺的不补造**。
    宁可格式略简，也不要在参考文献里编造卷期页。
    """
    parts: list[str] = []

    if entry.authors:
        authors = list(entry.authors)
        # GB/T 7714：超过 3 位用「等」
        if len(authors) > 3:
            authors = authors[:3] + ["等"]
        parts.append(", ".join(authors) + ".")

    title = entry.title or "（缺题名）"
    mark = _GB_MARK.get(entry.cn_type, "[J]")
    parts.append(f"{title}{mark}.")

    tail: list[str] = []
    if entry.venue:
        tail.append(entry.venue)
    if entry.year is not None:
        tail.append(str(entry.year))

    volume_part = ""
    if entry.volume:
        volume_part = entry.volume
        if entry.number:
            volume_part += f"({entry.number})"
    if volume_part:
        tail.append(volume_part)

    if tail:
        seg = ", ".join(tail)
        if entry.pages:
            seg += f": {entry.pages}"
        parts.append(seg + ".")

    if entry.doi:
        parts.append(f"DOI: {entry.doi}.")

    return " ".join(parts)


def render_bibtex_file(entries: Iterable[BibEntry]) -> str:
    """渲染整个 ``.bib`` 文件。"""
    blocks = [to_bibtex(e) for e in entries]
    return "\n\n".join(blocks) + "\n" if blocks else ""


def render_gbt7714_list(entries: Sequence[BibEntry]) -> str:
    """渲染 GB/T 7714 参考文献列表（带 ``[n]`` 序号）。"""
    return "\n".join(
        f"[{i}] {to_gbt7714(e)}" for i, e in enumerate(entries, start=1)
    )


# ─────────────────── 文献身份判定（共享层） ───────────────────
#
# 为什么放在这里而不是 B 阶段：
# **D 阶段的「文献引用」节点也要用它**（判断"正文里引的这篇在不在文献库里"），
# 而阶段之间不该互相 import。凡是两处以上要用的判定，都放共享层。

_PUNCT_RE = re.compile(
    r"""[\s\u3000\-_·.,;:!?、，。；：！？"'“”‘’()（）\[\]【】《》<>]+"""
)


def normalize_title(title: str) -> str:
    """标题规范化：去空白与标点、转小写。用于去重与身份比对。"""
    return _PUNCT_RE.sub("", (title or "").lower())


def dedupe_key(entry: BibEntry) -> str:
    """文献身份键：**DOI 优先，其次规范化标题**。

    为什么 DOI 优先：同名论文真实存在（不同年份、不同团队撞名），
    只有 DOI 能唯一确定一篇文献。
    """
    doi = (getattr(entry, "doi", None) or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    return "title:" + normalize_title(getattr(entry, "title", "") or "")
