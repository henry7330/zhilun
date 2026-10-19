"""子部分 ① 文档模型：把 Markdown 草稿解析成结构化视图。

这一层是**防腐层（anti-corruption layer）**
-----------------------------------------
六条规则都**只看这里产出的模型，永远不看原始 Markdown**。

好处有三：
1. **不重复**：章节识别逻辑只写一遍，而不是六条规则各写一遍；
2. **不脆弱**：改一处解析逻辑，规则不受影响；反过来规则调整也不必碰解析；
3. **能做跨规则检查**：「图 3 被正文引用了，但全篇没有图 3」需要**同时**
   拿到题注列表与引用列表才判得出来 —— 各自扫原文是做不到的。

将来支持 DOCX / LaTeX 时，只需再写一个解析器产出**同样的 Dataclass**，
六条规则一行都不用改。

关键约定：**每个元素都记录行号**。没有行号的报告等于没有报告 ——
用户拿到"章节缺失"却不知道去哪看。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Section",
    "Caption",
    "CrossRef",
    "Reference",
    "DocModel",
    "parse_markdown",
    "count_words",
]

# ── 识别模式 ──

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)[.、\s]+(\S.*)$")

#: 题注：# 后面的正文里出现「图 1」「表 2-1」「Figure 3」「Table 3」
_CAPTION = re.compile(
    r"^\s*(?:[-*+]\s*)?(?:\*\*)?\s*"
    r"(图|表|Figure|Table|Fig\.)\s*"
    r"(\d+(?:[-.]\d+)*)"
    r"\s*(?:\*\*)?\s*[:：.、]?\s*(.*)$",
    re.IGNORECASE,
)

#: 行尾的中文句末标点。以它结尾的行是**正文句子**，不是题注。
#:
#: 为什么需要这条：``图 3 给出了两类方法的对比结果。`` 这句**也以「图 3」开头**，
#: 与题注 ``图 3 方法对比`` 在词法上无法区分。实测踩过 ——
#: 误判为题注后，它不再被算作"引用"，于是反向误报出「图 3 未被正文引用」。
#:
#: 判据：中文题注按惯例**不加句号**。这条规则对中文可靠；
#: 英文题注可能以句点结尾，见下方 ``_CAPTION_TEXT_MAX`` 的兜底说明。
_PROSE_TAIL = re.compile(r"[。！？；]\s*$")

#: 题注文本长度上限：超过这个长度基本是在写句子而不是给图起名。
#: 这是给英文（题注可能以 ``.`` 结尾）留的兜底判据。
_CAPTION_TEXT_MAX = 50

#: 正文中的交叉引用。
#:
#: ⚠️ 这里**不能**用 ``(?<![\w])`` 做前瞻排除（实测踩过）：
#: Python 的 ``\w`` **匹配汉字**，而中文里"图"前面几乎总是汉字
#: （"如图 3 所示"），于是前瞻恒不成立、**所有中文交叉引用都扫不到**。
#: 后果是双向的：「引用了不存在的图」漏检，同时「图确实被引用」被误报。
#:
#: 这里只排除**紧邻拉丁字母**的情况（如 ``xFigure1``），
#: 中文字符一律允许 —— 因为后面的 ``\s*(\d+)`` 已经把"图书""意图"这类
#: 无数字的复合词自然挡掉了。
_CROSSREF = re.compile(
    r"(?<![A-Za-z])(图|表|Figure|Table|Fig\.)\s*(\d+(?:[-.]\d+)*)",
    re.IGNORECASE,
)

#: 参考文献条目：[1] ... 或 1. ...
_REF_ENTRY = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)[.、])\s*(.+)$")

#: 默认的参考文献章节标题
DEFAULT_REF_TITLES = ("参考文献", "references", "bibliography", "引用文献")

#: 正文字数统计：CJK 字符按字算，西文按词算
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_WESTERN_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def count_words(text: str) -> int:
    """统计字数：中文字按字计，西文按词计。

    为什么不统一按空格切：中文没有词间空格，按空格切会把一整段算成 1 个"词"，
    导致中文字数被严重低估（一篇 8000 字的中文论文会被算成几十）。
    """
    return len(_CJK.findall(text)) + len(_WESTERN_WORD.findall(text))


@dataclass
class Section:
    """章节节点。"""

    level: int
    title: str
    line: int
    number: str | None = None
    children: list["Section"] = field(default_factory=list)
    #: 该章节覆盖的行区间（含标题行，不含子章节）
    span: tuple[int, int] = (0, 0)

    @property
    def normalized_title(self) -> str:
        """去掉编号后的标题，便于与模板比对。

        「1.2 研究方法」与「研究方法」应当视为同一章节 ——
        否则模板里写「研究方法」就永远匹配不上带编号的标题。
        """
        return self.number and self.title or self.title


@dataclass
class Caption:
    """图表题注。"""

    kind: str          # "figure" | "table"
    number: str        # "1" / "2-1"
    text: str
    line: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.number)


@dataclass
class CrossRef:
    """正文中对图表的引用。"""

    kind: str
    number: str
    line: int
    raw: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.number)


@dataclass
class Reference:
    """一条参考文献（解析后的结构化形式）。"""

    index: int | None
    raw: str
    line: int
    ref_type: str | None = None      # J / M / D / C / EB/OL ...
    authors: list[str] = field(default_factory=list)
    title: str | None = None
    year: int | None = None
    venue: str | None = None

    @property
    def author_text(self) -> str:
        return ", ".join(self.authors) if self.authors else ""


@dataclass
class DocModel:
    """整篇草稿的结构化视图。"""

    path: Path
    lines: list[str]
    sections: list[Section]
    references: list[Reference]
    captions: list[Caption]
    cross_refs: list[CrossRef]
    body_text: str
    reference_titles: tuple[str, ...] = DEFAULT_REF_TITLES

    @property
    def body_words(self) -> int:
        return count_words(self.body_text)

    def flat_sections(self) -> list[Section]:
        """按文档顺序展开章节树。"""
        out: list[Section] = []

        def walk(nodes: list[Section]) -> None:
            for n in nodes:
                out.append(n)
                walk(n.children)

        walk(self.sections)
        return out

    def section_titles(self) -> list[str]:
        return [s.title for s in self.flat_sections()]

    def captions_of(self, kind: str) -> list[Caption]:
        return [c for c in self.captions if c.kind == kind]


# ─────────────────────── 解析 ───────────────────────


def _looks_like_caption(remainder: str) -> bool:
    """判断「图 N」后面的那段文字是**题注**还是**正文句子**。

    见 ``_PROSE_TAIL`` 与 ``_CAPTION_TEXT_MAX`` 的说明 ——
    这是词法层面无法完全解决的问题，只能靠启发式，因此这里如实标注为启发式，
    而不是假装它是确定性的。
    """
    text = remainder.strip()
    if _PROSE_TAIL.search(text):
        return False
    if len(text) > _CAPTION_TEXT_MAX:
        return False
    return True


def _classify_caption_kind(token: str) -> str:
    return "table" if token.lower() in ("表", "table") else "figure"


def _normalize_ref_title(title: str) -> str:
    """去掉标题里的编号与空白，便于匹配「参考文献」章节。"""
    t = title.strip().lower()
    t = re.sub(r"^\d+(?:\.\d+)*[.、\s]+", "", t)
    return t.strip(" .:：")


def _parse_reference(raw: str, index: int | None, line: int) -> Reference:
    """从一条参考文献的原文里抽结构化字段。

    这里只做**保守抽取**：宁可抽不到（字段为 None），也不要抽错。
    抽错会让格式规则报出**假的错误**，比漏报更伤信任。
    """
    ref = Reference(index=index, raw=raw, line=line)

    # 文献类型标识：[J] [M] [D] [C] [EB/OL] ...
    type_match = re.search(r"\[([A-Za-z]+(?:/[A-Za-z]+)?)\]", raw)
    if type_match:
        ref.ref_type = type_match.group(1).upper()

    # 年份：优先取 4 位数字且落在合理区间
    years = [int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", raw)]
    if years:
        ref.year = years[0]

    # 作者：第一个句点之前的片段，按逗号/分号切
    head = re.split(r"[.。]", raw, maxsplit=1)[0]
    if head and not re.fullmatch(r"[\d\s\[\]]+", head):
        parts = [p.strip() for p in re.split(r"[,，;；]", head) if p.strip()]
        if parts:
            ref.authors = parts

    # 题名：类型标识之前的最后一段
    if type_match:
        before = raw[: type_match.start()]
        pieces = [p.strip() for p in re.split(r"[.。]", before) if p.strip()]
        if pieces:
            ref.title = pieces[-1]

    # 刊名/出版者：类型标识之后到年份之前
    after = raw[type_match.end() :] if type_match else raw
    venue_match = re.match(r"\s*[.。]?\s*([^,，.。]+)", after)
    if venue_match:
        candidate = venue_match.group(1).strip()
        if candidate and not candidate.isdigit():
            ref.venue = candidate

    return ref


def parse_markdown(text: str, *, path: str | Path = "<memory>") -> DocModel:
    """解析 Markdown 草稿。

    解析顺序刻意设计为：**先识别参考文献章节的边界，再解析其余部分**。
    否则参考文献里的「图 1」这类字样会被误当成正文交叉引用，
    造成大量假报警 —— 这是规则引擎最容易踩的坑之一。
    """
    lines = text.splitlines()
    doc_path = Path(path)

    # ── 第一遍：找标题，并确定参考文献章节的行区间 ──
    headings: list[tuple[int, str, int]] = []  # (level, title, line_no)
    for i, line in enumerate(lines):
        m = _HEADING.match(line)
        if m:
            headings.append((len(m.group(1)), m.group(2).strip(), i))

    ref_start: int | None = None
    ref_level = 99
    for level, title, line_no in headings:
        if _normalize_ref_title(title) in DEFAULT_REF_TITLES:
            ref_start = line_no
            ref_level = level
            break

    ref_end = len(lines)
    if ref_start is not None:
        for level, _title, line_no in headings:
            if line_no > ref_start and level <= ref_level:
                ref_end = line_no
                break

    def in_ref_block(line_no: int) -> bool:
        return ref_start is not None and ref_start <= line_no < ref_end

    # ── 第二遍：构建章节树 ──
    roots: list[Section] = []
    stack: list[Section] = []
    for level, title, line_no in headings:
        num_match = _NUMBERED_HEADING.match(title)
        number = num_match.group(1) if num_match else None
        node = Section(level=level, title=title, line=line_no, number=number)
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    flat = []

    def walk(nodes: list[Section]) -> None:
        for n in nodes:
            flat.append(n)
            walk(n.children)

    walk(roots)
    for i, node in enumerate(flat):
        end = flat[i + 1].line if i + 1 < len(flat) else len(lines)
        node.span = (node.line, end)

    # ── 第三遍：题注 / 交叉引用 / 参考文献 ──
    captions: list[Caption] = []
    captions_by_line: dict[int, Caption] = {}
    for i, line in enumerate(lines):
        if in_ref_block(i):
            continue
        m = _CAPTION.match(line)
        if m and _looks_like_caption(m.group(3) or ""):
            cap = Caption(
                kind=_classify_caption_kind(m.group(1)),
                number=m.group(2),
                text=(m.group(3) or "").strip(),
                line=i,
            )
            captions.append(cap)
            captions_by_line[i] = cap

    cross_refs: list[CrossRef] = []
    for i, line in enumerate(lines):
        if in_ref_block(i):
            continue
        if i in captions_by_line:
            # 题注行自身不算"引用"
            continue
        for m in _CROSSREF.finditer(line):
            cross_refs.append(
                CrossRef(
                    kind=_classify_caption_kind(m.group(1)),
                    number=m.group(2),
                    line=i,
                    raw=m.group(0),
                )
            )

    references: list[Reference] = []
    if ref_start is not None:
        for i in range(ref_start, ref_end):
            m = _REF_ENTRY.match(lines[i])
            if not m:
                continue
            idx = int(m.group(1) or m.group(2))
            body = m.group(3).strip()
            if body:
                references.append(_parse_reference(body, idx, i))

    # ── 正文文本（排除参考文献块，用于字数统计）──
    body_lines = [ln for i, ln in enumerate(lines) if not in_ref_block(i)]
    body_text = "\n".join(body_lines)

    return DocModel(
        path=doc_path,
        lines=lines,
        sections=roots,
        references=references,
        captions=captions,
        cross_refs=cross_refs,
        body_text=body_text,
    )
