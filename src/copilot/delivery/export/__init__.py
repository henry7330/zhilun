"""F1 · 文档导出。

层级定位::

    src/copilot
    └── delivery
        └── F1 文档导出       ← 本包
            ├── bib.py     ① 文献模型 ② BibTeX（含 LaTeX 转义）③ GB/T 7714
            ├── latex.py   ④ LaTeX 骨架生成
            └── verify.py  ⑤ 导出一致性校验（回读反查）★

**零依赖**：纯文本生成，不需要 python-docx / LibreOffice。
Word / PDF 导出需要额外依赖，标为规划（见模块 README 的「范围取舍」）。

典型用法::

    from copilot.quality.format import parse_markdown
    from copilot.delivery.export import export_paper

    doc = parse_markdown(open("thesis.md", encoding="utf-8").read())
    result = export_paper(doc)
    print(result.summary())          # 导出结果 + 一致性校验结论
    open("thesis.tex", "w", encoding="utf-8").write(result.tex)
    open("refs.bib", "w", encoding="utf-8").write(result.bibtex)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from copilot.delivery.export.bib import (
    LATEX_ESCAPES,
    BibEntry,
    escape_latex,
    from_reference,
    render_bibtex_file,
    render_gbt7714_list,
    to_bibtex,
    to_gbt7714,
)
from copilot.delivery.export.latex import (
    DEFAULT_PREAMBLE,
    resolve_title,
    to_latex_skeleton,
)
from copilot.delivery.export.verify import (
    ExportDiff,
    ExportVerification,
    verify_bibtex,
    verify_latex,
)
from copilot.quality.format.docmodel import DocModel

__all__ = [
    "export_paper",
    "ExportResult",
    # 文献
    "BibEntry",
    "from_reference",
    "to_bibtex",
    "to_gbt7714",
    "render_bibtex_file",
    "render_gbt7714_list",
    "escape_latex",
    "LATEX_ESCAPES",
    # LaTeX
    "to_latex_skeleton",
    "DEFAULT_PREAMBLE",
    # 校验
    "verify_latex",
    "verify_bibtex",
    "ExportVerification",
    "ExportDiff",
]


@dataclass
class ExportResult:
    """一次导出的全部产物与校验结论。"""

    tex: str
    bibtex: str
    references_gbt7714: str
    entries: list[BibEntry] = field(default_factory=list)
    latex_check: ExportVerification | None = None
    bib_check: ExportVerification | None = None

    @property
    def ok(self) -> bool:
        """两项校验都通过才算导出成功。"""
        return bool(
            self.latex_check
            and self.latex_check.ok
            and self.bib_check
            and self.bib_check.ok
        )

    def summary(self) -> str:
        lines = [
            f"LaTeX 骨架：{len(self.tex)} 字符",
            f"BibTeX     ：{len(self.entries)} 条",
            f"GB/T 7714  ：{len(self.entries)} 条",
        ]
        if self.latex_check is not None:
            lines.append(f"结构校验   ：{self.latex_check.summary()}")
        if self.bib_check is not None:
            lines.append(f"文献校验   ：{self.bib_check.summary()}")
        lines.append(f"结论       ：{'✅ 通过' if self.ok else '❌ 未通过'}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "tex_chars": len(self.tex),
            "entries": len(self.entries),
            "latex_check": None if self.latex_check is None else self.latex_check.to_dict(),
            "bib_check": None if self.bib_check is None else self.bib_check.to_dict(),
        }


def export_paper(
    doc: DocModel,
    *,
    title: str | None = None,
    verify: bool = True,
) -> ExportResult:
    """一次性导出 LaTeX 骨架 + BibTeX + GB/T 7714 列表，并做一致性校验。

    ``verify=True`` 时会把生成的产物**读回来**与原稿对比 ——
    这是本模块最重要的一步（见 ``verify.py`` 的说明）。
    """
    used_keys: set[str] = set()
    entries = [from_reference(r, used_keys=used_keys) for r in doc.references]

    resolved = resolve_title(doc, title)
    tex = to_latex_skeleton(doc, title=title)
    bibtex = render_bibtex_file(entries)
    gbt = render_gbt7714_list(entries)

    result = ExportResult(
        tex=tex,
        bibtex=bibtex,
        references_gbt7714=gbt,
        entries=entries,
    )
    if verify:
        result.latex_check = verify_latex(tex, doc, document_title=resolved)
        result.bib_check = verify_bibtex(bibtex, doc)
    return result
