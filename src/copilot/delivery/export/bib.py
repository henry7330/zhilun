"""F1 · 参考文献导出 —— **转发的转发层**。

真正的实现已上移到 ``copilot.bib``（共享层），因为 B3 引用管理也要用它，
而跨阶段 import 会破坏"阶段独立"。

保留这个文件是为了**所有旧 import 一行都不用改**：

    from copilot.delivery.export import BibEntry, to_bibtex   # 照旧可用
    from copilot.delivery.export.bib import escape_latex     # 照旧可用
"""

from copilot.bib import (  # noqa: F401
    LATEX_ESCAPES,
    BibEntry,
    escape_latex,
    from_reference,
    render_bibtex_file,
    render_gbt7714_list,
    to_bibtex,
    to_gbt7714,
)

__all__ = [
    "BibEntry",
    "from_reference",
    "to_bibtex",
    "to_gbt7714",
    "render_bibtex_file",
    "render_gbt7714_list",
    "escape_latex",
    "LATEX_ESCAPES",
]