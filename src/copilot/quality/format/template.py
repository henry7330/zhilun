"""子部分 ② 模板配置：把「合格标准」表达成数据。

为什么配置与引擎必须分离
------------------------
格式要求**因学校而异**：有的要求「文献综述」独立成章，有的并入绪论；
有的参考文献用 GB/T 7714，有的用 APA。

如果把这些要求硬编码进规则代码，会得到两种结局之一：
- 换一所学校就要改代码（校验器无法复用）；
- 或者规则里堆满 ``if 学校 == "A" ... elif 学校 == "B" ...``，
  最后没人敢改。

把「**要求是什么**」抽成 `Template` 数据后，加载不同配置就等于支持不同学校，
而规则引擎保持不变。这也是本项目贯穿始终的做法 ——
**凡是"会变的东西"，都不要写进代码。**
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["Template", "load_template", "save_template", "default_template", "builtin_templates"]

#: 支持的参考文献风格标识（具体规则在 rules/reference.py 里实现）
SUPPORTED_REF_STYLES = ("gbt7714", "apa", "ieee")

#: 内置模板所在目录
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@dataclass
class Template:
    """一份格式要求。"""

    name: str
    #: 必需章节标题，**顺序即要求顺序**
    required_sections: list[str] = field(default_factory=list)
    #: True = 章节顺序必须与 required_sections 一致
    section_order_strict: bool = True
    #: 参考文献风格
    reference_style: str = "gbt7714"
    #: 正文（不含参考文献）字数下限；None 表示不检查
    min_body_words: int | None = None
    #: 作者超过几人后必须用「等」/「et al.」
    max_authors_before_etal: int = 3
    #: 允许的最大标题层级（超过则报「层级过深」）
    max_heading_level: int = 3
    #: 是否要求图表编号连续
    require_continuous_figure_numbering: bool = True
    #: 说明性文字，便于用户知道这份模板的出处
    description: str = ""

    def __post_init__(self) -> None:
        if self.reference_style not in SUPPORTED_REF_STYLES:
            raise ValueError(
                f"不支持的参考文献风格 {self.reference_style!r}；"
                f"可选：{', '.join(SUPPORTED_REF_STYLES)}"
            )
        if self.max_heading_level < 1:
            raise ValueError("max_heading_level 必须 >= 1")

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> Template:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            # 不静默忽略：用户写了错字段名却以为生效，是最难查的问题之一
            raise ValueError(f"模板里有未知字段：{', '.join(sorted(unknown))}")
        return cls(**data)


def default_template() -> Template:
    """通用本科毕业论文模板（作为没有指定模板时的兜底）。"""
    return Template(
        name="通用本科毕业论文",
        required_sections=[
            "摘要",
            "Abstract",
            "绪论",
            "文献综述",
            "研究方法",
            "实验与分析",
            "结论",
            "参考文献",
            "致谢",
        ],
        section_order_strict=True,
        reference_style="gbt7714",
        min_body_words=8000,
        max_authors_before_etal=3,
        max_heading_level=3,
        description="内置兜底模板；真实使用时请以学校下发的格式要求为准并替换。",
    )


def load_template(path: str | Path) -> Template:
    """从 JSON 文件加载模板。"""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"模板文件顶层必须是对象：{p}")
    return Template.from_json(data)


def save_template(template: Template, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(template.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return p


def builtin_templates() -> dict[str, Template]:
    """列出可用的内置模板：先读目录里的 JSON，再用兜底模板补齐。"""
    found: dict[str, Template] = {}
    if _TEMPLATE_DIR.is_dir():
        for f in sorted(_TEMPLATE_DIR.glob("*.json")):
            try:
                t = load_template(f)
            except Exception:
                continue
            found[f.stem] = t
    found.setdefault("default", default_template())
    return found
