"""格式校验的三类规则。

每个规则模块都导出同一种签名的函数::

    check_xxx(doc: DocModel, template: Template) -> list[Issue]

统一的签名让 :mod:`copilot.quality.format.checker` 可以简单地遍历调用，
新增规则只需写一个函数并登记进 ``DEFAULT_RULES``。
"""

from copilot.quality.format.rules.figures import check_figures
from copilot.quality.format.rules.reference import check_references
from copilot.quality.format.rules.structure import check_structure

__all__ = ["check_structure", "check_references", "check_figures"]
