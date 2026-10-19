"""P0 冒烟测试：确认包可安装、可导入、子包结构完整。

这一层不测业务逻辑（业务逻辑从 P1 证据绑定层开始有真实测试）。
它的作用是让「环境是否装好」这件事有一个可执行的判据，
而不是靠人肉敲 import 试。
"""

from __future__ import annotations

import importlib

import pytest

#: 期望存在的子包 —— **只列真实存在的**。
#:
#: 早期的 api / guard / mcp / obs / routing / skills 六个占位包已删除：
#: 它们属于**应用层**（服务化 / 工具接入 / 可视化可观测性 / 模型路由），
#: 本项目的目标是「学习技术栈 + 跑通最小验证系统」，
#: 空占位包只会让人误以为「这些能力已经规划好了」—— 那是噪音。
#: 若将来要做，应在文档里明确立项，而不是留一个空文件夹。
SUBPACKAGES = [
    "copilot",            # 共享能力层
    "copilot.evidence",   # 证据层（E2 归属，被多方复用）
    "copilot.retrieval",  # 检索层
    "copilot.memory",     # 记忆可信层
    "copilot.graph",      # 编排层（能力流水线 + 业务流水线）
    "copilot.quality.format",        # E1
    "copilot.quality.report_parse",  # E3
    "copilot.quality.rewrite",       # E4
    "copilot.quality.deai",          # E5
    "copilot.quality.review",        # E6
]


def test_package_importable() -> None:
    pkg = importlib.import_module("copilot")
    assert pkg is not None


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_importable(name: str) -> None:
    """每个分层子包都应可独立导入——骨架缺失会在这里暴露。"""
    assert importlib.import_module(name) is not None


def test_python_version_supported() -> None:
    """需要 Python >=3.11（LangGraph 与类型语法都要求）。"""
    import sys

    assert sys.version_info >= (3, 11), (
        f"需要 Python >=3.11，当前 {sys.version_info.major}.{sys.version_info.minor}"
    )
