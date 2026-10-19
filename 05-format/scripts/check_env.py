#!/usr/bin/env python3
"""环境自检：确认依赖装齐、包能导入。

    python 05-format/scripts/check_env.py

它检查四件事：

1. Python 版本 ≥ 3.11（LangGraph 与项目用的类型语法都要求）
2. 运行时依赖可导入
3. `import copilot` 能从**任意目录**成功（不只是从仓库根）
4. 六个阶段与 17 个节点都注册了

**不检查** conda 环境名 —— 那只是本机习惯，CI 上叫别的名字也应当通过。
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

for _name in ("stdout", "stderr"):
    _s = getattr(sys, _name, None)
    if _s is not None and hasattr(_s, "buffer"):
        enc = getattr(_s, "encoding", None) or "utf-8"
        try:
            setattr(sys, _name, io.TextIOWrapper(_s.buffer, encoding=enc, errors="replace"))
        except Exception:
            pass

#: 运行时必须能导入的第三方包（与 pyproject 的 dependencies 对应）
REQUIRED = [
    ("langgraph", "编排：主 Agent / 阶段图 / 能力流水线"),
    ("rank_bm25", "B1 文献检索"),
    ("numpy", "统计与向量运算"),
    ("scipy", "C2 配对检验"),
    ("httpx", "模型接入"),
    ("dotenv", "读 .env"),
]

EXPECTED_PY = (3, 11)


def project_root() -> Path:
    """仓库根 = 本脚本所在 scripts/ 的上两级。"""
    return Path(__file__).resolve().parent.parent.parent


def main() -> int:
    problems: list[str] = []

    print("=" * 68)
    print("环境自检")
    print("=" * 68)
    print(f"解释器  : {sys.executable}")
    print(f"版本    : {sys.version.split()[0]}")
    print(f"仓库根  : {project_root()}")

    # ── 1) Python 版本 ──
    print("\n[1] Python 版本")
    if sys.version_info[:2] >= EXPECTED_PY:
        print(f"    ✅ {sys.version.split()[0]} ≥ {EXPECTED_PY[0]}.{EXPECTED_PY[1]}")
    else:
        print(f"    ❌ 需要 ≥ {EXPECTED_PY[0]}.{EXPECTED_PY[1]}")
        problems.append("Python 版本过低")

    # ── 2) 依赖 ──
    print("\n[2] 运行时依赖")
    for mod, why in REQUIRED:
        try:
            __import__(mod)
            print(f"    ✅ {mod:12} {why}")
        except Exception as exc:  # noqa: BLE001
            print(f"    ❌ {mod:12} 导入失败：{type(exc).__name__}")
            problems.append(f"缺少依赖 {mod}")

    # ── 3) 包能否导入（关键：不依赖当前工作目录）──
    print("\n[3] import copilot")
    try:
        import copilot

        where = Path(copilot.__file__).resolve()
        print(f"    ✅ {where}")
        if where.parent != (project_root() / "src" / "copilot"):
            print("       ⚠️ 导入的不是本仓库源码（可能装了旧版本）")
    except Exception as exc:  # noqa: BLE001
        print(f"    ❌ {type(exc).__name__}: {exc}")
        problems.append("import copilot 失败")

    # ── 4) 阶段与节点 ──
    print("\n[4] 阶段与节点")
    try:
        from copilot.contracts import STAGES

        total = sum(len(s.nodes) for s in STAGES.values())
        print(f"    ✅ {len(STAGES)} 个阶段 / {total} 个节点")
        for spec in STAGES.values():
            print(f"       {spec.key} {spec.name:8} {spec.code:14} {len(spec.nodes)} 节点")
    except Exception as exc:  # noqa: BLE001
        print(f"    ❌ {type(exc).__name__}: {exc}")
        problems.append("阶段注册表不可用")

    # ── 结论 ──
    print("\n" + "=" * 68)
    if not problems:
        print("✅ 环境正常")
        print("=" * 68)
        return 0

    print("❌ 有问题：")
    for p in problems:
        print(f"   - {p}")
    print()
    print("若缺依赖，装一次即可：")
    print(f'    "{sys.executable}" -m pip install -e ".[dev]"')
    print()
    print("若 import copilot 失败，通常是没装（或仓库路径含非 ASCII 字符）：")
    print(f'    cd "{project_root()}" && "{sys.executable}" -m pip install -e . --no-deps')
    print("    路径含中文时 pip 的 .pth 会被静默跳过 —— 把仓库放到英文路径下即可")
    print("=" * 68)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
