"""让包可以直接 ``python -m copilot`` 运行。"""

from __future__ import annotations

from copilot.cli import main

if __name__ == "__main__":
    raise SystemExit(main())