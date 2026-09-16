"""
scripts/plot_all.py — 重新生成 docs/figures/ 下的所有图。

它会依次运行同目录下的 `plot_*.py`（除自己外），每个脚本独立进程执行，
避免 matplotlib 全局状态互相影响。

用法：
    uv run python scripts/plot_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    scripts = sorted(
        p for p in HERE.glob("plot_*.py") if p.name != Path(__file__).name
    )
    if not scripts:
        print("no plot_*.py scripts found")
        return
    for script in scripts:
        print(f"=== {script.name} ===")
        subprocess.run([sys.executable, str(script)], check=True, cwd=HERE)
    print(f"\ndone: regenerated {len(scripts)} figure script(s)")


if __name__ == "__main__":
    main()
