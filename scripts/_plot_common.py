"""
scripts/_plot_common.py — 图表生成的共享约定（供 scripts/plot_*.py 复用）。

约定
----
1. 每个脚本命名 `scripts/plot_<name>.py`，调用 `save(fig, "<name>")`；
2. 图统一输出到 `docs/figures/<name>.svg` 和 `<name>.png`；
3. 图内文字统一用英文 + DejaVu Sans（字形完整、无系统字体依赖），
   中文解释写在博客正文的图注里；
4. 在 markdown 里用相对路径引用：
       ![说明](figures/<name>.svg)
   （markdown 文件在 docs/ 下，故路径是 figures/...）

重生成所有图：
    uv run python scripts/plot_all.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "docs" / "figures"


def setup_style() -> None:
    """统一的绘图风格（幂等，可重复调用）。"""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"


def save(fig, name: str) -> tuple[Path, Path]:
    """把图保存为 docs/figures/<name>.svg 和 .png，返回两个路径。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = FIG_DIR / f"{name}.svg"
    png_path = FIG_DIR / f"{name}.png"
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=150)
    print(f"saved: {svg_path}")
    print(f"saved: {png_path}")
    return svg_path, png_path
