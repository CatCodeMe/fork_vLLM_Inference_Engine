"""
scripts/plot_attention_landscape.py — 注意力优化全景图（按「优化什么」分六条正交轴）。

对应 docs/0009-attention-landscape.md。很多名词容易混，其实它们分属不同维度：
  KV 缓存大小 / 稀疏度 / 渐近复杂度 / IO 与 kernel / 显存管理 / 数值精度

用法：uv run python scripts/plot_attention_landscape.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _plot_common import save, setup_style

CARDS = [
    ("What to cache", "reduce KV-cache size",
     "MQA · GQA · MLA", "#1f77b4"),
    ("How much to attend", "sparsity of positions",
     "Sliding window · NSA · DSA", "#2ca02c"),
    ("Asymptotic cost", "O(n) instead of O(n²)",
     "Linear attention · SSM / Mamba", "#9467bd"),
    ("How to compute", "IO / kernel efficiency",
     "FlashAttention (exact, IO-aware)", "#ff7f0e"),
    ("How to store", "memory management",
     "PagedAttention (vLLM blocks)", "#8c564b"),
    ("Precision", "numerics of the matmuls",
     "Quantized attention (SageAttention)\nKV-cache quantization", "#d62728"),
]


def main() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.axis("off")

    for idx, (title, subtitle, methods, color) in enumerate(CARDS):
        row, col = divmod(idx, 3)          # row 0 = top
        x0, y0 = col + 0.06, (1 - row) + 0.06
        w, h = 0.88, 0.88
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=color + "14", edgecolor=color, lw=1.8))
        cx = x0 + w / 2
        ax.text(cx, y0 + h - 0.16, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)
        ax.text(cx, y0 + h - 0.30, subtitle, ha="center", va="center",
                fontsize=8, color="#666666", style="italic")
        ax.text(cx, y0 + h / 2 - 0.06, methods, ha="center", va="center",
                fontsize=9, color="#222222")

    fig.suptitle("Attention optimization landscape — six orthogonal axes",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "attention_landscape")


if __name__ == "__main__":
    main()
