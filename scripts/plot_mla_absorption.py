"""
scripts/plot_mla_absorption.py — MLA「矩阵吸收」的维度变化图。

对应 docs/0009-attention-landscape.md。

MLA 推理时为了不物化每头的 K，会把上投影矩阵 W^UK 折进 W^Q：
    q·k_C = (W^Q h)·(W^UK c) = (W^UK^T W^Q h)·c
于是「先 up-project 再点积」变成「对 q 换个投影后直接和缓存的 c 点积」。

左：朴素 MLA —— 每步要把 c up-project 成 k（n_h·d_h 维），代价大
右：吸收后   —— 只缓存 c（d_c 维），q 换成 q'（n_h·d_c 维）后直接与 c 点积

用法：uv run python scripts/plot_mla_absorption.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _plot_common import save, setup_style

CACHE_EC = "#d62728"     # 缓存相关高亮
NORMAL_EC = "#2f6fb5"


def node(ax, x, y, text, w=0.26, h=0.13, ec=NORMAL_EC, fc="#eef4fb", fs=8):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fc, edgecolor=ec, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#222222")


def arrow(ax, p1, p2, label="", color="#444444", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=11, color=color, lw=1.3,
        connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.03, label, ha="center", va="bottom",
                fontsize=7.5, color=color)


def panel_naive(ax):
    ax.set_title("Naive MLA: materialize K each step", fontsize=11)
    node(ax, 0.12, 0.55, "h\n[d]", ec="#555555", fc="#f2f2f2")
    node(ax, 0.55, 0.88, "q\n[$n_h d_h$]", ec=NORMAL_EC)
    node(ax, 0.55, 0.22, "c\n[$d_c$]\ncached", ec=CACHE_EC, fc="#fdeeee")
    node(ax, 0.85, 0.55, "k\n[$n_h d_h$]", ec=NORMAL_EC)
    node(ax, 1.0, 0.55, "", w=0.0, h=0.0)  # no-op, keeps xlim tidy
    arrow(ax, (0.12, 0.62), (0.55, 0.84), "W^Q")
    arrow(ax, (0.12, 0.48), (0.55, 0.28), "W^DKV")
    arrow(ax, (0.55, 0.29), (0.85, 0.49), "W^UK")
    ax.text(0.5, 0.02, "score = q · k   (k has $n_h d_h$ dims — expensive)",
            ha="center", fontsize=8, color="#666666")


def panel_absorbed(ax):
    ax.set_title("Absorbed: fold W^UK into W^Q", fontsize=11)
    node(ax, 0.12, 0.55, "h\n[d]", ec="#555555", fc="#f2f2f2")
    node(ax, 0.55, 0.85, "q' = $W^{UK\\top} W^Q h$\n[$n_h d_c$]", ec=NORMAL_EC)
    node(ax, 0.55, 0.25, "c\n[$d_c$]\ncached", ec=CACHE_EC, fc="#fdeeee")
    arrow(ax, (0.12, 0.62), (0.55, 0.80), "absorb")
    arrow(ax, (0.12, 0.48), (0.55, 0.31), "W^DKV")
    ax.text(0.5, 0.02, "score = q' · c   (no K materialized; cache stays $d_c$)",
            ha="center", fontsize=8, color="#666666")


def main() -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    for ax in axes:
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.axis("off")
    panel_naive(axes[0])
    panel_absorbed(axes[1])
    fig.suptitle("MLA matrix absorption — dimension change", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "mla_absorption")


if __name__ == "__main__":
    main()
