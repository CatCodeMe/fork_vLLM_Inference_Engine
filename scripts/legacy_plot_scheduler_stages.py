# ⚠️ LEGACY：本脚本已被 docs/figures/scheduler-*.drawio.svg（drawio 手绘）取代，
#    不再由 plot_all.py 调用（文件名前缀 legacy_ 使其不被 glob 到）。
#    保留仅供参考；画图请直接编辑 .drawio.svg（见 docs/0004-figures.md）。
"""
scripts/plot_scheduler_stages.py — Phase 2 调度器 _schedule() 的阶段流程图。

对应 docs/0006-scheduler-lifecycle.md。展示 run_loop 的循环、Step 0~3、
淘汰收尾，以及内存不足时的 swap-out 旁路。
（图内文字用英文，遵循 docs/0004-figures.md 约定。）

用法：uv run python scripts/plot_scheduler_stages.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _plot_common import save, setup_style

CX = 0.0            # 主列中心
BW, BH = 7.0, 1.0   # 主列盒子宽高


def box(ax, x, y, title, body="", color="#2f6fb5", face="#eef4fb", w=BW, h=BH):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        facecolor=face, edgecolor=color, lw=1.4))
    if body:
        ax.text(x, y + h * 0.22, title, ha="center", va="center",
                fontsize=8.3, fontweight="bold", color=color)
        ax.text(x, y - h * 0.22, body, ha="center", va="center",
                fontsize=7.2, color="#333333")
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=8.3, fontweight="bold", color=color)


def v_arrow(ax, x, y1, y2, color="#444444", dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x, y1), (x, y2), arrowstyle="-|>", mutation_scale=12,
        color=color, lw=1.4, linestyle="--" if dashed else "-"))


def main() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 7.8))

    ys = {
        "loop": 10.4, "sched": 9.0, "s0": 7.6, "s1": 6.2,
        "s2": 4.8, "s3": 3.4, "evict": 2.0,
    }

    box(ax, CX, ys["loop"], "run_loop()  ·  background asyncio.Task",
        "idle sleep ~1ms when nothing to do", color="#555555", face="#f2f2f2")
    box(ax, CX, ys["sched"], "_schedule()  —  one iteration",
        "runs Step 0 -> 3 sequentially", color="#111111", face="#f7f7f7")
    box(ax, CX, ys["s0"], "Step 0 · swap-in",
        "restore CPU-swapped sequences when device has room", color="#2ca02c")
    box(ax, CX, ys["s1"], "Step 1 · admit  (bookkeeping only, no forward)",
        "dequeue -> running, state = chunked_prefilling", color="#1f77b4")
    box(ax, CX, ys["s2"], "Step 2 · chunked prefill  (one chunk per seq)",
        "allocate blocks -> forward -> write paged pool; capped by prefill_budget_tokens",
        color="#ff7f0e")
    box(ax, CX, ys["s3"], "Step 3 · decode  (1 token per decoding seq)",
        "write_token -> forward -> argmax -> write paged pool", color="#ff7f0e")
    box(ax, CX, ys["evict"], "evict finished",
        "clear paged pool -> free blocks -> resolve future", color="#8c564b")

    for a, b in [("loop", "sched"), ("sched", "s0"), ("s0", "s1"),
                 ("s1", "s2"), ("s2", "s3"), ("s3", "evict")]:
        v_arrow(ax, CX, ys[a] - BH / 2 - 0.06, ys[b] + BH / 2 + 0.06)

    # 循环回到 _schedule
    rx = CX + BW / 2
    ax.add_patch(FancyArrowPatch((rx, ys["evict"]), (rx + 1.4, ys["evict"]),
                                 arrowstyle="-", color="#555555", lw=1.2))
    v_arrow(ax, rx + 1.4, ys["evict"], ys["sched"], color="#555555", dashed=True)
    ax.add_patch(FancyArrowPatch((rx + 1.4, ys["sched"]), (rx, ys["sched"]),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color="#555555", lw=1.2))
    ax.text(rx + 1.55, (ys["evict"] + ys["sched"]) / 2,
            "next iteration\n(continuous batching)", ha="left", va="center",
            fontsize=7.5, color="#555555")

    # OOM 旁路
    hx, hy = CX + 6.2, (ys["s2"] + ys["s3"]) / 2
    box(ax, hx, hy, "_try_swap_out_victim()",
        "on OutOfBlocksError: swap the largest\n decoding seq to CPU, then retry",
        color="#d62728", face="#fdeeee", w=5.0, h=1.3)
    ax.add_patch(FancyArrowPatch(
        (CX + BW / 2, ys["s2"] - 0.1), (hx - 2.5, hy + 0.35),
        arrowstyle="-|>", mutation_scale=12, color="#d62728", lw=1.2,
        linestyle="--", connectionstyle="arc3,rad=0.2"))
    ax.add_patch(FancyArrowPatch(
        (hx - 2.5, hy - 0.35), (CX + BW / 2, ys["s3"] + 0.1),
        arrowstyle="-|>", mutation_scale=12, color="#d62728", lw=1.2,
        linestyle="--", connectionstyle="arc3,rad=0.2"))
    ax.text(CX + BW / 2 + 0.12, ys["s2"] - 0.42, "OOM", fontsize=7, color="#d62728")

    ax.set_xlim(CX - BW / 2 - 0.6, hx + 2.7)
    ax.set_ylim(ys["evict"] - 0.9, ys["loop"] + 0.9)
    ax.axis("off")
    ax.set_title("Phase 2 scheduler — one _schedule() iteration", fontsize=12)
    save(fig, "scheduler_stages")


if __name__ == "__main__":
    main()
