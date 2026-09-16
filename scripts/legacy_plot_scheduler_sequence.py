# ⚠️ LEGACY：本脚本已被 docs/figures/scheduler-*.drawio.svg（drawio 手绘）取代，
#    不再由 plot_all.py 调用（文件名前缀 legacy_ 使其不被 glob 到）。
#    保留仅供参考；画图请直接编辑 .drawio.svg（见 docs/0004-figures.md）。
"""
scripts/plot_scheduler_sequence.py — Phase 2 调度器「一次请求」的时序图。

对应 docs/0006-scheduler-lifecycle.md。画出从 POST /generate 到 future 解析、
返回 JSON 的完整调用链，以及后台 run_loop 的 Step 0~3。
（图内文字用英文，遵循 docs/0004-figures.md 约定。）

用法：uv run python scripts/plot_scheduler_sequence.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _plot_common import save, setup_style

LANES = [
    "Client /\nHTTP",
    "endpoint_\ngenerate()",
    "Request\nQueue",
    "_schedule()\n(run_loop task)",
    "engine\nblocks · pool · model",
    "future",
]

# (y, x_from, x_to, label, color, dashed)
EVENTS = [
    (10.6, 0, 1, "POST /generate", "#333333", False),
    (9.9, 1, 2, "add_request(): tokenize -> Sequence.create", "#1f77b4", False),
    (9.3, 2, 5, "enqueue() -> create future", "#1f77b4", False),
    (8.8, 2, 1, "return (seq, future)", "#1f77b4", True),
    (7.6, 3, 2, "Step 1 · dequeue()", "#2ca02c", False),
    (7.0, 2, 3, "sequence", "#2ca02c", True),
    (6.3, 3, 4, "Step 2 · chunked prefill (forward + write paged pool)", "#ff7f0e", False),
    (5.6, 3, 4, "Step 3 · decode 1 token (forward -> argmax)", "#ff7f0e", False),
    (5.0, 4, 3, "next_token_id", "#ff7f0e", True),
    (4.3, 3, 4, "evict finished: clear pool -> free blocks", "#8c564b", False),
    (3.6, 3, 5, "_resolve_sequence_future(): set_result(seq)", "#d62728", False),
    (2.9, 5, 1, "future resolves", "#d62728", True),
    (2.2, 1, 0, "HTTP 200 JSON result", "#333333", True),
]

OOM = (5.85, 3, 4, "OutOfBlocksError -> _try_swap_out_victim() -> retry", "#d62728")


def lane_headers(ax):
    for x, name in enumerate(LANES):
        ax.add_patch(FancyBboxPatch(
            (x - 0.42, 11.25), 0.84, 0.6,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor="#eef2f7", edgecolor="#4a6fa5", lw=1.0))
        ax.text(x, 11.55, name, ha="center", va="center", fontsize=8)
        ax.plot([x, x], [0.2, 11.25], ls=(0, (4, 3)), color="#9aa7b5", lw=0.9)


def arrow(ax, x1, x2, y, label, color, dashed):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                linestyle="--" if dashed else "-",
                                shrinkA=0, shrinkB=0))
    ax.text((x1 + x2) / 2, y + 0.1, label, ha="center", va="bottom",
            fontsize=7.3, color=color)


def main() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(12.5, 7))
    lane_headers(ax)

    # endpoint 挂起区间
    ax.add_patch(FancyBboxPatch(
        (0.62, 2.4), 0.76, 6.4,
        boxstyle="square,pad=0", facecolor="#fff3cd", alpha=0.5, edgecolor="none"))
    ax.text(1.0, 5.6, "await future\n(handler suspended,\nno CPU held)", ha="center",
            va="center", fontsize=7.5, color="#8a6d3b")

    for y, x1, x2, label, color, dashed in EVENTS:
        arrow(ax, x1, x2, y, label, color, dashed)

    # OOM 旁路
    y, x1, x2, label, _ = OOM
    arrow(ax, x1, x2, y, label, "#d62728", True)

    ax.text(3.0, 1.4, "run_loop loops back to Step 0 (continuous batching)",
            ha="center", fontsize=8, color="#555555")
    ax.annotate("", xy=(3.0, 6.9), xytext=(3.0, 1.2),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2,
                                linestyle=":", connectionstyle="arc3,rad=-0.35"))
    ax.plot([4.0], [8.2], marker="o", color="#2ca02c", ms=5)
    ax.text(4.15, 8.2, "Step 0 · swap-in runs first (if any swapped seq)",
            fontsize=7.3, color="#2ca02c", va="center")

    ax.set_xlim(-0.6, len(LANES) - 0.4)
    ax.set_ylim(0.2, 12.1)
    ax.axis("off")
    ax.set_title("Phase 2 scheduler — request lifecycle (sequence diagram)", fontsize=12)
    save(fig, "scheduler_sequence")


if __name__ == "__main__":
    main()
