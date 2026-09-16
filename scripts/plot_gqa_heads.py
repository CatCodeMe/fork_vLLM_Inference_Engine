"""
scripts/plot_gqa_heads.py — 生成注意力头分组示意图（MHA / GQA / MQA）。

对应博客 4.6 节。以 Qwen2-0.5B 的 14 个 query 头为基准：
  MHA: 14 个 KV 头（1:1）
  GQA:  2 个 KV 头（Qwen2 采用，7 个 query 共享 1 个 KV）
  MQA:  1 个 KV 头（全部共享）
KV 头越少，KV cache 越小、decode 越省带宽。

用法：uv run python scripts/plot_gqa_heads.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _plot_common import save, setup_style

N_Q = 14          # Qwen2-0.5B 的 query 头数
GREY = "#444444"


def draw_panel(ax, n_kv: int, title: str, note: str = "") -> None:
    cmap = plt.get_cmap("tab10")
    colors = [cmap(g % 10) for g in range(n_kv)]

    # 上行：query 头
    for i in range(N_Q):
        g = i * n_kv // N_Q                      # 该 query 头属于哪一组
        ax.add_patch(Rectangle((i, 1.0), 0.8, 0.55, facecolor=colors[g],
                               edgecolor="black", lw=0.6))
    # 下行：KV 头
    group_w = N_Q / n_kv
    for g in range(n_kv):
        x0 = g * group_w
        ax.add_patch(Rectangle((x0, 0.0), group_w - 0.2, 0.55,
                               facecolor=colors[g], edgecolor="black", lw=0.9))
        center = x0 + (group_w - 0.2) / 2
        # 连接线：该组的 query 头 -> KV 头
        for i in range(int(round(x0)), int(round(x0 + group_w))):
            ax.plot([i + 0.4, center], [1.0, 0.55], color=colors[g],
                    lw=0.7, alpha=0.8, zorder=0)

    ax.set_xlim(-3.2, N_Q + 0.5)
    ax.set_ylim(-0.9, 1.95)
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    ax.text(-1.0, 1.27, f"Q heads ({N_Q})", ha="right", va="center",
            fontsize=8, color=GREY)
    ax.text(-1.0, 0.27, f"KV heads ({n_kv})", ha="right", va="center",
            fontsize=8, color=GREY)
    if note:
        ax.text(N_Q / 2, -0.5, note, ha="center", va="center",
                fontsize=8, color=GREY)


def main() -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))
    draw_panel(axes[0], 14, "MHA  (n_kv = 14)", "1 query per KV head")
    draw_panel(axes[1], 2, "GQA  (n_kv = 2)  ← Qwen2", "7 queries share 1 KV head")
    draw_panel(axes[2], 1, "MQA  (n_kv = 1)", "all queries share 1 KV head")
    fig.suptitle("Attention head grouping: KV heads drive KV-cache size", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "gqa_heads")


if __name__ == "__main__":
    main()
