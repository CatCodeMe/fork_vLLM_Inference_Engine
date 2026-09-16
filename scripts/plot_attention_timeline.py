"""
scripts/plot_attention_timeline.py — 注意力机制演进时间线（两条优化路线）。

对应 docs/0008-attention-variants.md。

  上轨：减少「缓存什么」——压缩 KV cache 体积（MHA → MQA → GQA → MLA）
  下轨：减少「看多少」——稀疏化被注意的位置（Full → Sliding Window → NSA → DSA）

用法：uv run python scripts/plot_attention_timeline.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _plot_common import save, setup_style

# (year, label, note, color)
TRACK_KV = [
    (2017, "MHA", "every head has its own K/V\n(original Transformer)", "#d62728"),
    (2019, "MQA", "all heads share 1 K/V", "#bcbd22"),
    (2023, "GQA", "groups share K/V\n(Llama 2 / Qwen2)", "#ff7f0e"),
    (2024, "MLA", "low-rank latent +\ndecoupled RoPE (DeepSeek)", "#1f77b4"),
]
TRACK_SPARSE = [
    (2017, "Full attention", "attend to all past tokens", "#7f7f7f"),
    (2023, "Sliding window", "local window only\n(Mistral SWA)", "#9467bd"),
    (2025, "NSA", "natively trainable\nsparse attention (DeepSeek)", "#2ca02c"),
    (2025, "DSA", "top-k indexer on top of MLA\n(DeepSeek-V3.2)", "#17becf"),
]

Y_KV, Y_SP = 1.0, 0.0


def draw_track(ax, y: float, title: str, items):
    ax.hlines(y, 2016.4, 2026.0, color="#bbbbbb", lw=2, zorder=0)
    ax.text(2016.3, y + 0.09, title, ha="left", va="bottom",
            fontsize=9.5, fontweight="bold", color="#333333")
    for year, label, note, color in items:
        ax.plot([year], [y], marker="o", ms=9, color=color,
                markeredgecolor="black", markeredgewidth=0.6, zorder=2)
        ax.annotate(f"{label}\n{year}", (year, y),
                    textcoords="offset points", xytext=(0, 13),
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color=color)
        ax.annotate(note, (year, y), textcoords="offset points", xytext=(0, -15),
                    ha="center", va="top", fontsize=6.8, color="#555555")


def main() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(12.5, 5.2))

    draw_track(ax, Y_KV, "Track A · what to cache  (reduce KV cache size)", TRACK_KV)
    draw_track(ax, Y_SP, "Track B · how much to attend  (sparsify positions)", TRACK_SPARSE)

    ax.set_xlim(2016.2, 2026.2)
    ax.set_ylim(-0.55, 1.75)
    ax.set_xticks(range(2017, 2026))
    ax.set_xlabel("year")
    ax.set_yticks([])
    ax.set_title("Evolution of attention: two orthogonal optimization tracks", fontsize=12)
    save(fig, "attention_timeline")


if __name__ == "__main__":
    main()
