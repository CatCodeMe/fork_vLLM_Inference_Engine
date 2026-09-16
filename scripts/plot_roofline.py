"""
scripts/plot_roofline.py — 生成 prefill / decode 的 Roofline 示意图。

对应博客第 8 节。横轴是算术强度 I = FLOPs / Bytes，纵轴是可达到的性能。
屋顶 = min(峰值算力, I × 带宽)；分界点 ridge = 峰值算力 / 带宽。
  - decode：I ≈ 1（fp16 下每参数只算 2 FLOPs、却要读 2 字节），落在左侧内存受限区；
  - prefill：I ≈ n（权重复用 n 次），n 一大就进入右侧计算受限区；
  - decode 批处理 B 条序列会把 I 抬到 ≈ B，这正是批处理能提升吞吐的原因。

数值为示意（H100 级：约 1000 TFLOPS fp16、3.35 TB/s），仅用于说明量级。

用法：uv run python scripts/plot_roofline.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _plot_common import save, setup_style

PEAK_TFLOPS = 1000.0     # 峰值算力（示意）
BW_TBPS = 3.35           # 显存带宽（示意）
RIDGE = PEAK_TFLOPS / BW_TBPS   # FLOP/byte


def main() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    I = np.logspace(-1, 4, 400)                 # 算术强度
    perf = np.minimum(PEAK_TFLOPS, I * BW_TBPS)  # 屋顶
    ax.loglog(I, perf, color="black", lw=2, label="roofline = min(peak, I x BW)")
    ax.axvline(RIDGE, ls="--", color="grey", lw=1)
    ax.text(RIDGE * 1.15, 2, f"ridge = {RIDGE:.0f} FLOP/byte",
            rotation=90, va="bottom", fontsize=8, color="grey")

    # 标注两个区域
    ax.axvspan(0.1, RIDGE, color="tab:orange", alpha=0.08)
    ax.axvspan(RIDGE, 1e4, color="tab:blue", alpha=0.08)
    ax.text(3, 700, "memory-bound\n(decode)", fontsize=9, color="tab:orange",
            ha="center")
    ax.text(3000, 150, "compute-bound\n(prefill)", fontsize=9, color="tab:blue",
            ha="center")

    # 关键工作点
    points = [
        (1,    "decode, batch = 1", "tab:red"),
        (32,   "decode, batch = 32", "tab:red"),
        (512,  "prefill, n = 512", "tab:green"),
        (2048, "prefill, n = 2048", "tab:green"),
    ]
    for intensity, label, color in points:
        y = min(PEAK_TFLOPS, intensity * BW_TBPS)
        ax.scatter([intensity], [y], color=color, zorder=5)
        ax.annotate(label, (intensity, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color=color)

    ax.set_xlabel(r"arithmetic intensity  $I = \mathrm{FLOPs} / \mathrm{Bytes}$")
    ax.set_ylabel("achieved performance (TFLOPS)")
    ax.set_title("Roofline: prefill is compute-bound, decode is memory-bound")
    ax.set_xlim(0.1, 1e4)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "roofline_prefill_decode")


if __name__ == "__main__":
    main()
