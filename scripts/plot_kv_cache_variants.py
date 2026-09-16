"""
scripts/plot_kv_cache_variants.py — MHA / GQA / MQA / MLA 的 KV cache 体积对比。

对应博客 4.6 节。每个数字是「每 token 每层」的 KV cache 字节数（fp16，每元素 2 字节）：
  MHA / GQA / MQA : 2 * n_kv * d_h * 2        （K、V 各一份）
  MLA             : (d_c + d_h_rope) * 2      （只缓存低秩隐向量 + 解耦 RoPE 键）

对比两组：
  Qwen2-0.5B 风格（d_h=64, n_h=14，看 head 数的影响：MHA/GQA/MQA）
  DeepSeek-V2     （d_h=128, n_h=128，看 MLA 相对 MHA 的压缩：约 57x）

用法：uv run python scripts/plot_kv_cache_variants.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _plot_common import save, setup_style

BYTES_PER_ELEM = 2  # fp16


def kv_bytes_mha_like(n_kv: int, d_h: int) -> int:
    return 2 * n_kv * d_h * BYTES_PER_ELEM


def kv_bytes_mla(d_c: int, d_h_rope: int) -> int:
    return (d_c + d_h_rope) * BYTES_PER_ELEM


def main() -> None:
    setup_style()

    entries = [
        ("MHA\n(Qwen2 style)\nn_kv=14", kv_bytes_mha_like(14, 64), "#d62728"),
        ("GQA\n(Qwen2)\nn_kv=2", kv_bytes_mha_like(2, 64), "#ff7f0e"),
        ("MQA\nn_kv=1", kv_bytes_mha_like(1, 64), "#bcbd22"),
        ("MHA\n(DeepSeek-V2)\nn_kv=128", kv_bytes_mha_like(128, 128), "#d62728"),
        ("MLA\n(DeepSeek-V2)\nd_c=512 + 64", kv_bytes_mla(512, 64), "#1f77b4"),
    ]

    labels = [e[0] for e in entries]
    values_kb = [e[1] / 1024 for e in entries]
    colors = [e[2] for e in entries]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values_kb, color=colors, edgecolor="black", lw=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("KV cache per token per layer (KB, fp16, log scale)")
    ax.set_title("KV-cache size across attention variants")
    ax.grid(True, axis="y", which="both", alpha=0.3)

    for bar, kb in zip(bars, values_kb):
        ax.text(bar.get_x() + bar.get_width() / 2, kb * 1.15,
                f"{kb:.2f} KB", ha="center", va="bottom", fontsize=8)

    # MLA 相对同模型 MHA 的压缩比
    ratio = kv_bytes_mha_like(128, 128) / kv_bytes_mla(512, 64)
    ax.annotate(f"MLA vs MHA:\n~{ratio:.0f}x smaller",
                xy=(4, values_kb[4]), xytext=(2.9, values_kb[3]),
                fontsize=9, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2))

    ax.set_ylim(0.1, max(values_kb) * 4)
    save(fig, "kv_cache_variants")


if __name__ == "__main__":
    main()
