"""
scripts/plot_rope.py — 生成 RoPE（旋转位置编码）的辅助理解图。

产出（统一走 _plot_common）：docs/figures/rope_freq_decay.svg / .png

图分两栏：
  左：每个“二维平面”的旋转频率 theta_i，对比 base=1e4（原始 Transformer）与
      base=1e6（Qwen2）。base 越大，频率整体越低。
  右：同一个向量旋转 Delta 个位置后的“自点积”（与未旋转时的内积），
      随距离 Delta 的衰减曲线。展示 RoPE 的“长程衰减”倾向，以及 base=1e6
      为什么能覆盖更长的上下文（衰减更慢）。

用法：
    uv run python scripts/plot_rope.py
    uv run python scripts/plot_all.py     # 重生成所有图
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _plot_common import save, setup_style

# ── 配置 ──────────────────────────────────────────────────────────────────────

D_HEAD = 64                    # Qwen2 每个注意力头的维度（14 头 × 64 = 896）
BASES = [1e4, 1e6]             # 10^4 = 原始 Transformer；10^6 = Qwen2
LABELS = {1e4: r"base $=10^4$ (original)", 1e6: r"base $=10^6$ (Qwen2)"}
MAX_DELTA = 512                # 横轴：相对距离范围
N_SAMPLES = 4000               # 随机向量采样数（用于估计期望）


# ── RoPE 核心 ─────────────────────────────────────────────────────────────────

def theta_i(base: float, d_head: int = D_HEAD) -> np.ndarray:
    """每个二维平面的旋转频率 theta_i = base^(-2i/d_head)，i = 0..d_head/2-1。"""
    i = np.arange(d_head // 2)
    return base ** (-2.0 * i / d_head)


def rope_apply(x: np.ndarray, pos: float, base: float) -> np.ndarray:
    """把向量 x 按位置 pos 做 RoPE 旋转（Llama/HF 的 half-split 写法）。

    对每一对维度 (i, i + d/2) 旋转角度 pos * theta_i：
        x_i'       = x_i * cos - x_{i+d/2} * sin
        x_{i+d/2}' = x_{i+d/2} * cos + x_i * sin
    """
    half = x.shape[-1] // 2
    th = theta_i(base, x.shape[-1])          # [half]
    ang = pos * th                            # [half]
    cos = np.concatenate([np.cos(ang), np.cos(ang)])   # [d_head]
    sin = np.concatenate([np.sin(ang), np.sin(ang)])
    x1, x2 = x[:half], x[half:]
    rotate_half = np.concatenate([-x2, x1])   # [-x_{i+d/2}, x_i]
    return x * cos + rotate_half * sin


def self_dot_decay(base: float, max_delta: int = MAX_DELTA, n: int = N_SAMPLES) -> np.ndarray:
    """估计 E[ <RoPE(q, Delta), q> ]，q 为随机单位向量（Delta=0 时为 1）。"""
    rng = np.random.default_rng(0)
    q = rng.normal(size=(n, D_HEAD))
    q /= np.linalg.norm(q, axis=1, keepdims=True)      # 单位向量
    curve = np.empty(max_delta + 1)
    for delta in range(max_delta + 1):
        rotated = np.array([rope_apply(row, delta, base) for row in q])
        curve[delta] = float(np.mean(np.sum(rotated * q, axis=1)))
    return curve


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # 左图：各维频率
    for base in BASES:
        th = theta_i(base)
        ax1.plot(np.arange(len(th)), th, marker="o", ms=3, label=LABELS[base])
    ax1.set_yscale("log")
    ax1.set_xlabel(r"pair index $i$  (0 .. d/2-1,  d=%d)" % D_HEAD)
    ax1.set_ylabel(r"rotation frequency $\theta_i$  (log scale)")
    ax1.set_title("Per-plane RoPE frequencies")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # 右图：自点积衰减
    for base in BASES:
        curve = self_dot_decay(base)
        ax2.plot(np.arange(len(curve)), curve, label=LABELS[base])
    ax2.set_xlabel(r"relative distance $\Delta$")
    ax2.set_ylabel(r"mean self-dot  $\langle \mathrm{RoPE}(q,\Delta),\, q\rangle$")
    ax2.set_title("Long-term decay of RoPE similarity")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("RoPE (Rotary Position Embedding) — frequencies & long-term decay",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "rope_freq_decay")

    # 打印几个频率做对照（便于在文中引用）
    for base in BASES:
        th = theta_i(base)
        print(
            f"base={base:.0e}: theta_0={th[0]:.3e}  "
            f"theta_16={th[16]:.3e}  theta_31={th[31]:.3e}  "
            f"period(theta_31)=2pi/theta={2*np.pi/th[31]:.1f}"
        )


if __name__ == "__main__":
    main()
