"""Analytical KV-cache sizing from HuggingFace model metadata."""

# [LEARN] 这个模块不存数据，只"算账"：根据模型结构算出每个 token 的 KV 占用。
#         公式：bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
#         （2 = key 和 value 各一份；注意用 kv_heads 而非 attention heads，即 GQA）。

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class KVCacheConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: str
    bytes_per_token: int = field(init=False)
    bytes_per_token_mb: float = field(init=False)

    def __post_init__(self) -> None:
        dtype_bytes = {
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float32: 4,
        }.get(self.dtype, 2)
        self.bytes_per_token = (
            2 * self.num_layers * self.num_kv_heads * self.head_dim * dtype_bytes
        )
        self.bytes_per_token_mb = self.bytes_per_token / (1024 * 1024)


def compute_kv_cache_config(model, config) -> KVCacheConfig:
    """Build cache sizing metadata from a decoder-only HuggingFace model."""
    # [LEARN] num_key_value_heads 是 GQA/MQA 的关键：没有该字段就回退到
    #         num_attention_heads（即传统 MHA）。Qwen2-0.5B 是 2 个 KV head，
    #         所以 KV 内存只有按 head 数估算的 1/7。
    model_config = model.config
    num_attention_heads = model_config.num_attention_heads
    num_kv_heads = getattr(
        model_config, "num_key_value_heads", num_attention_heads
    )
    return KVCacheConfig(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=model_config.hidden_size // num_attention_heads,
        dtype=next(model.parameters()).dtype,  # 模型参数的 dtype（next() 含义见 sequential.prefill）
        device=config.device,
    )


def estimate_max_sequences(
    kv_cache_config: KVCacheConfig,
    available_memory_mb: float,
    avg_sequence_length: int = 512,
) -> int:
    """Estimate how many average-length sequence caches fit in memory."""
    # [GOTCHA] 纯估算函数，当前生产代码/调度器并未使用，只有单元测试在调它；
    #         实际并发上限由 BlockAllocator（块池）决定。
    memory_per_sequence_mb = (
        kv_cache_config.bytes_per_token_mb * avg_sequence_length
    )
    if memory_per_sequence_mb <= 0:
        return 1
    return max(1, math.floor(available_memory_mb / memory_per_sequence_mb))


def format_kv_cache_report(kv_cache_config: KVCacheConfig) -> str:
    """Return a human-readable summary of KV-cache sizing metadata."""
    return "\n".join(
        [
            "KV Cache Configuration",
            f"  Layers: {kv_cache_config.num_layers}",
            f"  KV heads: {kv_cache_config.num_kv_heads}",
            f"  Head dimension: {kv_cache_config.head_dim}",
            f"  Dtype: {kv_cache_config.dtype}",
            f"  Bytes per token: {kv_cache_config.bytes_per_token / 1024:.2f} KB",
            f"  Memory per token: {kv_cache_config.bytes_per_token_mb:.6f} MB",
        ]
    )
