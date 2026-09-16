"""
engine/paged_kv_cache.py — Phase 7: Paged KV Cache Manager.

Owns the actual torch tensor pool for all KV cache memory.  Uses
BlockAllocator (Phase 6) for block-level bookkeeping and provides
read/write access keyed by (seq_id, layer_idx, token_position).

No custom CUDA kernels — standard torch tensor indexing throughout.
No async.  Thread-safe via threading.Lock for cursor mutations.
"""

# [LEARN] 这里是 Phase 7 的"仓库"：一次性申请一整块大 tensor，之后靠索引读写，
#         推理过程中不再新分配显存。BlockAllocator 告诉你"第 k 个 token 在第几号
#         物理块"，本模块负责把它映射到 tensor 的具体下标。
# [WHY] 和 BlockAllocator 分开：allocator 是纯 Python 记账，可单独测试；
#       本模块才依赖 torch。

from __future__ import annotations

import threading
from typing import Optional

import torch

from inference_engine.engine.block_allocator import BlockAllocator
from inference_engine.engine.kv_cache_config import KVCacheConfig


class PagedKVCacheManager:
    """Pre-allocated paged KV cache pool.

    Parameters
    ----------
    kv_cache_config:
        Architectural metadata (layers, heads, head_dim, dtype, device).
    block_allocator:
        Phase 6 block allocator — passed in, not owned here.
    config:
        Engine config; reads ``kv_block_size``, ``kv_num_blocks``, ``device``.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        block_allocator: BlockAllocator,
        config,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.block_allocator = block_allocator
        self.config = config

        self.num_layers: int = kv_cache_config.num_layers
        self.num_kv_heads: int = kv_cache_config.num_kv_heads
        self.head_dim: int = kv_cache_config.head_dim
        self.block_size: int = config.kv_block_size
        self.num_blocks: int = config.kv_num_blocks
        self.device: str = config.device
        self.dtype: torch.dtype = kv_cache_config.dtype

        # ── Single large pre-allocated pool ───────────────────────────────────
        # Shape: [num_blocks, block_size, num_layers, num_kv_heads, head_dim]
        # Allocated ONCE here — never resized during inference.
        # [LEARN] 维度顺序很有讲究：把 block_size 放在第 2 维，则"取整块"是
        #         key_pool[block_id]（连续内存），"取块内某 token"是 [block_id, slot]。
        # [GOTCHA] 这里用的是 GQA：num_kv_heads (2) 远小于 attention heads，
        #         所以实际 KV 内存比按 head 数算的要小很多。
        self.key_pool: torch.Tensor = torch.zeros(
            [self.num_blocks, self.block_size, self.num_layers,
             self.num_kv_heads, self.head_dim],
            dtype=self.dtype,
            device=self.device,
        )
        self.value_pool: torch.Tensor = torch.zeros_like(self.key_pool)

        # seq_id → next global token write cursor (index across all blocks)
        self._seq_write_cursors: dict[str, int] = {}
        self._lock = threading.Lock()

    # ── Write ─────────────────────────────────────────────────────────────────

    def write_kv(
        self,
        seq_id: str,
        layer_idx: int,
        token_position: int,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Write a single token's KV pair into the pool.

        Parameters
        ----------
        seq_id:
            Owning sequence.
        layer_idx:
            Transformer layer index (0-based).
        token_position:
            Global token position within the sequence (0-based).
        key_tensor:
            Shape: [num_kv_heads, head_dim]
        value_tensor:
            Shape: [num_kv_heads, head_dim]
        """
        # [LEARN] 逻辑位置 → 物理位置的核心映射（类比页表）：
        #   block_idx_in_seq = token_position // block_size  → 这是该序列的第几个块
        #   slot_within_block = token_position %  block_size  → 块内第几个槽
        #   再用 block_allocator 把"第几个逻辑块"翻译成"哪个物理块号"。
        block_idx_in_seq = token_position // self.block_size
        slot_within_block = token_position % self.block_size

        block_ids = self.block_allocator.get_blocks(seq_id)
        physical_block_id = block_ids[block_idx_in_seq]

        # Direct tensor index assignment — no torch.cat / stack
        self.key_pool[physical_block_id, slot_within_block, layer_idx] = key_tensor
        self.value_pool[physical_block_id, slot_within_block, layer_idx] = value_tensor

        # Track write cursor
        with self._lock:
            cursor = self._seq_write_cursors.get(seq_id, 0)
            self._seq_write_cursors[seq_id] = max(cursor, token_position + 1)

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_kv_sequence(
        self,
        seq_id: str,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble all filled KV tensors for a sequence at one layer.

        Returns
        -------
        (keys, values)
            keys:   [total_filled_tokens, num_kv_heads, head_dim]
            values: [total_filled_tokens, num_kv_heads, head_dim]

        Only filled slots (``tokens_used`` on each block) are returned.
        """
        # [LEARN] 从各块取出"已填满的部分"，按分配顺序拼接成一个连续 KV。
        #         注意每块只取 :tokens_used，最后一个没填满的块不会把空洞也拼进去。
        block_ids = self.block_allocator.get_blocks(seq_id)

        key_slices: list[torch.Tensor] = []
        val_slices: list[torch.Tensor] = []

        for bid in block_ids:
            tokens_used = self.block_allocator.get_block(bid).tokens_used
            if tokens_used <= 0:
                continue
            # key_pool[bid, :tokens_used, layer_idx] → [tokens_used, num_kv_heads, head_dim]
            key_slices.append(self.key_pool[bid, :tokens_used, layer_idx])
            val_slices.append(self.value_pool[bid, :tokens_used, layer_idx])

        if not key_slices:
            # Return empty tensors with correct trailing dims
            empty = torch.zeros(
                0, self.num_kv_heads, self.head_dim,
                dtype=self.dtype, device=self.device,
            )
            return empty, empty.clone()

        keys = torch.cat(key_slices, dim=0)
        values = torch.cat(val_slices, dim=0)
        return keys, values

    def read_kv_block(
        self,
        block_id: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the raw key/value tensors for a physical block at one layer.

        Returns
        -------
        (keys, values)
            keys:   [block_size, num_kv_heads, head_dim]
            values: [block_size, num_kv_heads, head_dim]

        This is the low-level accessor — empty slots are NOT filtered out.
        """
        keys = self.key_pool[block_id, :, layer_idx]    # [block_size, H, D]
        values = self.value_pool[block_id, :, layer_idx]
        return keys, values

    # ── Maintenance ───────────────────────────────────────────────────────────

    def clear_sequence(self, seq_id: str) -> None:
        """Zero all pool slots occupied by *seq_id*.

        Must be called BEFORE ``block_allocator.free(seq_id)`` so we still
        know which block_ids belong to the sequence.

        Does NOT call block_allocator.free() — the caller is responsible.
        """
        # [GOTCHA] 调用顺序：先 clear_sequence() 再 block_allocator.free()。
        #         因为本方法要通过 get_blocks(seq_id) 查块号；free 之后就查不到了。
        #         完整顺序见 scheduler._resolve_sequence_future()。
        block_ids = self.block_allocator.get_blocks(seq_id)
        for bid in block_ids:
            # In-place zero via slice assignment
            self.key_pool[bid].zero_()
            self.value_pool[bid].zero_()

        with self._lock:
            self._seq_write_cursors.pop(seq_id, None)

    def copy_blocks(
        self,
        src_block_id: int,
        dst_block_id: int,
        layer_idx: Optional[int] = None,
    ) -> None:
        """Copy KV tensors from *src_block_id* to *dst_block_id* in-place.

        Parameters
        ----------
        layer_idx:
            If None, copy all layers.
            If specified, copy only that single layer.

        Uses ``Tensor.copy_()`` for in-place copy per the Phase 7 spec.
        Wired here for prefix caching / beam search — not invoked in Phase 7.
        """
        # [LEARN] 块拷贝是 prefix caching（多请求共享相同前缀）和 beam search
        #         的基础。当前版本还没启用，属于"接口就绪、功能未接"。
        if layer_idx is None:
            # Copy all layers: full [block_size, num_layers, num_kv_heads, head_dim]
            self.key_pool[dst_block_id].copy_(self.key_pool[src_block_id])
            self.value_pool[dst_block_id].copy_(self.value_pool[src_block_id])
        else:
            # Copy single layer: [block_size, num_kv_heads, head_dim]
            self.key_pool[dst_block_id, :, layer_idx].copy_(
                self.key_pool[src_block_id, :, layer_idx]
            )
            self.value_pool[dst_block_id, :, layer_idx].copy_(
                self.value_pool[src_block_id, :, layer_idx]
            )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a snapshot of the pool state.

        Returns
        -------
        dict with keys:
            pool_shape, pool_size_mb, num_blocks, block_size, num_layers,
            num_kv_heads, head_dim, dtype, device, active_sequences,
            block_allocator (nested dict from BlockAllocator.stats())
        """
        dtype_bytes = {
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float32: 4,
            torch.float64: 8,
        }.get(self.dtype, 2)

        total_elements = self.key_pool.nelement() + self.value_pool.nelement()
        pool_size_mb = total_elements * dtype_bytes / (1024 * 1024)

        with self._lock:
            active_seqs = len(self._seq_write_cursors)

        return {
            "pool_shape": list(self.key_pool.shape),
            "pool_size_mb": pool_size_mb,
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "active_sequences": active_seqs,
            "block_allocator": self.block_allocator.stats(),
        }
