"""
engine/block_allocator.py — Phase 6: Block Allocator.

Manages a fixed pool of logical KV-cache memory blocks.  Each block holds a
fixed number of token slots (block_size).  Sequences are assigned one or more
blocks as they grow.  Blocks can be freed, reused, and evicted under memory
pressure.

No torch dependency.  No async.  Thread-safe via threading.Lock.
"""

# [LEARN] 类比操作系统虚拟内存：序列看到的是一串连续的"逻辑块"，
#         底层物理块可以任选。这样就不用为每个序列预留 max_seq_len 的连续空间，
#         消除外部碎片。block_size=16 意味着每 16 个 token 占一个块。
# [WHY] 本模块只记"账"（谁占了哪些块、每块用了几个槽），不搬数据；
#       真正的 KV tensor 读写由 paged_kv_cache.py 负责。职责分离便于测试。

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Exceptions ────────────────────────────────────────────────────────────────

class BlockAllocatorError(Exception):
    """Base exception for all BlockAllocator errors."""


class OutOfBlocksError(BlockAllocatorError):
    """Raised when a block allocation request cannot be satisfied."""

    # [TRACE] 这个异常是 Phase 9 抢占的"扳机"：scheduler 捕到它后不杀请求，
    #         而是调 _try_swap_out_victim() 换个序列出去，再重试分配。
    def __init__(self, requested: int, available: int) -> None:
        super().__init__(
            f"Cannot allocate {requested} blocks — only {available} free"
        )
        self.requested = requested
        self.available = available


class BlockNotFoundError(BlockAllocatorError):
    """Raised when a block_id is not present in the pool."""

    def __init__(self, block_id: int) -> None:
        super().__init__(f"Block {block_id} not found in allocator")
        self.block_id = block_id


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class Block:
    """A single logical KV-cache block."""

    block_id: int
    block_size: int                           # max tokens this block can hold
    tokens_used: int = 0                      # how many token slots are filled
    # [LEARN] ref_count + is_dirty 是为 prefix caching / beam search 留的扩展位：
    #         多个序列可共享同一块（ref_count>1），is_dirty 表示块已被写过。
    #         当前版本每条序列独立占块，ref_count 恒为 0 或 1。
    ref_count: int = 0                        # sequences referencing this block
    seq_id: Optional[str] = None             # owning sequence (None = free)
    is_dirty: bool = False                    # True once written to
    allocated_at: float = field(default_factory=time.perf_counter)

    def is_free(self) -> bool:
        """Return True when the block belongs to no sequence."""
        return self.ref_count == 0 and self.seq_id is None

    def is_full(self) -> bool:
        """Return True when all token slots are occupied."""
        return self.tokens_used >= self.block_size

    def remaining_slots(self) -> int:
        """Number of free token slots in this block."""
        return self.block_size - self.tokens_used


# ── BlockAllocator ────────────────────────────────────────────────────────────

class BlockAllocator:
    """Thread-safe fixed-size block pool manager.

    Parameters
    ----------
    num_blocks:
        Total number of blocks in the pool (e.g. 256).
    block_size:
        Number of token slots per block (e.g. 16).
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero")
        if block_size <= 0:
            raise ValueError("block_size must be greater than zero")

        self._num_blocks = num_blocks
        self._block_size = block_size

        # block_id → Block
        self._blocks: dict[int, Block] = {
            i: Block(block_id=i, block_size=block_size)
            for i in range(num_blocks)
        }
        # Free pool — ordered so pop() takes the last element (LIFO within
        # the pool, but callers see FIFO because we reverse-append on free).
        # [GOTCHA] allocate() 实际用 pop(0) 从头部取，所以是 FIFO；
        #         free() 用 append 放到尾部。上面的注释描述的是早期写法，
        #         以 allocate/free 的实际代码为准。
        self._free_block_ids: list[int] = list(range(num_blocks))

        # seq_id → [block_id, ...] in allocation order
        self._seq_to_blocks: dict[str, list[int]] = {}

        # Cumulative counters
        self._total_allocated: int = 0
        self._total_freed: int = 0
        self._total_evictions: int = 0

        self._lock = threading.Lock()

    # ── Core allocation ───────────────────────────────────────────────────────

    def allocate(self, seq_id: str, num_blocks: int = 1) -> list[int]:
        """Allocate *num_blocks* free blocks to *seq_id*.

        Raises
        ------
        OutOfBlocksError
            If the pool does not have *num_blocks* free blocks.
        """
        # [LEARN] 原子分配：要么一次拿到全部 num_blocks，要么一个都不拿（抛异常）。
        #         这种"全有或全无"避免了部分分配后失败导致的脏状态。
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero")

        with self._lock:
            available = len(self._free_block_ids)
            if num_blocks > available:
                raise OutOfBlocksError(requested=num_blocks, available=available)

            allocated_ids: list[int] = []
            t_now = time.perf_counter()
            for _ in range(num_blocks):
                block_id = self._free_block_ids.pop(0)  # FIFO: take from front
                blk = self._blocks[block_id]
                blk.seq_id = seq_id
                blk.ref_count = 1
                blk.is_dirty = False
                blk.tokens_used = 0
                blk.allocated_at = t_now
                allocated_ids.append(block_id)

            if seq_id not in self._seq_to_blocks:
                self._seq_to_blocks[seq_id] = []
            self._seq_to_blocks[seq_id].extend(allocated_ids)
            self._total_allocated += num_blocks
            return allocated_ids

    def free(self, seq_id: str) -> int:
        """Free ALL blocks owned by *seq_id*.

        Returns the number of blocks that were freed (0 if seq_id had none).
        """
        # [LEARN] 归还时把块内 tokens_used/is_dirty 一并重置，块才能真正被复用。
        with self._lock:
            block_ids = self._seq_to_blocks.pop(seq_id, [])
            count = len(block_ids)
            for block_id in block_ids:
                blk = self._blocks[block_id]
                blk.ref_count = 0
                blk.seq_id = None
                blk.tokens_used = 0
                blk.is_dirty = False
                self._free_block_ids.append(block_id)
            self._total_freed += count
            return count

    def free_block(self, block_id: int) -> None:
        """Free a single block by *block_id*.

        Raises
        ------
        BlockNotFoundError
            If *block_id* is not a valid pool block.
        """
        with self._lock:
            if block_id not in self._blocks:
                raise BlockNotFoundError(block_id)

            blk = self._blocks[block_id]
            if blk.is_free():
                return

            # Remove from the owning sequence's list
            owner = blk.seq_id
            if owner is not None and owner in self._seq_to_blocks:
                try:
                    self._seq_to_blocks[owner].remove(block_id)
                except ValueError:
                    pass
                if not self._seq_to_blocks[owner]:
                    del self._seq_to_blocks[owner]

            blk.ref_count = 0
            blk.seq_id = None
            blk.tokens_used = 0
            blk.is_dirty = False
            self._free_block_ids.append(block_id)
            self._total_freed += 1

    def write_token(self, seq_id: str, count: int = 1) -> list[int]:
        """Record that *count* tokens were written to *seq_id*'s current block.

        Handles overflow: when the current (last) block fills up, a new block
        is automatically allocated.

        Returns
        -------
        list[int]
            Block ids that were newly allocated during this call (empty if no
            new block was needed).

        Raises
        ------
        BlockAllocatorError
            If *seq_id* has no blocks allocated yet.
        OutOfBlocksError
            If a new block is needed but the pool is exhausted.
        """
        # [LEARN] 这是 decode 每一步都会调的热函数：把"已用 token 数 + count"
        #         换算成"需要多少块"，不够就自动新分配。向上取整公式：
        #         required = ceil(target_tokens / block_size)。
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return []

        with self._lock:
            block_ids = self._seq_to_blocks.get(seq_id)
            if not block_ids:
                raise BlockAllocatorError(
                    f"No blocks allocated for seq_id={seq_id!r}"
                )

            current_tokens = sum(self._blocks[bid].tokens_used for bid in block_ids)
            target_tokens = current_tokens + count
            required_blocks = (target_tokens + self._block_size - 1) // self._block_size
            extra_blocks = max(0, required_blocks - len(block_ids))
            available = len(self._free_block_ids)
            if extra_blocks > available:
                raise OutOfBlocksError(requested=extra_blocks, available=available)

            newly_allocated: list[int] = []
            t_now = time.perf_counter()
            for _ in range(extra_blocks):
                block_id = self._free_block_ids.pop(0)
                blk = self._blocks[block_id]
                blk.seq_id = seq_id
                blk.ref_count = 1
                blk.is_dirty = False
                blk.tokens_used = 0
                blk.allocated_at = t_now
                block_ids.append(block_id)
                newly_allocated.append(block_id)
            self._total_allocated += extra_blocks

            self._set_token_count_locked(block_ids, target_tokens)
            return newly_allocated

    def set_token_count(self, seq_id: str, count: int) -> None:
        """Set exact filled-token accounting across a sequence's blocks."""
        if count < 0:
            raise ValueError("count must be non-negative")

        with self._lock:
            block_ids = self._seq_to_blocks.get(seq_id)
            if not block_ids:
                raise BlockAllocatorError(
                    f"No blocks allocated for seq_id={seq_id!r}"
                )
            capacity = len(block_ids) * self._block_size
            if count > capacity:
                raise ValueError(
                    f"Token count {count} exceeds sequence capacity {capacity}"
                )
            self._set_token_count_locked(block_ids, count)

    def _set_token_count_locked(self, block_ids: list[int], count: int) -> None:
        """Distribute *count* filled slots in block order; caller holds lock."""
        # [LEARN] 把总 token 数按顺序"填充"到各块：前 n-1 块填满，最后一块填余数。
        #         例如 count=20, block_size=16 → 块1: 16，块2: 4。
        remaining = count
        for block_id in block_ids:
            tokens_used = min(remaining, self._block_size)
            blk = self._blocks[block_id]
            blk.tokens_used = tokens_used
            blk.is_dirty = tokens_used > 0
            remaining -= tokens_used

    def get_blocks(self, seq_id: str) -> list[int]:
        """Return list of block_ids owned by *seq_id* (in allocation order)."""
        with self._lock:
            return list(self._seq_to_blocks.get(seq_id, []))

    def get_block(self, block_id: int) -> Block:
        """Return the Block object for *block_id*.

        Raises
        ------
        BlockNotFoundError
            If *block_id* is not in the pool.
        """
        with self._lock:
            if block_id not in self._blocks:
                raise BlockNotFoundError(block_id)
            return self._blocks[block_id]

    # ── Eviction ──────────────────────────────────────────────────────────────

    def evict_lru(self, n: int = 1) -> list[str]:
        """Evict the *n* sequences whose last block was allocated earliest (LRU).

        Uses the existing ``free()`` method internally.  Returns the list of
        evicted seq_ids.  If fewer than *n* sequences are active, all are
        evicted.
        """
        # [LEARN] evict_lru / evict_largest 是给上层预留的淘汰策略。注意：
        #         当前 scheduler 用的是自己的 largest-first 抢占（_try_swap_out_victim
        #         + CPU 交换），并不直接调这两个方法。
        with self._lock:
            candidates = list(self._seq_to_blocks.keys())

        # Sort by allocated_at of the LAST block (oldest = LRU)
        def _last_block_time(sid: str) -> float:
            with self._lock:
                ids = self._seq_to_blocks.get(sid, [])
            if not ids:
                return float("inf")
            return self._blocks[ids[-1]].allocated_at

        candidates.sort(key=_last_block_time)
        to_evict = candidates[:n]

        evicted: list[str] = []
        for sid in to_evict:
            self.free(sid)
            with self._lock:
                self._total_evictions += 1
            evicted.append(sid)

        return evicted

    def evict_largest(self, n: int = 1) -> list[str]:
        """Evict the *n* sequences holding the most blocks.

        Tie-break: oldest last-block allocated_at wins (evicted first).
        Returns the list of evicted seq_ids.
        """
        with self._lock:
            candidates = list(self._seq_to_blocks.keys())

        def _sort_key(sid: str):
            with self._lock:
                ids = self._seq_to_blocks.get(sid, [])
            block_count = len(ids)
            last_time = self._blocks[ids[-1]].allocated_at if ids else float("inf")
            # Most blocks first; oldest last-block as tie-break
            return (-block_count, last_time)

        candidates.sort(key=_sort_key)
        to_evict = candidates[:n]

        evicted: list[str] = []
        for sid in to_evict:
            self.free(sid)
            with self._lock:
                self._total_evictions += 1
            evicted.append(sid)

        return evicted

    # ── Query ──────────────────────────────────────────────────────────────────

    def num_free_blocks(self) -> int:
        """Number of blocks currently in the free pool."""
        with self._lock:
            return len(self._free_block_ids)

    def num_used_blocks(self) -> int:
        """Number of blocks currently allocated to sequences."""
        with self._lock:
            return self._num_blocks - len(self._free_block_ids)

    def num_blocks_for_seq(self, seq_id: str) -> int:
        """Number of blocks allocated to *seq_id*."""
        with self._lock:
            return len(self._seq_to_blocks.get(seq_id, []))

    def num_tokens_for_seq(self, seq_id: str) -> int:
        """Number of filled token slots allocated to *seq_id*."""
        with self._lock:
            return sum(
                self._blocks[block_id].tokens_used
                for block_id in self._seq_to_blocks.get(seq_id, [])
            )

    def active_sequences(self) -> list[str]:
        """List of seq_ids that currently hold at least one block."""
        with self._lock:
            return list(self._seq_to_blocks.keys())

    def stats(self) -> dict:
        """Return a snapshot of allocator state.

        Returns
        -------
        dict with keys:
            num_blocks, block_size, free_blocks, used_blocks,
            active_sequences, total_allocated, total_freed,
            total_evictions, utilization, per_sequence
        """
        with self._lock:
            free = len(self._free_block_ids)
            used = self._num_blocks - free
            utilization = used / self._num_blocks if self._num_blocks > 0 else 0.0
            per_sequence = {
                sid: {
                    "num_blocks": len(ids),
                    "tokens_used": sum(
                        self._blocks[bid].tokens_used for bid in ids
                    ),
                }
                for sid, ids in self._seq_to_blocks.items()
            }
            return {
                "num_blocks": self._num_blocks,
                "block_size": self._block_size,
                "free_blocks": free,
                "used_blocks": used,
                "active_sequences": len(self._seq_to_blocks),
                "total_allocated": self._total_allocated,
                "total_freed": self._total_freed,
                "total_evictions": self._total_evictions,
                "utilization": utilization,
                "per_sequence": per_sequence,
            }
