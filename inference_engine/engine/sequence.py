"""
engine/sequence.py — Sequence state dataclass for the Phase 2 continuous
batching scheduler.

Each request submitted to the scheduler is wrapped in a Sequence object that
carries all mutable state through its lifetime:


    waiting   →  prefill   →  decoding   →  finished
       ↑ created    ↑ admitted    ↑ first token    ↑ EOS or max_new_tokens

    waiting   →  chunked_prefilling  →  decoding   →  finished
       ↑ created    ↑ admitted (chunked)   ↑ all chunks done

    waiting   →  expired    (timed out in queue before prefill)
    waiting   →  cancelled  (explicit cancel before prefill)
    decoding  →  swapped    (KV blocks evicted to CPU under memory pressure;
                             re-enters decoding after swap-in)

    Full valid states in lifecycle order:
        waiting | prefill | chunked_prefilling | decoding | finished | expired | cancelled | swapped

The `past_key_values` field holds the per-sequence HuggingFace KV cache.
Storing KV caches per-sequence is the Phase 2 approach; unified KV cache
management (with eviction/swapping) is Phase 9.

Memory note: for Qwen2-0.5B (fp16) each sequence's KV cache grows by
approximately 3 MB per generated token.  At max_batch_size=4 and
max_new_tokens=50 this adds ~600 MB on top of the ~950 MB model weight
footprint.  Keep max_batch_size ≤ 4 and max_new_tokens ≤ 50 unless you have
confirmed available headroom.
"""

from __future__ import annotations

# [LEARN] Sequence 是整个引擎的"状态载体"。一个 HTTP 请求进来后，会被包装成
#         Sequence，然后在调度器里经历：
#             waiting → (chunked_)prefilling → decoding → finished
#         沿途所有可变状态（KV cache、已生成 token、耗时、占用块）都挂在这个对象上。
#         读调度器代码时，先看 Sequence 有哪些字段，就大致知道调度器在管什么。
# [TRACE] 关键：理解 state 字段的每一次赋值，就等于理解调度器的整个生命周期。

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Sequence:
    """Full mutable state for one scheduled generation request.

    Fields
    ------
    seq_id
        UUID4 hex string — unique identifier for this sequence.
    prompt
        Raw text of the user prompt.
    prompt_token_ids
        Token-id list produced by the tokenizer for `prompt`.
    generated_token_ids
        Token ids produced so far, including the first token from prefill.
        Grows by one per scheduler decode step.
    max_new_tokens
        Upper bound on tokens to generate.
    state
        Lifecycle state.  Mutated exclusively by the scheduler or
        RequestQueue:

        Normal path:   ``"waiting"`` → ``"prefill"`` → ``"decoding"`` → ``"finished"``
        Chunked path:  ``"waiting"`` → ``"chunked_prefilling"`` → ``"decoding"`` → ``"finished"``
        Timeout path:  ``"waiting"`` → ``"expired"``
        Cancel path:   ``"waiting"`` → ``"cancelled"``
        Swap path:     ``"decoding"`` → ``"swapped"`` → ``"decoding"``

        All valid states:
            ``waiting | prefill | chunked_prefilling | decoding | finished | expired | cancelled | swapped``
    past_key_values
        HuggingFace KV cache returned by the last model forward pass.
        ``None`` until prefill completes.
    ttft_ms
        Time-to-first-token in milliseconds.  Set at the end of prefill.
    arrival_time
        ``time.perf_counter()`` timestamp when the Sequence was created.
        Used to compute ``queue_wait_time_ms``.
    first_token_time
        ``time.perf_counter()`` timestamp when the first generated token was
        appended (i.e. at the end of prefill).  ``0.0`` until then.
    per_token_latencies_ms
        Wall-clock latency (ms) for each individual decode step.
        Does **not** include the prefill step; that is captured in ``ttft_ms``.
    finish_reason
        ``"length"``  — stopped because ``len(generated_token_ids) >= max_new_tokens``
        ``"eos"``     — stopped because EOS token was produced
        ``""``        — not yet finished
    queue_wait_time_ms
        Time (ms) from ``arrival_time`` to the moment prefill began.
        Set by the scheduler when the sequence is admitted from the waiting
        queue.  ``0.0`` until then.
    prefill_offset
        Number of prompt tokens already processed in chunked-prefill mode.
        Starts at 0; advances by ``prefill_chunk_size`` each step; once it
        reaches ``len(prompt_token_ids)`` the sequence transitions to
        ``"decoding"``.  Unused in the legacy full-prefill path.
    prefill_chunk_size
        Tokens to process per scheduler step during chunked prefill.
        Frozen at admission time from ``config.prefill_chunk_size``.
    prefill_start_time
        ``time.perf_counter()`` when the first chunk began.  Used to compute
        TTFT across potentially multiple chunks.  ``0.0`` until the first
        chunk starts.
    """

    seq_id: str
    prompt: str
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    max_new_tokens: int
    # [TRACE] state 是调度器与本对象的"契约"：
    #   waiting           刚入队，还没被 admit
    #   prefill           旧版整段 prefill（_prefill_sequence）进行中
    #   chunked_prefilling 分块 prefill 进行中（Phase 4 默认路径）
    #   decoding          已有首个 token，每步解一个 token
    #   finished          结束（EOS / 达到 max_new_tokens / OOM）
    #   expired/cancelled 入队后超时或被取消，从未真正跑过
    #   swapped           KV 被换到 CPU（Phase 9 抢占）
    # [GOTCHA] 只有调度器 / RequestQueue 能改 state；别在别处随手改，否则会破坏调度。
    state: str  # waiting | prefill | chunked_prefilling | decoding | finished | expired | cancelled | swapped
    # [WHY] Phase 2 把 KV cache 直接挂在 Sequence 上（每个序列一份 HF cache）。
    #       到 Phase 7/8 之后，KV 的真正来源是 paged pool；字段保留是为了让 decode
    #       热路径复用上一轮 forward 的 cache，避免每 token 从 pool 重建（MPS 上极慢）。
    # [TRACE] 结束或换出（swap-out）时才把它置回 None。
    past_key_values: Any              # HuggingFace KV cache; None until prefill done
    ttft_ms: float
    arrival_time: float
    first_token_time: float
    per_token_latencies_ms: List[float]
    finish_reason: str                # "length" | "eos" | ""
    queue_wait_time_ms: float         # arrival → prefill start; set by scheduler
    kv_token_count: int = 0
    kv_memory_mb: float = 0.0
    # [LEARN] 分块 prefill（chunked prefill）的游标：prompt 太长时不让它一次占满整个
    #         step，而是每次只推进 prefill_chunk_size 个 token，好让其它序列的 decode
    #         有机会插进来。prefill_offset 记录"prompt 已经吃到第几个 token"。
    # [GOTCHA] 它只统计 prompt token，不含生成的 token；判 prefill 是否完成要用它。
    prefill_offset: int = 0           # tokens processed so far (chunked prefill)
    prefill_chunk_size: int = 128     # frozen at admission from config
    prefill_start_time: float = 0.0   # perf_counter() when first chunk started
    finish_time: float = 0.0          # perf_counter() when terminal state was reached
    # [LEARN] 每个 prefill chunk 的耗时（ms），只会 append，不会重算。
    #         它是 `/spans` 把 prefill 画成"一个 span + N 个 chunk 子 span"的依据：
    #         span 视图不新增热路径埋点，直接由本字段 + 各个 *_time 推导。
    prefill_chunk_latencies_ms: List[float] = field(default_factory=list)
    error_message: str = ""           # internal failure detail, if any

    # ── Convenience helpers ───────────────────────────────────────────────────

    def is_finished(self) -> bool:
        """Return True when the sequence has reached terminal state."""
        return self.state == "finished"

    def is_prefill_done(self) -> bool:
        """Return True when all prompt tokens have been processed in chunked prefill."""
        # [LEARN] 分块 prefill 的"完成"定义：prompt 的每个 token 都被前向过至少一次。
        #         到达这一步后，调度器才会把它当"产出首个 token"的序列处理。
        return self.prefill_offset >= len(self.prompt_token_ids)

    def total_tokens(self) -> int:
        """Total token count: prompt tokens + generated tokens so far."""
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    def update_kv_stats(self, token_count: int, memory_mb: float) -> None:
        """Update the informational snapshot of this sequence's KV cache."""
        self.kv_token_count = token_count
        self.kv_memory_mb = memory_mb

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        prompt: str,
        prompt_token_ids: List[int],
        max_new_tokens: int,
    ) -> "Sequence":
        """Create a new Sequence in the ``'waiting'`` state.

        Assigns a fresh UUID4, records the current time as ``arrival_time``,
        and initialises all mutable fields to empty / zero.
        """
        # [TRACE] arrival_time 就是"请求到达时刻"。后面 queue_wait_time_ms 和
        #         total_latency_ms 都以它为起点，所以放置断点看这个值能对上排队时延。
        return cls(
            seq_id=uuid.uuid4().hex,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=[],
            max_new_tokens=max_new_tokens,
            state="waiting",
            past_key_values=None,
            ttft_ms=0.0,
            arrival_time=time.perf_counter(),
            first_token_time=0.0,
            per_token_latencies_ms=[],
            finish_reason="",
            queue_wait_time_ms=0.0,
            prefill_offset=0,
            prefill_chunk_size=128,   # overwritten by scheduler at admission
            prefill_start_time=0.0,
            finish_time=0.0,
            error_message="",
        )
