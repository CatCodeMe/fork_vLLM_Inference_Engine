"""
engine/scheduler.py — Continuous Batching Scheduler (Phase 2).

Architecture
------------
The scheduler owns a single background asyncio task (`run_loop`) that runs
continuously for the lifetime of the server. Blocking model operations are
dispatched to worker threads so they do not stall the event loop.

                       ┌─────────────────────────────────────────┐
  add_request()  ──►  │  RequestQueue  (Phase 3)               │
                       └────────────────────┬────────────────────┘
                                            │  _schedule()
                                            │   1. evict finished
                                            │   2. admit new (up to headroom)
                                            │   3. prefill each new sequence
                                            │   4. one decode step per running seq
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │  running  (list[Sequence])              │
                       └────────────────────┬────────────────────┘
                                            │  state == "finished"
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │  finished  (list[Sequence])             │
                       └─────────────────────────────────────────┘

Decode step design (Gap 1 resolution)
--------------------------------------
`_decode_step` does NOT call Phase 1's `decode()` function, which would run
the full remaining-token loop and block all other sequences for their entire
generation.  Instead, it performs a raw single-token model forward pass — the
same pattern as the inner loop of `decode()` in sequential.py — one step at a
time.  After each step the scheduler can preempt and serve other sequences.

Threading
---------
Tokenization retains the scheduler's shared executor. Phase 4 prefill and
decode helpers are blocking functions dispatched with ``asyncio.to_thread``.
The scheduler awaits each call, so model execution remains serial until true
tensor-level batching is introduced in Phase 8.

Memory (Gap 3 acknowledgement)
--------------------------------
Each Sequence holds a separate `past_key_values` object.  For Qwen2-0.5B
(fp16) this grows by ~3 MB per generated token.  At max_batch_size=4 and
max_new_tokens=50 this adds ~600 MB on top of the ~950 MB model weight
footprint.  KV cache eviction and swapping are Phase 9 features.
"""

# [LEARN] ── 连续批处理（continuous batching）一句话总结 ──────────────────────
#   传统做法：一次服务一个请求，跑完整个生成（几十个 token）再换下一个 → 队头阻塞。
#   本引擎：每个 step 不跑完整请求，而是让所有活跃序列各前进 1 个 token，
#          谁完成就立刻腾出名额给排队的新请求。这就是"iteration-level"调度。
#
# [LEARN] 线程模型（读代码时最容易困惑的地方）：
#   - 调度循环 run_loop() 跑在 FastAPI 的 asyncio 事件循环里。
#   - 模型 forward 是阻塞的 CPU/GPU 计算，通过 asyncio.to_thread / run_in_executor
#     丢到线程池，避免卡住事件循环（否则 /health、/metrics 都会没响应）。
#   - 共享线程池只有 1 个 worker，所以同一时刻只有一个 forward 在跑；
#     真正 tensor-level 批处理是后续优化点（README 的 "What I'd Do Differently"）。
#
# [TRACE] 一次 _schedule() 的四步，记住这个顺序就能读懂整个引擎：
#   Step 0 swap-in  : 内存够的话，把之前换到 CPU 的序列换回来
#   Step 1 admit    : 从队列取新请求进 running（只登记，不做前向）
#   Step 2 prefill  : 给正在 prefill 的序列各推一个 chunk
#   Step 3 decode   : 给所有 decoding 序列各解 1 个 token，并清理 finished

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from inference_engine.config import Config
from inference_engine.engine.attention_wrapper import build_past_key_values, extract_new_token_kv
from inference_engine.engine.block_allocator import BlockAllocator, OutOfBlocksError
from inference_engine.engine.cpu_swap_manager import CPUSwapManager, CPUSwapError
from inference_engine.engine.kv_cache_config import (
    compute_kv_cache_config,
    format_kv_cache_report,
)
from inference_engine.engine.kv_cache_tracker import KVCacheTracker
from inference_engine.engine.metrics_aggregator import MetricsAggregator
from inference_engine.engine.paged_kv_cache import PagedKVCacheManager
from inference_engine.engine.prefill_utils import run_prefill_single
from inference_engine.engine.request_queue import RequestQueue
from inference_engine.engine.sequence import Sequence
from inference_engine.engine.stage_tracker import StageTracker
from inference_engine.engine.sequential import GenerationResult, get_memory_stats
from inference_engine.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)


# ── 轨迹日志的着色（只影响控制台，不进文件/JSON）─────────────────────────────

# [LEARN] 按“在生命周期里的哪一步”着色，一眼分清 prefill 和 decode：
#   1/8 2/8 8/8 = HTTP/入队（暗）   3/8 = 调度循环 / swap（蓝）
#   4/8 = admit（洋红）              5/8 = prefill（青）
#   6/8 = decode（绿）               7/8 = finish（黄）    swap-out（红）
_ANSI = {"dim": "2", "red": "31", "green": "32", "yellow": "33",
         "blue": "34", "magenta": "35", "cyan": "36"}
_STAGE_COLOR = [
    ("swap-out", "red"), ("3/8", "blue"), ("4/8", "magenta"),
    ("5/8", "cyan"), ("6/8", "green"), ("7/8", "yellow"),
    ("1/8", "dim"), ("2/8", "dim"), ("8/8", "dim"),
]


def _color_enabled() -> bool:
    """只在“真的在终端上看”时上色：重定向到文件/管道/NO_COLOR 时自动关。"""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TRACE_COLOR", "").lower() in ("0", "false", "no"):
        return False
    return sys.stderr.isatty()


def _colorize(stage: str) -> str:
    """按 stage 给日志文本上色；不开色时原样返回。"""
    if not _color_enabled():
        return stage
    for prefix, color in _STAGE_COLOR:
        if prefix in stage:
            return f"\033[{_ANSI[color]}m{stage}\033[0m"
    return stage


# ── Compatibility helper ───────────────────────────────────────────────────────

def _extract_kv_layer(
    past_key_values,
    layer_idx: int,
) -> tuple:
    """Extract (key, value) tensors for one layer from past_key_values.

    Handles three formats returned by different transformers versions:

    1. Legacy tuple-of-tuples (transformers < ~4.36):
       ``past_key_values[layer_idx]`` → ``(key_tensor, value_tensor)``
       shape: [batch, num_kv_heads, seq_len, head_dim]

    2. DynamicCache with key_cache / value_cache lists (transformers ~4.36-4.40):
       ``past_key_values.key_cache[layer_idx]`` → key tensor
       ``past_key_values.value_cache[layer_idx]`` → value tensor

    3. DynamicCache with layers list (transformers >= ~4.41):
       ``past_key_values.layers[layer_idx].keys`` → key tensor
       ``past_key_values.layers[layer_idx].values`` → value tensor
       shape: [batch, num_kv_heads, seq_len, head_dim]

    Returns
    -------
    (key_tensor, value_tensor)
        Both shaped [batch, num_kv_heads, seq_len, head_dim].
    """
    # [GOTCHA] transformers 大版本升级会改 KV cache 的内部结构，这里做了三种兼容。
    #         本项目当前装的是 transformers 5.x，走的是 Format 3（.layers[i].keys）。
    #         换模型/降级 transformers 时如果报 AttributeError，先看这里。
    # Format 3: DynamicCache with .layers list holding DynamicLayer objects
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values

    # Format 2: DynamicCache with .key_cache / .value_cache lists
    if hasattr(past_key_values, "key_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]

    # Format 1: Legacy tuple-of-tuples
    layer = past_key_values[layer_idx]
    return layer[0], layer[1]


class ContinuousBatchingScheduler:
    """Iteration-level continuous batching scheduler.

    Parameters
    ----------
    model
        A loaded HuggingFace causal LM in eval mode.
    tokenizer
        The corresponding tokenizer (pad_token already set).
    config
        Engine Config; reads ``max_batch_size`` and ``scheduler_poll_interval_ms``.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: Config,
        metrics_collector: Optional[MetricsCollector] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.kv_cache_config = compute_kv_cache_config(model, config)
        allocated_mb, _ = get_memory_stats(config.device)
        available_mb = getattr(config, "kv_cache_max_memory_mb", 1024.0)
        self.kv_tracker = KVCacheTracker(
            self.kv_cache_config, max_memory_mb=available_mb
        )
        logger.info(
            "%s\n  Current allocated memory: %.2f MB",
            format_kv_cache_report(self.kv_cache_config),
            allocated_mb,
        )

        # [LEARN] 这里把各 Phase 的组件"接线"到一起，是理解架构的最佳入口：
        #   BlockAllocator      → 只管逻辑块号（哪些块被谁占着）
        #   PagedKVCacheManager → 真正存 KV 的大 tensor（按块号索引）
        #   CPUSwapManager      → 内存不够时把块搬到 CPU
        # 三者职责分离：allocator 是"账本"，paged cache 是"仓库"，swap 是"临时仓"。
        # Phase 6: Block allocator
        self.block_allocator = BlockAllocator(
            num_blocks=config.kv_num_blocks,
            block_size=config.kv_block_size,
        )

        # Phase 7: Paged KV cache tensor pool
        self.paged_kv_cache = PagedKVCacheManager(
            kv_cache_config=self.kv_cache_config,
            block_allocator=self.block_allocator,
            config=config,
        )
        kv_stats = self.paged_kv_cache.stats()
        print(
            f"[PagedKVCache] Pool size: {kv_stats['pool_size_mb']:.1f} MB "
            f"({kv_stats['num_blocks']} blocks \u00d7 {kv_stats['block_size']} tokens)"
        )

        # Phase 9: CPU staging pool for GPU ⇔ CPU swapping
        self.cpu_swap_manager = CPUSwapManager(
            kv_cache_config=self.kv_cache_config,
            block_size=config.kv_block_size,
            num_cpu_blocks=config.kv_num_cpu_blocks,
        )

        self.max_batch_size: int = config.max_batch_size
        self.prefill_budget_tokens: int = config.prefill_budget_tokens
        self.decode_batch_limit: int = config.decode_batch_limit
        self.stage_tracker = StageTracker(history_size=500)
        self._futures: dict[str, asyncio.Future] = {}

        # Phase 3 request queue — replaces raw asyncio.Queue.
        # maxsize = 8× batch size gives reasonable queue depth before 503.
        # request_timeout_ms comes from config (default 30 s).
        self.request_queue: RequestQueue = RequestQueue(
            maxsize=config.max_batch_size * 8,
            request_timeout_ms=getattr(config, "request_timeout_ms", 30_000.0),
        )
        self.running: List[Sequence] = []      # currently active sequences
        history_size = max(1, getattr(config, "metrics_history_size", 100))
        self.finished: deque[Sequence] = deque(maxlen=history_size)
        self.total_finished: int = 0
        self.swapped_out: List[Sequence] = []  # Phase 9: sequences swapped to CPU

        # Phase 10: Unified metrics aggregator
        # If no MetricsCollector is provided by the server layer, create a
        # local one so the aggregator always has something to read from.
        if metrics_collector is None:
            metrics_collector = MetricsCollector(
                history_size=getattr(config, "metrics_history_size", 100)
            )
        self._metrics_collector = metrics_collector
        self.metrics_aggregator = MetricsAggregator(
            metrics_collector=self._metrics_collector,
            stage_tracker=self.stage_tracker,
            kv_tracker=self.kv_tracker,
            block_allocator=self.block_allocator,
            paged_kv_cache=self.paged_kv_cache,
            cpu_swap_manager=self.cpu_swap_manager,
            request_queue=self.request_queue,
            history_window_seconds=60.0,
        )

        # Control
        self._stop_event: asyncio.Event = asyncio.Event()
        self._loop_task: Optional[asyncio.Task] = None

        # Single shared executor — one warm thread, no per-token spawning.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="scheduler_inference"
        )

        # Metrics
        # list of (timestamp: float, batch_size: int)
        scheduler_history_size = history_size * 10
        self.batch_size_over_time: deque[Tuple[float, int]] = deque(
            maxlen=scheduler_history_size
        )
        # wall-clock duration of each _schedule() call in ms
        self.scheduler_step_latency_ms: deque[float] = deque(
            maxlen=scheduler_history_size
        )

        # Device resolved once from model parameters (next() 含义见 sequential.prefill)
        self._device: torch.device = next(model.parameters()).device

        logger.info(
            "ContinuousBatchingScheduler init: max_batch_size=%d device=%s",
            self.max_batch_size,
            self._device,
        )
        # [TRACE] 调度轨迹日志的 step 计数器（供 _trace 使用，见下方）
        self._trace_step: int = 0
        # [TRACE] 轨迹环形缓冲区：GET /trace 的数据源（有界，不会无限长）
        self._trace_log: deque = deque(maxlen=2000)

        # [LEARN] F32 的修复：只让 prefill 算最后一位的 logits，省掉 lm_head 的浪费。
        #         `logits_to_keep` 是 transformers >= 4.45 的参数，所以启动时探测一次 ——
        #         不支持就传空字典，退回旧行为（正确但慢）。
        #         注意对**最后一个** chunk 也是安全的：那时 logits 变成 [1,1,vocab]，
        #         `logits[:, -1, :]` 依旧取到最后一个位置。
        self._logits_to_keep: dict = (
            {"logits_to_keep": 1}
            if "logits_to_keep" in inspect.signature(model.forward).parameters
            else {}
        )
        logger.info(
            "logits_to_keep 优化: %s（transformers %s）",
            "启用" if self._logits_to_keep else "不支持，已退回",
            __import__("transformers").__version__,
        )

    # ── Trace logging ────────────────────────────────────────────────────────

    # [LEARN] 一条"跟着一个请求走完全程"的日志线，按 **N/8** 标号（共 8 步）：
    #
    #   1/8 收到      ← app_v2.endpoint_generate（HTTP 层）
    #   2/8 入队      ← scheduler.add_request（分词 + 建 Sequence）
    #   3/8 step     ← _schedule 开头：一次调度循环开始（自己递增 [step NNN]）
    #   3/8 swap-in  ← Step 0：把之前换到 CPU 的序列换回来
    #   5/8 swap-out ← Step 2 分配块失败时抢占（把某个序列换到 CPU）
    #   4/8 admit    ← Step 1：从 waiting 队列接纳进 running
    #   5/8 prefill  ← Step 2：每个 chunk 一行
    #   6/8 decode   ← Step 3：每个序列推进 1 个 token
    #   7/8 finish   ← Step 4：淘汰 finished（finish_reason + tokens）
    #   8/8 返回      ← app_v2.endpoint_generate（HTTP 响应）
    #
    # 两个标号各管一事，别看混：
    #   N/8         = 在【一个请求的生命周期】里的位置（固定 8 步，看它在干什么）
    #   [step NNN]  = 在【调度器】里的第几次循环（自己递增，看它跟谁在同一次调度里）
    #
    # 用法：`rg '\[(REQ|step [0-9]+)\]' server.log` 就能抽出全部轨迹。
    def record_trace(
        self,
        stage: str,
        msg: str,
        seq_id: Optional[str] = None,
        req: bool = False,
    ) -> None:
        """记录一条调度轨迹（同时进日志和环形缓冲区）。

        req=True 用于 HTTP 层/入队路径（标 [REQ]，不带 step 号）；
        否则用当前 step 标号标 [step NNN]。

        颜色只加在【日志输出】上（按 stage 着色），缓冲区里存的是纯文本 ——
        所以 `/trace` 的 JSON 和重定向到文件里的日志都不会混进 ANSI 转义码。
        """
        tag = "[REQ]" if req else f"[step {self._trace_step:03d}]"
        logger.info("%s %s %s", tag, _colorize(stage), msg)
        self._trace_log.append({
            "tag": tag,
            "step": None if req else self._trace_step,
            "stage": stage,
            "seq": seq_id,
            "msg": msg,
        })

    def get_trace(self, limit: int = 200) -> dict:
        """返回最近的调度轨迹（供 GET /trace 与 demo 脚本使用）。"""
        items = list(self._trace_log)
        return {
            "step": self._trace_step,
            "buffered": len(items),
            "returned": min(limit, len(items)),
            "entries": items[-limit:],
        }

    def _trace(self, stage: str, msg: str, seq_id: Optional[str] = None) -> None:
        """写一行带 step 标号的调度轨迹日志（record_trace 的简写）。"""
        self.record_trace(stage, msg, seq_id=seq_id)

    # ── Span 视图（OTel 风格）────────────────────────────────────────────────

    # [LEARN] 这是 `/spans` 的来源，也是本文档反复提到的"换个视角看同一份数据"：
    #         **不新增任何热路径埋点** —— 所有时间点都是 Sequence 上已有的字段：
    #             arrival_time / prefill_start_time / first_token_time / finish_time
    #             + prefill_chunk_latencies_ms / per_token_latencies_ms
    #         所以打开 span 视图对引擎性能零影响（事件日志那套也是同理）。
    #
    # 输出形状（有父子、有 start/dur，能直接画瀑布图）：
    #   request                      ← 根 span
    #     ├─ queue_wait
    #     ├─ prefill  (attrs: chunks, prompt_tokens)
    #     │    ├─ chunk#0 … chunk#N
    #     └─ decode   (attrs: tokens)
    #          ├─ token#0 … token#N
    def get_spans(self, limit: int = 50, max_children: int = 64) -> dict:
        """把最近处理过的请求导出成 span 树。

        limit        最多返回多少个请求（按总时长排序后取前 N）
        max_children 每个阶段最多展开多少个子 span（token 可能有几百个，避免 JSON 爆炸）
        """
        now = time.perf_counter()
        pool = list(self.finished) + list(self.running)
        spans: List[dict] = []

        for seq in pool:
            t0 = seq.arrival_time
            t_end = seq.finish_time or now          # 还在跑 → 用 now（未闭合）
            children: List[dict] = []

            # 1) 排队等待：arrival → 第一个 chunk 开始
            if seq.prefill_start_time:
                children.append({
                    "name": "queue_wait", "start_ms": 0.0,
                    "dur_ms": round((seq.prefill_start_time - t0) * 1000.0, 2),
                    "attrs": {},
                })

            # 2) prefill：第一个 chunk 开始 → 首个 token（含每个 chunk 的子 span）
            if seq.prefill_start_time:
                pf_start = seq.prefill_start_time
                pf_end = seq.first_token_time or t_end
                chunks = seq.prefill_chunk_latencies_ms
                sub, off = [], 0.0
                for i, ms in enumerate(chunks[:max_children]):
                    sub.append({"name": f"chunk#{i}", "start_ms": round(off, 2),
                                "dur_ms": round(ms, 2), "attrs": {}})
                    off += ms
                if len(chunks) > max_children:
                    sub.append({"name": f"…另 {len(chunks) - max_children} 个 chunk",
                                "start_ms": round(off, 2), "dur_ms": 0.0, "attrs": {}})
                children.append({
                    "name": "prefill",
                    "start_ms": round((pf_start - t0) * 1000.0, 2),
                    "dur_ms": round((pf_end - pf_start) * 1000.0, 2),
                    "attrs": {"chunks": len(chunks),
                              "prompt_tokens": len(seq.prompt_token_ids)},
                    "children": sub,
                })

            # 3) decode：首个 token → 结束（含每 token 的子 span）
            if seq.first_token_time:
                d_start = seq.first_token_time
                d_end = seq.finish_time or now
                lats = seq.per_token_latencies_ms
                sub, off = [], 0.0
                for i, ms in enumerate(lats[:max_children]):
                    sub.append({"name": f"token#{i}", "start_ms": round(off, 2),
                                "dur_ms": round(ms, 2), "attrs": {}})
                    off += ms
                if len(lats) > max_children:
                    sub.append({"name": f"…另 {len(lats) - max_children} 个 token",
                                "start_ms": round(off, 2), "dur_ms": 0.0, "attrs": {}})
                children.append({
                    "name": "decode",
                    "start_ms": round((d_start - t0) * 1000.0, 2),
                    "dur_ms": round((d_end - d_start) * 1000.0, 2),
                    "attrs": {"tokens": len(seq.generated_token_ids)},
                    "children": sub,
                })

            spans.append({
                "seq": seq.seq_id[:8],
                "name": "request",
                "start_ms": 0.0,
                "dur_ms": round((t_end - t0) * 1000.0, 2),
                "attrs": {
                    "state": seq.state,
                    "finish_reason": seq.finish_reason or None,
                    "prompt_tokens": len(seq.prompt_token_ids),
                    "generated_tokens": len(seq.generated_token_ids),
                    "ttft_ms": round(seq.ttft_ms, 2),
                },
                "children": children,
            })

        spans.sort(key=lambda x: -x["dur_ms"])
        return {"count": len(spans), "step": self._trace_step,
                "spans": spans[:limit] if limit else spans}

    @staticmethod
    def _kv_cache_len(past_kv: Any) -> int:
        """数一数 past_key_values 里已经缓存了多少个 token。

        [LEARN] 这是验证「三处 offset 一致」的工具（见 0014 §3）：
        `past_kv` 的长度就是模型用来推 position 的那个 offset，
        它必须等于 seq.prefill_offset，否则生成的 position 会错位（**且不报错**）。
        """
        if past_kv is None:
            return 0
        if hasattr(past_kv, "get_seq_length"):          # transformers >= 4.38 DynamicCache
            try:
                return int(past_kv.get_seq_length())
            except Exception:
                pass
        try:                                              # 旧式 tuple-of-tuples
            return int(past_kv[0][0].shape[-2])
        except Exception:
            return -1

    # ── Public API ────────────────────────────────────────────────────────────

    # [LEARN] 为什么这个方法是 async，而 _prefill_chunk_blocking / _decode_step_single
    #         这些“干重活”的却是普通同步方法？判据只有一条：
    #
    #             async def  ⟺  函数体里有 await
    #
    #         同步函数里写 await 是语法错误，所以“要不要 async”不是风格选择，
    #         而是被“要不要 await”唯一决定的。本文件全部方法都遵守这条（可用
    #         ast 扫一遍验证：async 的都有 await，同步的都没有）。
    #
    #         这里为什么必须 await：
    #           ① 分词是 CPU 密集的，直接调用会卡住整个事件循环（它还同时跑着
    #              run_loop 和其他 HTTP handler）→ 丢到 self._executor 线程池
    #           ② request_queue.enqueue 本身是 coroutine（内部要先清超时、再
    #              在锁内判满，见 request_queue.py:enqueue）→ 是 coroutine 就得 await
    #
    # [GOTCHA] 反过来极易搞错：**async def ≠ “会在后台跑”**。
    #         async def 若不 await，被 await 时会同步占住事件循环，只是白包了
    #         一层 coroutine。所以 _xxx_blocking 这类方法**必须保持同步**，由调用
    #         方用 asyncio.to_thread 搬出去（见 _schedule 里的两处 to_thread）。
    #         改名成 async 有两种静默故障，都不报错：
    #           A. 调用方写成 await self._decode_step_single(seq)
    #              → forward 直接在事件循环里跑，循环被独占（结果正确，吞吐垮掉）
    #           B. 调用方保持 run_in_executor(..., self._decode_step_single, seq)
    #              → 线程里只拿到一个 coroutine 对象，函数体**根本不会执行**，
    #                只在 GC 时冒一句 “coroutine was never awaited”
    async def add_request(
        self, prompt: str, max_new_tokens: int
    ) -> Tuple[Sequence, asyncio.Future]:
        """Tokenize *prompt*, create a Sequence in state 'waiting', enqueue it.

        Returns a ``(Sequence, asyncio.Future)`` tuple:
        - Sequence: the caller's request handle.
        - asyncio.Future: wired to the request lifecycle; resolved with
          TimeoutError on expiry, CancelledError on cancellation, or the
          finished Sequence on success.

        Raises QueueFullError if the RequestQueue is at capacity — propagated
        to the caller; the server layer converts it to HTTP 503.

        Tokenization runs in the shared executor to avoid blocking the event
        loop on long prompts.
        """
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        # [TRACE] 这里是 HTTP 请求进入引擎的入口：
        #         分词(在线程池) → 建 Sequence → 入队拿 future → 返回给 server。
        #         之后 server 在 endpoint_generate 里 await future 等待结果。
        loop = asyncio.get_running_loop()
        prompt_token_ids: List[int] = await loop.run_in_executor(
            self._executor,
            lambda: self.tokenizer(prompt, add_special_tokens=True)["input_ids"],
        )
        if not prompt_token_ids:
            raise ValueError("The prompt produced no input tokens")

        seq = Sequence.create(
            prompt=prompt,
            prompt_token_ids=list(prompt_token_ids),
            max_new_tokens=max_new_tokens,
        )
        # QueueFullError propagates to caller — do NOT catch here.
        future = await self.request_queue.enqueue(seq)
        self._futures[seq.seq_id] = future

        def _forget_completed_future(done_future: asyncio.Future) -> None:
            if self._futures.get(seq.seq_id) is done_future:
                self._futures.pop(seq.seq_id, None)

        future.add_done_callback(_forget_completed_future)
        logger.debug("Enqueued seq_id=%s prompt_len=%d", seq.seq_id, len(prompt_token_ids))
        # [TRACE] ① 入队：这是请求进入引擎的唯一点
        self.record_trace(
            "2/8 入队",
            f"seq={seq.seq_id[:8]} prompt_len={len(prompt_token_ids)} "
            f"max_new={max_new_tokens} 队列深度={len(self.request_queue)}"
            f"/{self.request_queue.maxsize}",
            seq_id=seq.seq_id,
            req=True,
        )
        return seq, future

    def _record_first_token(self, seq: Sequence) -> None:
        """Record the prefill token and finish immediately when appropriate."""
        # [GOTCHA] prefill 产出的第一个 token 可能就是 EOS，或者 max_new_tokens=1，
        #         这两种情况要在这里直接收尾；否则序列会多跑一步 decode。
        self.metrics_aggregator.record_token_generated(1)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and seq.generated_token_ids[-1] == eos_id:
            seq.finish_reason = "eos"
            seq.state = "finished"
        elif len(seq.generated_token_ids) >= seq.max_new_tokens:
            seq.finish_reason = "length"
            seq.state = "finished"

        if seq.is_finished():
            seq.past_key_values = None
            self.kv_tracker.unregister_sequence(seq.seq_id)

    def _fail_sequence(self, seq: Sequence, exc: Exception) -> None:
        """Move a sequence to a terminal error state without killing the loop."""
        logger.exception("Inference failed for seq_id=%s", seq.seq_id, exc_info=exc)
        seq.state = "finished"
        seq.finish_reason = "error"
        seq.error_message = str(exc)
        seq.past_key_values = None
        self.kv_tracker.unregister_sequence(seq.seq_id)

    # ── Internal: prefill ─────────────────────────────────────────────────────

    def _prefill_sequence(self, seq: Sequence) -> None:
        """Run the prefill pass for *seq* and transition it to 'decoding'.

        [GOTCHA] 遗留代码（legacy）：这是早期"整段一次性 prefill"的版本，
                 当前 _schedule() 走的是分块版 _prefill_chunk_blocking()，
                 本方法已不被生产路径调用，保留作为对照阅读。

        Sets:
        - seq.past_key_values
        - seq.generated_token_ids (first token appended)
        - seq.ttft_ms
        - seq.first_token_time
        - seq.queue_wait_time_ms
        - seq.state = "decoding"
        """
        # Record queue wait before blocking
        prefill_start = time.perf_counter()
        seq.queue_wait_time_ms = (prefill_start - seq.arrival_time) * 1000.0
        seq.state = "prefill"

        past_key_values, first_token_id, ttft_ms = run_prefill_single(
            self.model,
            seq.prompt_token_ids,
            self._device,
        )

        seq.past_key_values = past_key_values
        seq.ttft_ms = ttft_ms
        seq.first_token_time = time.perf_counter()
        seq.generated_token_ids.append(first_token_id)
        seq.state = "decoding"
        prompt_token_count = len(seq.prompt_token_ids)
        self.kv_tracker.register_sequence(seq.seq_id, prompt_token_count)
        seq.update_kv_stats(
            token_count=prompt_token_count,
            memory_mb=self.kv_tracker.sequence_memory_mb(seq.seq_id),
        )

        # Phase 6: allocate KV cache blocks for the prompt
        import math
        blocks_needed = math.ceil(prompt_token_count / self.config.kv_block_size)
        try:
            block_ids = self.block_allocator.allocate(seq.seq_id, blocks_needed)
            self.block_allocator.set_token_count(seq.seq_id, prompt_token_count)
        except OutOfBlocksError:
            # Phase 9: instead of killing the sequence, try to swap out the
            # largest running sequence to free device blocks, then retry.
            swapped_ok = self._try_swap_out_victim()
            if not swapped_ok:
                seq.state = "finished"
                seq.finish_reason = "oom"
                seq.past_key_values = None
                self.kv_tracker.unregister_sequence(seq.seq_id)
                return
            # Retry allocation once after swap-out freed some blocks
            try:
                block_ids = self.block_allocator.allocate(seq.seq_id, blocks_needed)
                self.block_allocator.set_token_count(seq.seq_id, prompt_token_count)
            except OutOfBlocksError:
                seq.state = "finished"
                seq.finish_reason = "oom"
                seq.past_key_values = None
                self.kv_tracker.unregister_sequence(seq.seq_id)
                return

        # Phase 7: write prompt KV tensors into paged pool
        # HuggingFace past_key_values may be a tuple-of-tuples (legacy) or
        # DynamicCache (transformers >= 4.38). Use the compat helper.
        past_kv = seq.past_key_values
        seq_len = len(seq.prompt_token_ids)
        for layer_idx in range(self.kv_cache_config.num_layers):
            layer_key, layer_val = _extract_kv_layer(past_kv, layer_idx)
            # shape: [1, num_kv_heads, seq_len, head_dim]
            for token_position in range(seq_len):
                key_slice = layer_key[0, :, token_position, :]
                val_slice = layer_val[0, :, token_position, :]
                self.paged_kv_cache.write_kv(
                    seq.seq_id, layer_idx, token_position, key_slice, val_slice
                )
        # Phase 8: release HuggingFace KV tensor — pool is now source of truth
        seq.past_key_values = None
        self._record_first_token(seq)

        logger.debug(
            "Prefill done: seq_id=%s ttft=%.1f ms first_token=%d",
            seq.seq_id,
            ttft_ms,
            first_token_id,
        )

    # ── Internal: chunked prefill ──────────────────────────────────────────────

    def _prefill_chunk_blocking(self, seq: Sequence) -> None:
        """Run ONE chunk of the prefill pass for *seq*.

        # [LEARN] 名字里的 _blocking 是**契约**：这是个会阻塞的同步函数，
        #         只能由 async 调用方通过 asyncio.to_thread / run_in_executor 派发，
        #         绝不能在 async 函数里直接 await 调用。改名的前提是先改所有调用点。

        This is the core of the chunked-prefill feature.  Instead of processing
        the full prompt in a single blocking forward pass (which delays decode
        for all other sequences), we advance by at most ``seq.prefill_chunk_size``
        tokens per scheduler step, then yield back to the event loop so other
        sequences can decode.

        Chunk logic
        -----------
        - First chunk  (seq.prefill_offset == 0): cold forward pass, no prior KV.
        - Middle chunks: reconstruct accumulated KV from the paged pool, then
          run a forward pass over the next slice of prompt tokens.
        - Final chunk  (offset+chunk reaches end of prompt): same as middle, but
          additionally samples the first output token and transitions to decoding.

        After each chunk:
        - New KV tensors are written into the paged pool.
        - Block-allocator token accounting is updated.
        - ``seq.prefill_offset`` is advanced.

        This is a BLOCKING function intended to run inside the shared executor.
        """
        # [TRACE] 一次调用 = 一个 chunk 的 prefill，逐步做：
        #   1) 首个 chunk：分配整段 prompt 所需块（一次性），记录排队时延
        #   2) 非首个 chunk：从 paged pool 重建已积累的 KV
        #   3) 对当前 chunk 做 forward
        #   4) 把新 KV 写回 paged pool，推进 prefill_offset
        #   5) 若是最后一个 chunk：采样首个输出 token，转入 decoding
        import math

        t_chunk_start = time.perf_counter()

        prompt_ids = seq.prompt_token_ids
        prompt_len = len(prompt_ids)
        chunk_start = seq.prefill_offset
        chunk_end = min(chunk_start + seq.prefill_chunk_size, prompt_len)
        chunk_ids = prompt_ids[chunk_start:chunk_end]
        chunk_len = len(chunk_ids)
        is_first_chunk = chunk_start == 0
        is_last_chunk = chunk_end >= prompt_len

        # ── Record queue wait and start time on the first chunk ───────────────
        # [LEARN] 第一个 chunk 才记录 queue_wait_time_ms：即从 Sequence 创建
        #         （arrival_time）到真正开始算的等待时间，是 TTFT 的重要组成。
        if is_first_chunk:
            seq.queue_wait_time_ms = (t_chunk_start - seq.arrival_time) * 1000.0
            seq.prefill_start_time = t_chunk_start
            seq.state = "chunked_prefilling"

            # Allocate blocks for the full prompt upfront (same as full prefill)
            blocks_needed = math.ceil(prompt_len / self.config.kv_block_size)
            try:
                block_ids = self.block_allocator.allocate(seq.seq_id, blocks_needed)
                self.block_allocator.set_token_count(seq.seq_id, 0)
            except OutOfBlocksError:
                swapped_ok = self._try_swap_out_victim()
                if not swapped_ok:
                    seq.state = "finished"
                    seq.finish_reason = "oom"
                    return
                try:
                    block_ids = self.block_allocator.allocate(seq.seq_id, blocks_needed)
                    self.block_allocator.set_token_count(seq.seq_id, 0)
                except OutOfBlocksError:
                    seq.state = "finished"
                    seq.finish_reason = "oom"
                    return

            # Register with KV tracker using prompt length
            self.kv_tracker.register_sequence(seq.seq_id, 0)

            # [TRACE] ④ 首个 chunk 的额外动作（注意是【整段 prompt】的块，不是本 chunk）
            free_blocks = self.block_allocator.num_free_blocks()
            self._trace(
                "5/8 prefill",
                f"seq={seq.seq_id[:8]} 首个 chunk：按整段 prompt 分配 {blocks_needed} 个块"
                f"（{prompt_len} token / block={self.config.kv_block_size}），"
                f"分配后池内剩余 {free_blocks} 块；排队等待 {seq.queue_wait_time_ms:.1f}ms",
                seq_id=seq.seq_id,
            )

        # ── Build past_key_values from paged pool (empty on first chunk) ──────
        if is_first_chunk:
            past_kv = None
        else:
            past_kv = build_past_key_values(
                seq_id=seq.seq_id,
                paged_kv_cache=self.paged_kv_cache,
                num_layers=self.kv_cache_config.num_layers,
                device=str(self._device),
                use_dynamic_cache=True,
            )
            # [TRACE] ④ 三处 offset 的一致性自检（0014 §3）：
            #         模型靠 past_kv 的长度推 position，它必须等于 prefill_offset
            rebuilt = self._kv_cache_len(past_kv)
            if rebuilt != chunk_start:
                logger.error(
                    "[S%03d] ④  ⚠️ offset 不一致：prefill_offset=%d 但 pool 里读到 %d 条 KV"
                    " —— position 会错位且不会报错（见 0014 §3）",
                    self._trace_step, chunk_start, rebuilt,
                )
            else:
                self._trace(
                    "5/8 prefill",
                    f"seq={seq.seq_id[:8]} 从 pool 重建 past_kv：{rebuilt} 条 KV"
                    f" == prefill_offset={chunk_start} ✓",
                    seq_id=seq.seq_id,
                )

        # ── Forward pass over this chunk ──────────────────────────────────────
        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
                # [WHY] 只有**最后一个** chunk 需要 logits（取最后一位采样首个 token）；
                #       中间的 chunk 只取 KV，logits 算了也是丢。而 lm_head 不便宜：
                #       chunk_len=128 时 logits 是 [1,128,151936] fp16 = 38.9 MB，
                #       lm_head 占该 chunk 全部线性算力的 ~28%。
                #       实测（本机 MPS，chunk_len=132）：134.4ms → 102.6ms，提速 23.7%。
                # [GOTCHA] logits_to_keep 是 transformers >= 4.45 才有的参数，
                #          所以 _logits_to_keep 是启动时探测出来的（见 __init__）：
                #          旧版本退回 {}，行为与修前一致（只是慢）。
                **self._logits_to_keep,
            )

        new_past_kv = outputs.past_key_values

        # ── Write new chunk KV into paged pool ────────────────────────────────
        # token_position is the global index within the full prompt.
        for token_pos_in_chunk in range(chunk_len):
            global_pos = chunk_start + token_pos_in_chunk
            for layer_idx in range(self.kv_cache_config.num_layers):
                layer_key, layer_val = _extract_kv_layer(new_past_kv, layer_idx)
                # layer_key shape: [1, num_kv_heads, total_tokens_so_far, head_dim]
                # We need the column corresponding to this specific prompt token.
                # The output KV spans [prior_kv_len .. prior_kv_len + chunk_len].
                # token_pos_in_chunk maps to that column.
                # Model outputs contain the prior cache followed by this chunk;
                # address the newly appended suffix, not the cache prefix.
                # [GOTCHA] 负索引技巧：source_position = token_pos_in_chunk - chunk_len
                #         等价于从新输出 KV 的末尾往前取 chunk_len 个列，正好是本 chunk
                #         新追加的那段（模型输出包含 prior cache + 本次 chunk）。
                source_position = token_pos_in_chunk - chunk_len
                key_slice = layer_key[0, :, source_position, :]
                val_slice = layer_val[0, :, source_position, :]
                self.paged_kv_cache.write_kv(
                    seq.seq_id, layer_idx, global_pos, key_slice, val_slice
                )

        # ── Advance offset ────────────────────────────────────────────────────
        seq.prefill_offset = chunk_end
        self.block_allocator.set_token_count(seq.seq_id, chunk_end)
        self.kv_tracker.update_sequence(seq.seq_id, chunk_end)
        seq.update_kv_stats(
            token_count=chunk_end,
            memory_mb=self.kv_tracker.sequence_memory_mb(seq.seq_id),
        )
        # [TRACE] ④ 本 chunk 的完整“前→后”变化（每 chunk 一行，边界一目了然）
        self._trace(
            "5/8 prefill",
            f"seq={seq.seq_id[:8]} chunk=[{chunk_start},{chunk_end}) {chunk_len} token"
            f" → 写 pool 位置 {chunk_start}–{chunk_end - 1}；"
            f"offset {chunk_start}→{chunk_end}；prompt 剩余 {prompt_len - chunk_end}",
            seq_id=seq.seq_id,
        )
        # [TRACE] 每个 chunk 的耗时 → /spans 拿它把 prefill 展开成 N 个 chunk 子 span
        seq.prefill_chunk_latencies_ms.append(
            (time.perf_counter() - t_chunk_start) * 1000.0
        )

        # ── If this was the last chunk, sample first token & transition ───────
        if is_last_chunk:
            logits = outputs.logits       # [1, chunk_len, vocab_size]
            first_token_id = int(logits[:, -1, :].argmax(dim=-1).item())
            seq.generated_token_ids.append(first_token_id)

            seq.ttft_ms = (time.perf_counter() - seq.prefill_start_time) * 1000.0
            seq.first_token_time = time.perf_counter()
            seq.state = "decoding"
            self._record_first_token(seq)
            # [TRACE] ④ 分块 prefill 收官：转 decoding（这个序列开始进 Step 3）
            self._trace(
                "5/8 prefill",
                f"seq={seq.seq_id[:8]} ✅ 最后一个 chunk 完成：共 "
                f"{math.ceil(prompt_len / seq.prefill_chunk_size)} 个 chunk，"
                f"采样首个 token={first_token_id}，ttft={seq.ttft_ms:.1f}ms，"
                f"state → decoding",
                seq_id=seq.seq_id,
            )
            logger.debug(
                "Chunked prefill done: seq_id=%s chunks=%d ttft=%.1f ms first_token=%d",
                seq.seq_id,
                math.ceil(prompt_len / seq.prefill_chunk_size),
                seq.ttft_ms,
                first_token_id,
            )
        else:
            logger.debug(
                "Chunk %d-%d / %d done for seq_id=%s (%.1f ms)",
                chunk_start,
                chunk_end - 1,
                prompt_len - 1,
                seq.seq_id,
                (time.perf_counter() - t_chunk_start) * 1000.0,
            )

    # ── Internal: single decode step ──────────────────────────────────────────

    def _decode_one_step_blocking(self, seq: Sequence) -> None:
        """Run ONE raw model forward pass for *seq*.

        [GOTCHA] 遗留包装器：内部直接调 _decode_step_single，当前生产路径不经过它。

        This is the inner loop of sequential.py's decode() extracted to run
        exactly once.  Bypasses decode() entirely — that function would block
        for the full remaining-token loop.

        Updates seq.past_key_values, appends the new token, records latency,
        and sets finish state if EOS or max_new_tokens is reached.

        This is a BLOCKING function intended to run inside the shared executor.
        """
        self._decode_step_single(seq)

    def _decode_step_single(self, seq: Sequence) -> None:
        """Run one blocking decode step — uses live past_key_values on the sequence.

        # [GOTCHA] 本方法必须保持同步（同 _prefill_chunk_blocking）。如果改成
        #         async def，run_in_executor 只会把 coroutine 对象当“结果”返回，
        #         函数体不执行、token 不增加，表现为序列永远跑不完。

        Performance design
        ------------------
        We keep ``seq.past_key_values`` alive between steps (the HuggingFace
        KV cache object returned by the previous forward pass).  This avoids
        reconstructing the full cache from the paged pool on every token, which
        is prohibitively expensive on MPS due to Metal dispatch overhead for
        the many small tensor operations involved (24 layers × read + permute +
        cache.update = ~240 ms overhead per token on MPS vs ~24 ms for the
        forward pass itself).

        The paged pool is still updated every step so eviction/swapping (Phase 9)
        remains accurate — we just never *read* from it during normal decode.
        When a sequence is swapped out and later swapped back in, its
        ``past_key_values`` is rebuilt from the pool at that point (handled by
        the swap-in path in ``_schedule``).
        """
        # [WHY] 这是整个引擎最反直觉的一处设计：paged pool 每步都在写，但 decode
        #       时"读"的是挂在 seq 上的 live past_key_values，而不是从 pool 拼回来。
        #       原因：MPS 上大量小 tensor 的 read+permute+update 开销约 240ms/token，
        #       而一次 forward 只要 ~24ms。pool 只保证 swap/eviction 时数据正确。
        t_step = time.perf_counter()

        # Use the live KV cache kept on the sequence (None only on first decode
        # step after chunked prefill, which sets state="decoding" but leaves
        # past_key_values=None — in that case we rebuild once from the pool).
        past_kv = seq.past_key_values
        if past_kv is None:
            past_kv = build_past_key_values(
                seq_id=seq.seq_id,
                paged_kv_cache=self.paged_kv_cache,
                num_layers=self.kv_cache_config.num_layers,
                device=str(self._device),
                use_dynamic_cache=True,
            )

        # The decode input token is the newest generated token.  Reconstruct
        # the existing cache first, then reserve its new slot so an unwritten
        # slot is never included in past_key_values.
        # [TRACE] 先给本步要写入的 token 预留一个块槽位，再决定输入 token。
        #         若块用完（OutOfBlocksError）→ 触发抢占 swap-out，再重试。
        token_position = len(seq.prompt_token_ids) + len(seq.generated_token_ids) - 1
        try:
            self.block_allocator.write_token(seq.seq_id, count=1)
        except OutOfBlocksError:
            swapped_ok = self._try_swap_out_victim(exclude_seq_id=seq.seq_id)
            if swapped_ok:
                try:
                    self.block_allocator.write_token(seq.seq_id, count=1)
                except OutOfBlocksError:
                    swapped_ok = False
            if not swapped_ok:
                seq.state = "finished"
                seq.finish_reason = "oom"
                seq.past_key_values = None
                self.kv_tracker.unregister_sequence(seq.seq_id)
                return

        # Determine next input token
        last_token_id = (
            seq.generated_token_ids[-1]
            if seq.generated_token_ids
            else seq.prompt_token_ids[-1]
        )
        input_ids = torch.tensor(
            [[last_token_id]], dtype=torch.long, device=self._device
        )

        # Forward pass — one token with live KV cache
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )

        # Greedy sample
        next_token_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        seq.generated_token_ids.append(next_token_id)

        # Keep the live KV cache on the sequence for the next step
        seq.past_key_values = outputs.past_key_values

        # Also write the new token's KV into the paged pool so eviction/swap
        # tracking remains accurate even though we don't read from it here.
        # [LEARN] 这就是"热路径用 live cache、冷路径靠 pool"的分工：平时不读 pool，
        #         但每步都写 pool，保证被 swap-out 后还能从 pool 完整恢复。
        new_past_kv = outputs.past_key_values
        for layer_idx in range(self.kv_cache_config.num_layers):
            key_slice, value_slice = extract_new_token_kv(
                new_past_kv, layer_idx, token_position
            )
            self.paged_kv_cache.write_kv(
                seq.seq_id, layer_idx, token_position, key_slice, value_slice
            )

        # Update KV tracker
        cached_tokens = token_position + 1
        self.kv_tracker.update_sequence(seq.seq_id, cached_tokens)
        seq.update_kv_stats(
            token_count=cached_tokens,
            memory_mb=self.kv_tracker.sequence_memory_mb(seq.seq_id),
        )

        # Phase 10: record token throughput for aggregated metrics
        self.metrics_aggregator.record_token_generated(1)

        # Record per-token latency
        step_ms = (time.perf_counter() - t_step) * 1000.0
        seq.per_token_latencies_ms.append(step_ms)

        # Check termination
        eos_id = self.tokenizer.eos_token_id
        eos_hit = eos_id is not None and next_token_id == eos_id
        length_hit = len(seq.generated_token_ids) >= seq.max_new_tokens

        if eos_hit:
            seq.finish_reason = "eos"
            seq.state = "finished"
            # Free live KV memory immediately on completion
            seq.past_key_values = None
            logger.debug("seq_id=%s finished (eos)", seq.seq_id)
        elif length_hit:
            seq.finish_reason = "length"
            seq.state = "finished"
            # Free live KV memory immediately on completion
            seq.past_key_values = None
            logger.debug("seq_id=%s finished (length)", seq.seq_id)

        if seq.is_finished():
            self.kv_tracker.unregister_sequence(seq.seq_id)

    async def _decode_step(self) -> None:
        """Run one decode step for every sequence in self.running.

        Sequences are decoded serially within the shared executor (one warm
        thread).  This is the Phase 2 constraint: true tensor-level batching
        across sequences is deferred to Phase 8.

        The iteration-level scheduling value is still demonstrated: at every
        call to _schedule(), all running sequences advance by exactly one token
        before any one sequence monopolises future steps.
        """
        if not self.running:
            return

        # [LEARN] 注意这里是 for 循环 + await，逐个序列串行 decode，不是真正的
        #         tensor batching。连续批处理的收益来自"每步轮转"而非并行加速。
        loop = asyncio.get_event_loop()
        for seq in list(self.running):   # snapshot — finished ones removed later
            if seq.state != "decoding":
                continue
            await loop.run_in_executor(
                self._executor,
                self._decode_step_single,
                seq,
            )

    def _resolve_sequence_future(self, seq: Sequence) -> None:
        """Resolve the lifecycle future associated with a finished sequence."""
        # [TRACE] 请求"完成"的收尾点：
        #   1) 解析 future → 挂在上面的 endpoint_generate 被唤醒并组装 HTTP 响应
        #   2) 记录指标（延迟、吞吐、内存）
        #   3) 先 clear paged pool，再 free 块（顺序不能反，见下方注释）
        seq.finish_time = seq.finish_time or time.perf_counter()
        self.total_finished += 1
        total_latency_ms = (seq.finish_time - seq.arrival_time) * 1000.0
        generated_tokens = len(seq.generated_token_ids)
        future = self._futures.pop(seq.seq_id, None)
        if future is not None and not future.done():
            future.set_result(seq)

        try:
            allocated_mb, reserved_mb = get_memory_stats(str(self._device))
            self._metrics_collector.append(
                GenerationResult(
                    prompt=seq.prompt,
                    generated_text=self.tokenizer.decode(
                        seq.generated_token_ids, skip_special_tokens=True
                    ),
                    prompt_tokens=len(seq.prompt_token_ids),
                    generated_tokens=generated_tokens,
                    ttft_ms=seq.ttft_ms,
                    total_latency_ms=total_latency_ms,
                    tokens_per_second=(
                        generated_tokens / total_latency_ms * 1000.0
                        if total_latency_ms > 0 else 0.0
                    ),
                    per_token_latencies_ms=list(seq.per_token_latencies_ms),
                    gpu_memory_allocated_mb=allocated_mb,
                    gpu_memory_reserved_mb=reserved_mb,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
        except Exception:
            logger.exception("Failed to record metrics for seq_id=%s", seq.seq_id)

        # Phase 10: record completion for throughput and SLO tracking
        self.metrics_aggregator.record_request_finished(seq.finish_reason)
        # Phase 7: zero paged pool slots before releasing blocks
        # [GOTCHA] 必须"先清零 pool，再释放块"。因为 clear_sequence 需要
        #         block_allocator.get_blocks(seq_id) 才能知道清哪些；一旦先
        #         free 了块号归属，就找不到对应的物理块了。
        try:
            self.paged_kv_cache.clear_sequence(seq.seq_id)
        except Exception:
            logger.exception("Failed to clear KV cache for seq_id=%s", seq.seq_id)
        finally:
            # Phase 6: release block allocator memory for this sequence
            self.block_allocator.free(seq.seq_id)

    # ── Internal: Phase 9 swap helpers ────────────────────────────────────────

    def _try_swap_out_victim(self, exclude_seq_id: Optional[str] = None) -> bool:
        """Find the largest running sequence and swap its KV blocks to CPU.

        Selection policy: largest by block count (most device memory freed per
        swap) rather than LRU.  Freeing one large sequence is more efficient
        than several small ones.

        Returns
        -------
        bool
            True if a victim was successfully swapped out; False if no
            candidates exist or the CPU pool is full.
        """
        # [LEARN] 抢占策略：优先换出"占用块最多"的序列（largest-first），
        #         而不是 LRU。因为一次大 swap 比多次小 swap 释放的设备内存更多、
        #         拷贝开销更划算。exclude_seq_id 用于避免把"正在申请内存的自己"换出去。
        candidates = [
            sequence
            for sequence in self.running
            if sequence.state == "decoding"
            and sequence.seq_id != exclude_seq_id
            and self.block_allocator.num_blocks_for_seq(sequence.seq_id) > 0
        ]
        if not candidates:
            return False

        # Pick the sequence holding the most device blocks
        victim = max(
            candidates,
            key=lambda s: self.block_allocator.num_blocks_for_seq(s.seq_id),
        )

        device_block_ids = self.block_allocator.get_blocks(victim.seq_id)
        if not device_block_ids:
            # Nothing to swap — sequence holds no blocks yet
            return False

        try:
            self.cpu_swap_manager.swap_out(
                victim.seq_id,
                device_block_ids,
                self.paged_kv_cache,
                self.block_allocator,
            )
        except CPUSwapError:
            return False

        # Transition victim to 'swapped' state and move it out of running
        victim.past_key_values = None
        victim.state = "swapped"
        self.running.remove(victim)
        self.swapped_out.append(victim)
        logger.debug(
            "Swapped out seq_id=%s (%d blocks) to CPU",
            victim.seq_id,
            len(device_block_ids),
        )
        # [TRACE] ⚡ 抢占：因为“新序列要整段 prompt 的块”而块不够，只好把运行中的
        #           某个序列临时搬到 CPU。注意选的是【占块最多】的那个。
        logger.info(
            "[S%03d] ⚡ swap-out seq=%s 换出 %d 个块到 CPU（largest-first）　"
            "→ running=%d swapped=%d 空闲块=%d",
            self._trace_step, victim.seq_id[:8], len(device_block_ids),
            len(self.running), len(self.swapped_out),
            self.block_allocator.num_free_blocks(),
            seq_id=victim.seq_id,
        )
        return True

    # ── Internal: scheduler step ──────────────────────────────────────────────

    async def _schedule(self) -> None:
        """Run prefill admission, chunked-prefill advancement, then one decode pass.

        Scheduling order per step
        -------------------------
        0. Swap-in: restore any sequences whose KV blocks were evicted to CPU.
        1. Admit: pull new sequences from the waiting queue into self.running,
           stamping their prefill_chunk_size and setting state to
           "chunked_prefilling".  No forward pass happens here.
        2. Chunked prefill: for each "chunked_prefilling" sequence, run exactly
           one chunk (≤ prefill_chunk_size tokens) via _prefill_chunk_blocking.
           This is bounded by the prefill_budget_tokens cap so a very long
           prompt cannot monopolise an entire step.  When the last chunk
           finishes the sequence transitions to "decoding" automatically.
        3. Decode: advance every "decoding" sequence by one token.
        4. Eviction: move finished sequences out of self.running.

        Why this order matters
        ----------------------
        By separating admission (Step 1) from the forward pass (Step 2), a
        newly-admitted sequence sits in self.running immediately but only
        processes chunk_size tokens before decode runs.  This bounds the
        maximum delay experienced by already-decoding sequences to the cost of
        one chunk forward pass rather than the full prompt forward pass.
        """
        step_start = time.perf_counter()
        running_limit = min(self.max_batch_size, self.decode_batch_limit)

        # Timeouts must progress even while the active batch is full.
        await self.request_queue.expire_timed_out()

        # [TRACE] 每个 step 的头部：三个容器的当前尺寸，一眼看出"谁在等、谁在跑"
        self._trace_step += 1
        self._trace(
            "3/8 step",
            f"running={len(self.running)} waiting={len(self.request_queue)} "
            f"swapped={len(self.swapped_out)} 上限={running_limit}",
        )

        # --- Phase 9: Swap-in check ---
        # [TRACE] Step 0：把之前被换到 CPU 的序列换回设备。只有设备块够才换，
        #         并且受 running_limit 限制，避免换回来又超员。
        # Before admitting new sequences, try to restore any swapped-out
        # sequences if the device pool now has enough room.
        for victim in list(self.swapped_out):
            if len(self.running) >= running_limit:
                break
            swap_record = self.cpu_swap_manager.get_swapped_sequence(victim.seq_id)
            if swap_record is None:
                # Already cleaned up — remove from list
                self.swapped_out.remove(victim)
                continue
            if self.block_allocator.num_free_blocks() >= swap_record.original_num_blocks:
                try:
                    self.cpu_swap_manager.swap_in(
                        victim.seq_id,
                        self.paged_kv_cache,
                        self.block_allocator,
                    )
                    victim.state = "decoding"
                    self.swapped_out.remove(victim)
                    self.running.append(victim)
                    # [TRACE] ② 抢占后换回（仅在实际发生时打，避免每步刷屏）
                    self._trace(
                        "3/8 swap-in",
                        f"seq={victim.seq_id[:8]} 块={swap_record.original_num_blocks} "
                        f"→ 回到 decoding（swapped 还剩 {len(self.swapped_out)}）",
                        seq_id=victim.seq_id,
                    )
                    logger.debug(
                        "Swapped in seq_id=%s back to decoding", victim.seq_id
                    )
                except OutOfBlocksError:
                    continue  # still not enough room, try next iteration

        # --- Stage 1: Admit new sequences (no prefill forward pass here) ---
        # [TRACE] Step 1：只做"登记"——把请求从队列搬进 running、标成
        #         chunked_prefilling。真正的计算在 Step 2，这样新序列不会在
        #         一个 step 里霸占整个 forward。
        # Sequences are added to self.running with state "chunked_prefilling".
        # The actual forward pass happens in Stage 2 below.
        admit_start = time.perf_counter()
        sequences_admitted = 0

        while len(self.running) < running_limit:
            queued = await self.request_queue.dequeue()
            if queued is None:
                break
            seq = queued.sequence
            # Stamp the chunk size from config (frozen for this sequence's lifetime)
            seq.prefill_chunk_size = min(
                self.config.prefill_chunk_size, self.prefill_budget_tokens
            )
            seq.state = "chunked_prefilling"
            self.running.append(seq)
            self.request_queue.mark_admitted()
            sequences_admitted += 1
            # [TRACE] ③ admit：只登记，不做前向。记下 frozen 的 chunk 大小
            self._trace(
                "4/8 admit",
                f"seq={seq.seq_id[:8]} prompt_len={len(seq.prompt_token_ids)} "
                f"chunk_size={seq.prefill_chunk_size} chunk数="
                f"{-(-len(seq.prompt_token_ids) // max(seq.prefill_chunk_size, 1))} "
                f"(running {len(self.running)}/{running_limit})",
                seq_id=seq.seq_id,
            )

        # --- Stage 2: Advance chunked prefill for in-progress sequences ---
        # [TRACE] Step 2：每个 prefill 中的序列最多推进一个 chunk，且总 token 数
        #         受 prefill_budget_tokens 限制；预算用尽就 break，让位给 decode。
        prefill_start = time.perf_counter()
        tokens_this_iteration = 0
        sequences_prefilled = 0   # counts sequences that finished their last chunk

        # [TRACE] ③ 背压：批已满但队列还有人等，打一行（这是 503 的上游）
        if sequences_admitted == 0 and len(self.request_queue) > 0:
            self._trace(
                "4/8 admit",
                f"跳过：running 已满（{len(self.running)}/{running_limit}），"
                f"队列还有 {len(self.request_queue)} 个等待",
            )

        for seq in list(self.running):
            if seq.state != "chunked_prefilling":
                continue
            # Respect the per-step token budget to avoid starving decode.
            chunk_tokens = min(seq.prefill_chunk_size, len(seq.prompt_token_ids) - seq.prefill_offset)
            if tokens_this_iteration + chunk_tokens > self.prefill_budget_tokens:
                # Budget exhausted — this sequence will get its chunk next step.
                break
            offset_before = seq.prefill_offset
            try:
                # [GOTCHA] 这两处 to_thread 是 _xxx_blocking 保持同步的唯一理由。
                #         写成 await self._prefill_chunk_blocking(seq) 会独占事件循环；
                #         若把被调方改成 async def，则函数体不执行（见 add_request 上方注释）。
                await asyncio.to_thread(self._prefill_chunk_blocking, seq)
                tokens_this_iteration += seq.prefill_offset - offset_before
            except Exception as exc:
                self._fail_sequence(seq, exc)
            if seq.is_prefill_done() and seq.generated_token_ids:
                sequences_prefilled += 1  # last chunk completed
            elif seq.state == "finished":
                # OOM during chunk allocation — will be evicted in Stage 3
                pass

        prefill_latency_ms = (time.perf_counter() - prefill_start) * 1000.0
        self.stage_tracker.record_prefill(
            sequences_prefilled=sequences_prefilled,
            tokens_prefilled=tokens_this_iteration,
            latency_ms=prefill_latency_ms,
            budget_tokens=self.prefill_budget_tokens,
        )

        # --- Stage 3: Decode + eviction ---
        # [TRACE] Step 3：所有 decoding 序列各解 1 个 token；finished 的移出 running
        #         并解析 future。这一步是"连续"的关键——完成的序列立刻让出名额。
        decode_start = time.perf_counter()
        sequences_decoded = 0
        still_running: List[Sequence] = []
        for seq in list(self.running):
            if seq.state == "swapped":
                continue
            if seq.state == "decoding":
                generated_before = len(seq.generated_token_ids)
                try:
                    await asyncio.to_thread(self._decode_step_single, seq)   # 同上：同步体必须搬出去
                    if len(seq.generated_token_ids) > generated_before:
                        sequences_decoded += 1
                        # [TRACE] ⑤ decode：解完就记（保证同一序列的 ⑤ 在 ⑥ 之前，
                        #           也保证 ⑥ 在它引起的 ⑦ 之前）
                        self._trace(
                            "6/8 decode",
                            f"seq={seq.seq_id[:8]} → {len(seq.generated_token_ids)}"
                            f"/{seq.max_new_tokens} token",
                            seq_id=seq.seq_id,
                        )
                except Exception as exc:
                    self._fail_sequence(seq, exc)
            if seq.state == "swapped":
                continue
            if not seq.is_finished():
                still_running.append(seq)
            else:
                self.finished.append(seq)
                # [TRACE] ⑥ 淘汰：必须在 _resolve_sequence_future 之前记，
                #           否则 handler 那边的 ⑦ 会先出现在轨迹里（因果倒置）
                self._trace(
                    "7/8 finish",
                    f"seq={seq.seq_id[:8]} reason={seq.finish_reason} "
                    f"tokens={len(seq.generated_token_ids)}/{seq.max_new_tokens} "
                    f"ttft={seq.ttft_ms:.1f}ms → 清 pool + 还块 + future.set_result",
                    seq_id=seq.seq_id,
                )
                self._resolve_sequence_future(seq)
        self.running = still_running

        decode_latency_ms = (time.perf_counter() - decode_start) * 1000.0
        self.stage_tracker.record_decode(
            sequences_decoded=sequences_decoded,
            latency_ms=decode_latency_ms,
            batch_limit=self.decode_batch_limit,
        )

        step_latency_ms = (time.perf_counter() - step_start) * 1000.0
        self.scheduler_step_latency_ms.append(step_latency_ms)
        self.batch_size_over_time.append((time.perf_counter(), len(self.running)))

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def run_loop(self) -> None:
        """Main scheduler loop — runs as a background asyncio Task.

        Idle sleep interval is ``scheduler_poll_interval_ms / 1000`` seconds
        (default 0.001 s = 1 ms) when both the waiting queue and running list
        are empty.
        """
        idle_sleep_s = self.config.scheduler_poll_interval_ms / 1000.0
        logger.info("Scheduler run_loop started (idle_sleep=%.3f s)", idle_sleep_s)

        while not self._stop_event.is_set():
            if (
                not self.running
                and not self.swapped_out
                and len(self.request_queue) == 0
            ):
                # [LEARN] 完全空闲（无 running/无 swap/无排队）时睡 1ms 再查，
                #         避免空转烧 CPU。默认 scheduler_poll_interval_ms=1.0。
                await asyncio.sleep(idle_sleep_s)
                continue
            await self._schedule()

        logger.info("Scheduler run_loop stopped.")

    def start(self) -> None:
        """Schedule run_loop() as a background asyncio Task.

        Must be called from within a running event loop (e.g. inside a FastAPI
        lifespan context).
        """
        if self._stop_event.is_set():
            raise RuntimeError("A stopped scheduler cannot be restarted")
        if self._loop_task is not None and not self._loop_task.done():
            raise RuntimeError("Scheduler is already running")
        self._loop_task = asyncio.create_task(self.run_loop(), name="scheduler_loop")

    async def stop(self) -> None:
        """Gracefully stop the scheduler loop and await its completion."""
        self._stop_event.set()
        if self._loop_task is not None:
            try:
                await asyncio.wait_for(self._loop_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Scheduler loop did not stop within 5 s; cancelling.")
                self._loop_task.cancel()
                try:
                    await self._loop_task
                except asyncio.CancelledError:
                    pass
        for future in list(self._futures.values()):
            if not future.done():
                future.cancel()
        self._futures.clear()
        self._executor.shutdown(wait=False)
        logger.info("Scheduler stopped. Finished %d sequences.", self.total_finished)

    # ── Public metrics API (Phase 10) ─────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Return a single unified metrics dict via MetricsAggregator.

        This is the ONLY method the server layer should call for metrics.
        All tracker stats, derived throughput, SLO compliance, and latency
        percentiles are assembled inside MetricsAggregator.full_report().
        """
        report = self.metrics_aggregator.full_report(
            requests_in_flight=len(self.running) + len(self.swapped_out),
            requests_waiting=len(self.request_queue),
        )
        report["summary"] = self._metrics_collector.compute_summary()
        report["scheduler"] = {
            "batch_size_over_time": list(self.batch_size_over_time),
            "scheduler_step_latency_ms": list(self.scheduler_step_latency_ms),
        }
        report["sequences"] = [
            {
                "seq_id": seq.seq_id,
                "state": seq.state,
                "finish_reason": seq.finish_reason,
                "prompt_tokens": len(seq.prompt_token_ids),
                "generated_tokens": len(seq.generated_token_ids),
                "ttft_ms": seq.ttft_ms,
                "queue_wait_time_ms": seq.queue_wait_time_ms,
            }
            for seq in self.finished
        ]
        return report
