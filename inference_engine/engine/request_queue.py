"""
engine/request_queue.py — Phase 3 request queue with timeout, cancellation,
and backpressure.

Replaces the raw asyncio.Queue used in Phase 2 with a structured layer that
adds:
  - Hard capacity cap (maxsize) with QueueFullError on overflow
  - Per-request timeout: requests that wait longer than request_timeout_ms
    before entering prefill are expired and their futures resolved with
    asyncio.TimeoutError
  - Explicit cancellation by seq_id
  - Structured stats (queue_depth, total_enqueued, total_expired, total_admitted)

Internal storage
----------------
Uses a plain list (not asyncio.Queue) to allow O(n) expiry scanning and
O(1) front-of-queue dequeue.  All mutations are protected by asyncio.Lock,
which is non-blocking in the asyncio sense (it does not release the GIL to
a thread pool) and appropriate here because every operation is called from
the scheduler's single event-loop task.

Priority field
--------------
QueuedRequest.priority is wired and stored but ordering is NOT changed in
Phase 3 — insertion order (FIFO within all priorities) is preserved.
Priority-aware scheduling is reserved for a future phase.

Future wiring
-------------
Each QueuedRequest carries an asyncio.Future.  It is resolved with an
exception on timeout or cancellation, or with the finished Sequence by the
scheduler.  The server awaits this future directly.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional

from inference_engine.engine.sequence import Sequence

# [LEARN] RequestQueue 解决的是"请求排队与背压"：
#         - 队列满了 → QueueFullError → 服务层转成 HTTP 503（快速失败，不拖垮引擎）
#         - 等太久还没被调度 → expired，future 抛 asyncio.TimeoutError → HTTP 504
# [WHY] 用普通 list 而不是 asyncio.Queue：因为超时扫描需要遍历整个队列（O(n)），
#       而取队首是 O(1) 的 pop(0)；asyncio.Queue 不提供这两种能力。
# [GOTCHA] 所有操作都在"调度器所在的事件循环"里调用，所以用 asyncio.Lock 即可，
#         不需要线程锁。


# ── Exception ─────────────────────────────────────────────────────────────────


class QueueFullError(Exception):
    """Raised by RequestQueue.enqueue() when the queue has reached maxsize."""

    # [TRACE] 这个异常会一直冒泡到 server/app_v2.py 的 endpoint_generate，
    #         在那里被翻译成 HTTP 503 "Server at capacity, retry later"。
    def __init__(self, maxsize: int) -> None:
        super().__init__(f"Request queue is full ({maxsize} requests waiting)")
        self.maxsize = maxsize


# ── QueuedRequest dataclass ───────────────────────────────────────────────────


@dataclass
class QueuedRequest:
    """Wraps a Sequence with queue-management metadata.

    Fields
    ------
    sequence
        The Sequence object representing this generation request.
    enqueue_time
        ``time.perf_counter()`` when the request entered the queue.
        Used by expire_timed_out() to compute elapsed wait time.
    future
        An ``asyncio.Future`` created at enqueue time.  Resolved with:
        - ``asyncio.TimeoutError`` on expiry
        - ``asyncio.CancelledError`` on explicit cancellation
        - the finished Sequence on successful completion
    priority
        Lower integer = higher priority.  Reserved for future use; ordering
        is NOT changed in Phase 3 — FIFO is preserved across all priorities.
    """

    sequence: Sequence
    enqueue_time: float
    # [LEARN] future 是"请求生命周期"的信号灯：服务器层 await 它，谁把它 set_* ，
    #         谁就决定了这个 HTTP 请求最终返回 成功 / 504 / 取消。
    #         - 超时   → set_exception(asyncio.TimeoutError)
    #         - 取消   → set_exception(asyncio.CancelledError)
    #         - 成功   → 由 scheduler._resolve_sequence_future() set_result(seq)
    future: asyncio.Future
    priority: int = 0


# ── RequestQueue ──────────────────────────────────────────────────────────────


class RequestQueue:
    """Structured request queue with timeout enforcement and cancellation.

    Parameters
    ----------
    maxsize
        Hard cap on the number of waiting requests.  Attempting to enqueue
        beyond this limit raises QueueFullError immediately.
    request_timeout_ms
        Maximum time (ms) a request may wait before entering prefill.
        Checked lazily on every dequeue() call via expire_timed_out().
    """

    def __init__(
        self,
        maxsize: int,
        request_timeout_ms: float = 30_000.0,
    ) -> None:
        self.maxsize = maxsize
        self.request_timeout_ms = request_timeout_ms

        # Internal storage — list preserves insertion order for FIFO
        self._queue: List[QueuedRequest] = []
        self._lock: asyncio.Lock = asyncio.Lock()

        # Counters
        self._total_enqueued: int = 0
        self._total_expired: int = 0
        self._total_admitted: int = 0   # incremented by scheduler on prefill entry

    # ── Enqueue ───────────────────────────────────────────────────────────────

    # [LEARN] 为什么 enqueue 是 async？先排除一个常见误解：**它不是为了性能**。
    #         实测 async + lock 的净开销 ≈ 418 ns（裸 list.append 32 ns），
    #         而一次 decode forward ≈ 24 ms —— 占比 2e-5，测都测不出来。
    #
    #         它存在的理由是「约定」：整个类都用 async with self._lock，等于在
    #         代码里声明「本对象只允许在事件循环里访问」。
    #
    # [GOTCHA] 而且要注意：enqueue / dequeue / expire_timed_out / cancel **四个方法
    #         的临界区里一个 await 都没有** —— 所有 await 都在 async with self._lock
    #         *之前*。单事件循环下，acquire→release 之间没有挂起点，其他协程根本没
    #         机会插进来，所以这把锁其实**永远不会真正竞争**。它是「访问边界的声明」，
    #         不是「互斥实现」。
    #
    #         真正的价值是防未来 bug：一旦有人往临界区里加一个 await（写盘、上报
    #         指标、换成 Redis/磁盘队列），当前写法自动正确，而同步写法会静默出错
    #         （被交错执行）。另外也别指望把它换成 threading.Lock 就能保护什么 ——
    #         队列从未被 to_thread 访问过，跨线程场景需要的是完全不同的设计。
    async def enqueue(
        self,
        sequence: Sequence,
        priority: int = 0,
    ) -> asyncio.Future:
        """Add *sequence* to the waiting queue and return its asyncio.Future.

        Parameters
        ----------
        sequence
            A Sequence in state ``"waiting"``.
        priority
            Lower = higher priority.  Stored but not used for ordering in
            Phase 3 (FIFO preserved).

        Returns
        -------
        asyncio.Future
            Resolved with TimeoutError on expiry or CancelledError on cancel.
            The scheduler resolves it with the finished Sequence on success.

        Raises
        ------
        QueueFullError
            If ``len(_queue) >= maxsize`` at the time of enqueue.
        """
        # Reclaim stale capacity before rejecting a new request.
        # [WHY] 先清理超时请求再判满，避免"其实已经过期但还占着名额"导致误报 QueueFull。
        await self.expire_timed_out()

        async with self._lock:
            if len(self._queue) >= self.maxsize:
                raise QueueFullError(self.maxsize)

            loop = asyncio.get_event_loop()
            fut: asyncio.Future = loop.create_future()
            # [GOTCHA] 这里只是创建 future，真正 set_result 的是调度器；
            #         如果请求永远没被调度，就会由 expire_timed_out / cancel 来收尾。

            item = QueuedRequest(
                sequence=sequence,
                enqueue_time=time.perf_counter(),
                future=fut,
                priority=priority,
            )
            self._queue.append(item)
            self._total_enqueued += 1

        return fut

    # ── Dequeue ───────────────────────────────────────────────────────────────

    async def dequeue(self) -> Optional[QueuedRequest]:
        """Admit the next waiting request into the prefill stage.

        Calls expire_timed_out() first to remove stale entries, then returns
        the first remaining QueuedRequest (FIFO) and removes it from the queue.

        Returns None if the queue is empty after expiry scanning.

        Note: _total_admitted is NOT incremented here — the scheduler records
        admission after it moves the dequeued sequence into the active batch.
        """
        # Expire stale entries before considering admission
        await self.expire_timed_out()

        async with self._lock:
            if not self._queue:
                return None
            # [LEARN] pop(0) = 从队头取，保证 FIFO（先来先服务）。
            # [TRACE] 注意：这里不增加 _total_admitted，真正的 admit 记账在
            #         scheduler._schedule() 调用 mark_admitted() 时发生。
            item = self._queue.pop(0)
            return item

    def mark_admitted(self) -> None:
        """Record that one dequeued request entered the scheduler."""
        self._total_admitted += 1

    # ── Expiry ────────────────────────────────────────────────────────────────

    async def expire_timed_out(self) -> int:
        """Scan the queue and expire all requests that have exceeded the timeout.

        For each expired QueuedRequest:
        - Sets ``sequence.state = "expired"``
        - Resolves the future with asyncio.TimeoutError
        - Increments _total_expired
        - Removes the item from _queue

        Returns
        -------
        int
            Number of requests expired in this call.
        """
        now = time.perf_counter()
        expired_count = 0
        still_waiting: List[QueuedRequest] = []

        async with self._lock:
            for item in self._queue:
                elapsed_ms = (now - item.enqueue_time) * 1000.0
                if elapsed_ms > self.request_timeout_ms:
                    item.sequence.state = "expired"
                    if not item.future.done():
                        item.future.set_exception(
                            asyncio.TimeoutError(
                                f"Request {item.sequence.seq_id} timed out "
                                f"after {elapsed_ms:.1f} ms"
                            )
                        )
                    self._total_expired += 1
                    expired_count += 1
                else:
                    still_waiting.append(item)

            # [LEARN] 惰性过期（lazy expiry）：只在 enqueue / dequeue 时扫描，
            #         不额外起定时器。优点是简单、无后台任务；缺点是超时精度
            #         取决于调度器调用这些方法的频率。
            self._queue = still_waiting

        return expired_count

    # ── Cancellation ─────────────────────────────────────────────────────────

    async def cancel(self, seq_id: str) -> bool:
        """Cancel a waiting request by seq_id.

        If the request is found in the queue and not already in a terminal
        state, sets its state to ``"cancelled"``, resolves the future with
        asyncio.CancelledError, removes it from the queue, and returns True.

        Returns False if no matching request is found or it has already been
        removed (e.g. already admitted or expired).
        """
        async with self._lock:
            for i, item in enumerate(self._queue):
                if item.sequence.seq_id == seq_id:
                    # Only cancel if still in a cancellable state
                    # [GOTCHA] 已经 decoding/finished 的序列不在这里取消：它们已经
                    #         在调度器里跑了，取消要由调度器路径负责，否则状态会打架。
                    if item.sequence.state not in ("expired", "cancelled",
                                                   "decoding", "finished"):
                        item.sequence.state = "cancelled"
                        if not item.future.done():
                            item.future.set_exception(
                                asyncio.CancelledError(
                                    f"Request {seq_id} cancelled"
                                )
                            )
                    del self._queue[i]
                    return True

        return False

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a snapshot of queue statistics.

        Returns
        -------
        dict with keys:
            queue_depth      Current number of waiting requests.
            total_enqueued   Cumulative requests ever added to the queue.
            total_expired    Cumulative requests removed by timeout.
            total_admitted   Cumulative requests that entered prefill
                             (incremented by the scheduler, not by dequeue).
            maxsize          The hard capacity cap.
            oldest_wait_ms   Age (ms) of the front-of-queue item;
                             0.0 if the queue is empty.
        """
        now = time.perf_counter()
        oldest_wait_ms = 0.0
        if self._queue:
            oldest_wait_ms = (now - self._queue[0].enqueue_time) * 1000.0

        return {
            "queue_depth": len(self._queue),
            "total_enqueued": self._total_enqueued,
            "total_expired": self._total_expired,
            "total_admitted": self._total_admitted,
            "maxsize": self.maxsize,
            "oldest_wait_ms": oldest_wait_ms,
        }

    # ── Sizing ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the current number of waiting requests."""
        return len(self._queue)
