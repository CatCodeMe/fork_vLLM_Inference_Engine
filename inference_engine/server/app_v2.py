"""
server/app_v2.py — FastAPI server for Phase 2 continuous-batching inference.

Differences from Phase 1 (app.py)
----------------------------------
* No inference_lock, no single-request serialisation.
* On startup a ContinuousBatchingScheduler is created and its run_loop() is
  started as a background asyncio Task.
* POST /generate submits a request and awaits its lifecycle future.
* GET /metrics exposes scheduler-level telemetry (batch_size_over_time,
  scheduler_step_latency_ms, per-sequence stats) in addition to the per-
  request summary statistics from Phase 1.
* GET /health includes current_batch_size and queue_depth.
* Runs on port 8001 so Phase 1 (port 8000) and Phase 2 can run side-by-side
  for direct comparison.

Completion signalling
---------------------
The /generate handler awaits the lifecycle future created by RequestQueue.
Successful generation resolves it with the finished Sequence; queue expiry
resolves it with TimeoutError.

Endpoints
---------
POST /generate      Submit prompt; blocks until generation finishes.
GET  /metrics       Scheduler telemetry + per-sequence summary stats.
GET  /health        Liveness check with current batch occupancy.
"""

# [LEARN] Phase 2 服务器：只负责"收请求/等结果/返回 JSON"，调度全交给后台的
#         ContinuousBatchingScheduler。它自己不再有 inference_lock。
# [TRACE] 生命周期：lifespan 启动时加载模型 → 建 scheduler → scheduler.start()
#         起后台 run_loop；关闭时 scheduler.stop() 优雅停止。

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference_engine.config import Config
from inference_engine.engine.request_queue import QueueFullError
from inference_engine.engine.kv_cache_config import format_kv_cache_report
from inference_engine.engine.scheduler import ContinuousBatchingScheduler
from inference_engine.engine.sequence import Sequence
from inference_engine.models.loader import LoadedModel, load_model_and_tokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Module-level singletons (populated during lifespan startup) ───────────────

_config: Optional[Config] = None
_loaded_model: Optional[LoadedModel] = None
_scheduler: Optional[ContinuousBatchingScheduler] = None

# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model, create scheduler, start background loop.
    Shutdown: stop scheduler gracefully.
    """
    # [LEARN] lifespan 不是“死声明”，而是 FastAPI 注册、由 uvicorn 在特定时刻
    #         调用的生命周期钩子。底层是 ASGI 的 Lifespan 协议：
    #           uvicorn 启动 → 向 app 发 lifespan.startup → Starlette 进入这个
    #           async 上下文管理器、执行到 yield 之前 → 回 lifespan.startup.complete
    #           → uvicorn 才开始接受 HTTP 连接（日志 "Application startup complete."）
    #           收到退出信号 → 发 lifespan.shutdown → 执行 yield 之后 → 退出。
    #         所以模型只加载一次，且发生在“能收请求”之前。
    # [GOTCHA] 多进程（uvicorn --workers N）时，每个 worker 各跑一次 lifespan，
    #         即每个进程都会加载一份模型。
    global _config, _loaded_model, _scheduler

    _config = Config()
    logger.info(
        "Phase 2 server starting: model=%s device=%s max_batch_size=%d",
        _config.model_name,
        _config.device,
        _config.max_batch_size,
    )

    # [LEARN] load_model_and_tokenizer 是“阻塞”的同步函数（网络/磁盘 IO + CPU 计算）。
    #         直接调用会卡住整个事件循环。run_in_executor 把它丢到 worker 线程执行，
    #         事件循环保持可响应。
    # [GOTCHA] await 仍然在“等结果”，所以启动流程依旧是顺序的——这里换来的不是
    #         并行，而是“不阻塞事件循环”。启动阶段影响不大，但在请求处理路径上
    #         是必须的（见 app.py 的 _run_generate）。
    # [LEARN] Python 3.9+ 更简洁的等价写法：await asyncio.to_thread(fn, ...)。
    # [GOTCHA] get_event_loop() 在 3.10+ 已废弃（见 0003-code-review-findings 的 F10），
    #         更推荐 asyncio.get_running_loop()。
    loop = asyncio.get_event_loop()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as tmp_exec:
        _loaded_model = await loop.run_in_executor(
            tmp_exec, load_model_and_tokenizer, _config
        )
    logger.info("Model loaded successfully on device=%s", _loaded_model.device)

    _scheduler = ContinuousBatchingScheduler(
        model=_loaded_model.model,
        tokenizer=_loaded_model.tokenizer,
        config=_config,
    )
    print(format_kv_cache_report(_scheduler.kv_cache_config))
    # [LEARN] start() 只是 create_task(run_loop())，不阻塞启动；真正调度在后台跑。
    _scheduler.start()   # creates asyncio.Task for run_loop()
    logger.info("Scheduler started (max_batch_size=%d)", _config.max_batch_size)

    yield  # ── server is running ────────────────────────────────────────────

    logger.info("Phase 2 server shutting down …")
    await _scheduler.stop()
    logger.info("Scheduler stopped. %d sequences finished.", _scheduler.total_finished)


# ── App ───────────────────────────────────────────────────────────────────────

# [LEARN] 创建 FastAPI 应用；lifespan=lifespan 接上生命周期函数（同上）。
app = FastAPI(
    title="Continuous Batching LLM Inference Server",
    description=(
        "Phase 2: iteration-level continuous batching scheduler. "
        "Multiple requests are batched dynamically; no rewrite of Phase 1."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────


# [LEARN] Pydantic 请求模型（同 app.py）：le=512 就是参数校验的来源。
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_new_tokens: int = Field(
        default=50, ge=1, le=512, description="Maximum tokens to generate"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sequence_to_result_dict(seq: Sequence, device: str) -> dict:
    """Convert a finished Sequence into a GenerationResult-compatible dict."""
    from inference_engine.engine.sequential import get_memory_stats

    generated_text = seq.prompt  # start with prompt — Phase 1 decode() does same
    if seq.generated_token_ids:
        generated_text = _scheduler.tokenizer.decode(  # type: ignore[union-attr]
            seq.generated_token_ids, skip_special_tokens=True
        )

    finish_time = seq.finish_time or time.perf_counter()
    total_latency_ms = (finish_time - seq.arrival_time) * 1000.0
    n_gen = len(seq.generated_token_ids)
    tps = (n_gen / total_latency_ms * 1000.0) if total_latency_ms > 0 else 0.0

    allocated_mb, reserved_mb = get_memory_stats(device)

    return {
        "seq_id": seq.seq_id,
        "prompt": seq.prompt,
        "generated_text": generated_text,
        "prompt_tokens": len(seq.prompt_token_ids),
        "generated_tokens": n_gen,
        "ttft_ms": seq.ttft_ms,
        "total_latency_ms": total_latency_ms,
        "tokens_per_second": tps,
        "per_token_latencies_ms": seq.per_token_latencies_ms,
        "gpu_memory_allocated_mb": allocated_mb,
        "gpu_memory_reserved_mb": reserved_mb,
        "finish_reason": seq.finish_reason,
        "queue_wait_time_ms": seq.queue_wait_time_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


# [LEARN] 路由装饰器（同 app.py）。这个 handler 的模式与 Phase 1 不同：
#         它把请求丢给 scheduler 排队，然后 await 一个 future，
#         直到后台调度器把该序列跑完再返回——所以它是“异步等待”而非“阻塞生成”。
@app.post("/generate", response_class=JSONResponse)
async def endpoint_generate(request: GenerateRequest):
    """Submit *prompt* to the continuous batching scheduler.

    The request lifecycle future resolves on completion and raises when a
    queued request expires, so terminal states cannot leave the handler stuck.
    """
    if _scheduler is None or _loaded_model is None:
        raise HTTPException(status_code=503, detail="Scheduler not ready")

    # [TRACE] ⓪ 请求到达 HTTP 层（后面 ①~⑥ 都在 scheduler 里，⑦ 回到这里）
    t_recv = time.perf_counter()
    _scheduler.record_trace(
        "1/8 收到",
        f"POST /generate prompt={len(request.prompt)} 字符 max_new_tokens={request.max_new_tokens}",
        req=True,
    )

    try:
        seq, future = await _scheduler.add_request(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
        )
    except QueueFullError:
        raise HTTPException(
            status_code=503,
            detail="Server at capacity, retry later",
        )
    except Exception as exc:
        logger.exception("Failed to enqueue request: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to enqueue request") from exc

    try:
        seq = await future
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Request timed out in queue") from exc
    except asyncio.CancelledError:
        # [GOTCHA] 客户端断开时只能取消"还在排队"的请求；已经 admit 进 running 的
        #         序列，cancel() 返回 False，生成会继续跑完（README 列为后续改进点）。
        await _scheduler.request_queue.cancel(seq.seq_id)
        raise

    if seq.finish_reason == "oom":
        raise HTTPException(status_code=503, detail="Insufficient KV-cache capacity")
    if seq.finish_reason == "error":
        logger.error("Generation failed for seq_id=%s: %s", seq.seq_id, seq.error_message)
        raise HTTPException(status_code=500, detail="Generation failed")

    # [TRACE] ⑦ 终点：future 已解析，组装 JSON 返回
    body = _sequence_to_result_dict(seq, _loaded_model.device)
    _scheduler.record_trace(
        "8/8 返回",
        f"seq={seq.seq_id[:8]} tokens={body['generated_tokens']} "
        f"finish_reason={body['finish_reason']} ttft={body['ttft_ms']:.1f}ms "
        f"端到端={(time.perf_counter() - t_recv) * 1000.0:.1f}ms",
        seq_id=seq.seq_id,
        req=True,
    )
    return JSONResponse(content=body)


@app.get("/trace", response_class=JSONResponse)
async def endpoint_trace(limit: int = 200):
    """返回最近的调度轨迹（环形缓冲区），供 demo 脚本/调试使用。

    带 step 标号的 `[Snnn]` 行来自 scheduler step，`[REQ]` 行来自 HTTP/入队层。
    标号含义（1/8 ~ 8/8）见 `ContinuousBatchingScheduler.record_trace` 的注释。
    """
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not ready")
    return JSONResponse(content=_scheduler.get_trace(limit=limit))


@app.get("/spans", response_class=JSONResponse)
async def endpoint_spans(limit: int = 50, max_children: int = 64):
    """返回最近请求的 **span 树**（OTel 风格：有父子、有 start/duration）。

    与 `GET /trace` 的区别：
      · `/trace`  = **事件日志**（按时间顺序一行一条，适合 grep 排查）
      · `/spans`  = **span 树**（适合画瀑布图 / 导入 Jaeger 类工具）

    子树展开：request → queue_wait / prefill(+chunk#i) / decode(+token#i)。
    时间坐标是相对于该请求 arrival 的 ms。只需两个参数：
      · limit        返回多少个请求（按总时长降序）
      · max_children 每阶段最多展开多少子 span（防 token 多时 JSON 爆炸）
    """
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not ready")
    return JSONResponse(content=_scheduler.get_spans(limit=limit, max_children=max_children))


@app.get("/metrics", response_class=JSONResponse)
async def endpoint_metrics():
    """Return unified scheduler and system metrics via MetricsAggregator.

    Response structure (Phase 10)
    ------------------------------
    {
        "system":          SystemSnapshot (requests_in_flight, throughput, ...),
        "e2e_latency":     {"ttft_ms": {p50/p95/p99}, "total_latency_ms": {...}},
        "slo_compliance":  {"ttft_compliance_pct": float, ...},
        "stage_breakdown": prefill/decode stage telemetry,
        "kv_cache":        KV cache tracker stats,
        "paged_kv_cache":  paged pool stats,
        "cpu_swap":        CPU swap manager stats,
        "queue_stats":     request queue stats,
    }
    """
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not ready")

    return JSONResponse(content=_scheduler.get_metrics())



@app.get("/health", response_class=JSONResponse)
async def endpoint_health():
    """Liveness check with current scheduler occupancy."""
    if _loaded_model is None or _scheduler is None or _config is None:
        return JSONResponse(status_code=503, content={"status": "loading"})

    return JSONResponse(content={
        "status": "ok",
        "model": _config.model_name,
        "device": _loaded_model.device,
        "max_batch_size": _config.max_batch_size,
        "current_batch_size": len(_scheduler.running),
        "queue_depth": _scheduler.request_queue.stats()["queue_depth"],
        "sequences_finished": _scheduler.total_finished,
    })
