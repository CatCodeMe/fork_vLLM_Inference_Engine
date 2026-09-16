"""
server/app.py — FastAPI server for sequential LLM inference.

Sequential constraint — enforced at two layers
----------------------------------------------
1. asyncio.Lock (inference_lock): makes the "one request at a time" rule
   explicit and visible at the application layer.  If a second request arrives
   while inference is running, it waits on the lock before entering the
   executor — it does NOT return a 503.

2. ThreadPoolExecutor(max_workers=1): even if the lock were somehow bypassed,
   the executor can only run one blocking call at a time.

Running inference in an executor means the event loop is never blocked, so
health-checks and metrics endpoints remain responsive during long generations.

Endpoints
---------
POST /generate      Run inference; returns GenerationResult as JSON.
GET  /metrics       Returns last N results + summary statistics.
GET  /health        Returns model name, device, and status.
"""

# [LEARN] Phase 1 服务器：用两道锁强制"一次只服务一个请求"，作为性能对比基线。
#   1) asyncio.Lock  —— 应用层串行（后续请求在锁上排队，不会 503）
#   2) 单 worker 线程池 —— 物理层串行（即使锁被绕过也只能跑一个）
# [WHY] 推理放到线程池 run_in_executor，事件循环不被独占，/health 仍可响应。

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference_engine.config import Config
from inference_engine.engine.sequential import GenerationResult, generate
from inference_engine.metrics.collector import MetricsCollector
from inference_engine.models.loader import LoadedModel, load_model_and_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Module-level singletons (populated during lifespan startup) ───────────────

# [LEARN] Python/FastAPI 惯用法速读（下面几处都会用到）：
#   @asynccontextmanager   
#       Python 标准库 contextlib 的装饰器。把一个 async 生成器函数
#       （内部用 yield）“包装”成可以用 `async with` 的东西。
#   yield                   
#       生成器里的“分界点”：yield 之前的代码 = 进入时执行（启动），
#       yield 之后的代码 = 退出时执行（关闭）。FastAPI 用这个做 lifespan。
#   @app.post("/generate")  
#       装饰器（decorator）：把下面的 async 函数注册成路由，
#       收到 POST /generate 时就调它。@app.get 同理。
#   BaseModel + Field       
#       Pydantic 的数据模型：自动解析 JSON、校验类型/范围，
#       校验失败就返回 422（你之前那个 max_new_tokens ≤ 512 就是这里管）。
#   async / await           
#       “协程”：await 时把这个函数的控制权交回事件循环，
#       让服务器在等 IO/线程池时仍能处理其它请求。

_config: Optional[Config] = None
_loaded_model: Optional[LoadedModel] = None
_collector: Optional[MetricsCollector] = None

# The executor intentionally has max_workers=1 — this is the physical
# enforcement of sequential serving (independent of the asyncio lock).
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

# Explicit sequential lock.  async with inference_lock serialises all
# /generate calls at the application layer.
inference_lock = asyncio.Lock()


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load model + tokenizer, initialise metrics collector.
    Shutdown: flush metrics to disk.
    """
    # [LEARN] 这是 FastAPI 推荐的“生命周期”写法（lifespan）：
    #         uvicorn 启动服务前会先跑 yield 之前的代码，关闭时再跑 yield 之后的。
    #         因为“加载模型”很慢，放在这里只做一次，不占用每个请求。
    # [GOTCHA] 这里没有用 `async with` 手动调用——是 FastAPI 在下面
    #         `FastAPI(lifespan=lifespan)` 时替你管理进入/退出。
    global _config, _loaded_model, _collector

    _config = Config()
    logger.info("Starting server: model=%s device=%s", _config.model_name, _config.device)

    # Model loading is blocking — run in executor so startup doesn't block
    # the event loop (though in practice uvicorn handles this fine at startup).
    loop = asyncio.get_event_loop()
    _loaded_model = await loop.run_in_executor(
        _executor, load_model_and_tokenizer, _config
    )
    logger.info("Model loaded successfully")

    _collector = MetricsCollector(history_size=_config.metrics_history_size)

    yield  # ── server is running ────────────────────────────────────────────

    # Shutdown: persist metrics
    # [GOTCHA] 关闭时会把结果写到 metrics_output_path（默认 baseline_metrics.json），
    #         会覆盖仓库里已提交的基准文件；调试时用 METRICS_OUTPUT_PATH 改路径。
    logger.info("Shutting down — writing metrics to %s", _config.metrics_output_path)
    _collector.dump_to_json(_config.metrics_output_path)
    _executor.shutdown(wait=False)


# ── App ───────────────────────────────────────────────────────────────────────

# [LEARN] 创建 FastAPI 应用对象。lifespan=lifespan 就是把上面的生命周期函数接上；
#         title/description/version 会显示在自动生成的 /docs 页面上。
app = FastAPI(
    title="Sequential LLM Inference Server",
    description=(
        "Phase 1 baseline: intentionally naive sequential serving. "
        "One request at a time, no batching, no KV cache management."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────


# [LEARN] Pydantic 请求模型：客户端 POST 的 JSON 会先被解析成这个对象。
#         Field(ge=1, le=512) 就是之前 422 报错的来源（不在仓库里搜“错误文案”，
#         要搜“约束” le=512）。校验通过后，函数里就能直接点号访问 request.prompt。
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_new_tokens: int = Field(
        default=50, ge=1, le=512, description="Maximum tokens to generate"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _result_to_dict(result: GenerationResult) -> dict:
    """Convert GenerationResult dataclass to a JSON-serialisable dict."""
    return dataclasses.asdict(result)


def _run_generate(prompt: str, max_new_tokens: int) -> GenerationResult:
    """
    Blocking wrapper that calls generate().

    This is the function submitted to the ThreadPoolExecutor so it runs
    on a worker thread, keeping the event loop free.
    """
    assert _loaded_model is not None, "Model not loaded"
    assert _config is not None, "Config not initialised"

    return generate(
        model=_loaded_model.model,
        tokenizer=_loaded_model.tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=_loaded_model.device,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


# [LEARN] 路由装饰器：把这个 async 函数绑定到 POST /generate。
#         FastAPI 会自动：解析/校验请求体 → 调函数 → 把返回值序列化成 JSON。
#         下面所有 @app.get(...) 端点是同一机制。
@app.post("/generate", response_class=JSONResponse)
async def endpoint_generate(request: GenerateRequest):
    """
    Run sequential inference on *prompt*.

    Requests queue behind the inference_lock — only one runs at a time.
    This is intentional: we are measuring the sequential baseline.
    """
    if _loaded_model is None or _collector is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    # [TRACE] 请求在这里拿锁 → 提交线程池跑 generate() → 释放锁；
    #         同一时刻只有一个请求能进入临界区，这就是"顺序服务"的根源。
    async with inference_lock:
        loop = asyncio.get_event_loop()
        try:
            result: GenerationResult = await loop.run_in_executor(
                _executor,
                _run_generate,
                request.prompt,
                request.max_new_tokens,
            )
        except Exception as exc:
            logger.exception("Inference error: %s", exc)
            raise HTTPException(status_code=500, detail="Generation failed") from exc

    _collector.append(result)

    return JSONResponse(content=_result_to_dict(result))


@app.get("/metrics", response_class=JSONResponse)
async def endpoint_metrics():
    """
    Return all stored GenerationResult objects plus summary statistics.

    Summary includes p50 / p95 / p99 for TTFT and total latency.
    """
    if _collector is None:
        raise HTTPException(status_code=503, detail="Collector not ready")

    results = _collector.get_all()
    summary = _collector.compute_summary()

    return JSONResponse(
        content={
            "summary": summary,
            "results": [_result_to_dict(r) for r in results],
        }
    )


@app.get("/health", response_class=JSONResponse)
async def endpoint_health():
    """
    Lightweight health-check.  Returns 200 when the model is loaded.
    """
    if _loaded_model is None or _config is None:
        return JSONResponse(status_code=503, content={"status": "loading"})

    return JSONResponse(
        content={
            "status": "ok",
            "model": _config.model_name,
            "device": _loaded_model.device,
            "requests_served": len(_collector) if _collector else 0,
        }
    )
