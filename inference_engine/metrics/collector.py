"""
metrics/collector.py — Thread-safe storage and summary statistics for GenerationResult objects.

Design
------
* Uses a collections.deque with a fixed maxlen so memory is bounded.
* All mutation goes through a threading.Lock so it is safe to call from a
  thread-pool executor (which is how the server runs inference).
* compute_summary() uses numpy.percentile for p50 / p95 / p99.
* dump_to_json() serialises every result plus the summary to a single JSON file
  that can be loaded by downstream analysis scripts.
"""

# [LEARN] 这是 Phase 1（顺序服务）那套最简单的"指标收集器"：存最近 N 条结果、
#         算分位数、写 JSON。Phase 10 的 MetricsAggregator 会读它做统一报表。
# [WHY] 用 deque(maxlen=N) 而不是 list：内存有上界，旧的自动丢弃。

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import asdict
from typing import List

import numpy as np

from inference_engine.engine.sequential import GenerationResult

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Append-only, bounded store of GenerationResult objects.

    Parameters
    ----------
    history_size : int
        Maximum number of results kept in memory.  Older entries are
        automatically evicted (FIFO) once the limit is reached.
    """

    def __init__(self, history_size: int = 100) -> None:
        self._history_size = history_size
        self._results: deque[GenerationResult] = deque(maxlen=history_size)
        self._lock = threading.Lock()

    # ── Mutation ──────────────────────────────────────────────────────────────

    def append(self, result: GenerationResult) -> None:
        """Add a result.  Thread-safe."""
        with self._lock:
            self._results.append(result)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_all(self) -> List[GenerationResult]:
        """Return a snapshot of all stored results (oldest first)."""
        with self._lock:
            return list(self._results)

    def __len__(self) -> int:
        with self._lock:
            return len(self._results)

    # ── Statistics ────────────────────────────────────────────────────────────

    def compute_summary(self) -> dict:
        """
        Return a dict with p50 / p95 / p99 for TTFT and total latency,
        plus mean tokens-per-second and total request count.

        Returns an empty summary dict if no results are stored yet.
        """
        results = self.get_all()
        if not results:
            return {"count": 0}

        ttft_arr = np.array([r.ttft_ms for r in results], dtype=float)
        latency_arr = np.array([r.total_latency_ms for r in results], dtype=float)
        tps_arr = np.array([r.tokens_per_second for r in results], dtype=float)

        def pcts(arr: np.ndarray) -> dict:
            # [LEARN] P50/P95/P99 用 numpy.percentile 线性插值；
            #         样本很少时（比如只有 1 条）三个分位数都等于该值。
            return {
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        return {
            "count": len(results),
            "ttft_ms": pcts(ttft_arr),
            "total_latency_ms": pcts(latency_arr),
            "tokens_per_second": pcts(tps_arr),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def dump_to_json(self, filepath: str) -> None:
        """
        Write all results and summary statistics to *filepath* as JSON.

        The output structure is:
        {
            "summary": { ... },
            "results": [ { ...GenerationResult fields... }, ... ]
        }

        Called by the FastAPI lifespan shutdown handler so that the file is
        always written even if the server is terminated gracefully.
        """
        # [GOTCHA] 这个方法在 Phase 1 服务器关闭时被调用，默认写到
        #         config.metrics_output_path（baseline_metrics.json），
        #         会覆盖仓库里已提交的基准文件！调试时请改 METRICS_OUTPUT_PATH。
        results = self.get_all()
        summary = self.compute_summary()

        payload = {
            "summary": summary,
            "results": [asdict(r) for r in results],
        }

        try:
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            logger.info("Metrics written to '%s' (%d results)", filepath, len(results))
        except OSError as exc:
            logger.error("Failed to write metrics to '%s': %s", filepath, exc)
            raise
