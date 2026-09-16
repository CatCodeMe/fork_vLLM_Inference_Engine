"""Thread-safe runtime accounting for per-sequence KV-cache growth."""

# [LEARN] Phase 5 的"内存计量器"：不碰 torch，只用 token 数 × 每 token 字节数
#         估算 KV 内存，给 /metrics 提供 memory_pressure 等指标。
# [GOTCHA] 这是解析估算，不代表 paged pool 的真实占用；两者不要混用。

from __future__ import annotations

from threading import Lock

from inference_engine.engine.kv_cache_config import KVCacheConfig


class KVCacheTracker:
    def __init__(
        self, kv_cache_config: KVCacheConfig, max_memory_mb: float
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.max_memory_mb = max_memory_mb
        self._sequence_token_counts: dict[str, int] = {}
        self._peak_memory_mb = 0.0
        self._lock = Lock()

    def register_sequence(self, seq_id: str, prompt_token_count: int) -> None:
        with self._lock:
            if seq_id in self._sequence_token_counts:
                raise ValueError(f"Sequence {seq_id} is already registered")
            self._sequence_token_counts[seq_id] = prompt_token_count
            self._update_peak_locked()

    def update_sequence(self, seq_id: str, total_token_count: int) -> None:
        with self._lock:
            if seq_id not in self._sequence_token_counts:
                raise KeyError(f"Sequence {seq_id} is not registered")
            self._sequence_token_counts[seq_id] = total_token_count
            self._update_peak_locked()

    def unregister_sequence(self, seq_id: str) -> None:
        with self._lock:
            self._sequence_token_counts.pop(seq_id, None)

    def sequence_memory_mb(self, seq_id: str) -> float:
        with self._lock:
            token_count = self._sequence_token_counts.get(seq_id, 0)
        return token_count * self.kv_cache_config.bytes_per_token_mb

    def total_memory_mb(self) -> float:
        with self._lock:
            total = self._total_memory_mb_locked()
            self._peak_memory_mb = max(self._peak_memory_mb, total)
            return total

    def memory_pressure(self) -> float:
        # [LEARN] 压力 = 当前估算占用 / 配置上限，钳制在 [0,1]。
        #         上限是 config.kv_cache_max_memory_mb（默认 1024MB），
        #         仅用于指标展示，并不会触发实际抢占（抢占看的是块池是否耗尽）。
        if self.max_memory_mb <= 0:
            return 0.0
        pressure = self.total_memory_mb() / self.max_memory_mb
        return min(1.0, max(0.0, pressure))

    def eviction_candidates(self, n: int = 3) -> list[str]:
        # [LEARN] 只是"建议清单"（按 token 数从大到小），供人工/未来策略参考；
        #         真正的抢占选择在 scheduler._try_swap_out_victim()（按块数最大）。
        with self._lock:
            ordered = sorted(
                self._sequence_token_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        return [seq_id for seq_id, _ in ordered[:n]]

    def stats(self) -> dict:
        with self._lock:
            token_counts = dict(self._sequence_token_counts)
            total = self._total_memory_mb_locked()
            self._peak_memory_mb = max(self._peak_memory_mb, total)
            peak = self._peak_memory_mb

        bytes_per_token_mb = self.kv_cache_config.bytes_per_token_mb
        pressure = (
            min(1.0, max(0.0, total / self.max_memory_mb))
            if self.max_memory_mb > 0
            else 0.0
        )
        candidates = sorted(
            token_counts,
            key=token_counts.get,
            reverse=True,
        )[:3]
        return {
            "active_sequences": len(token_counts),
            "total_memory_mb": total,
            "peak_memory_mb": peak,
            "max_memory_mb": self.max_memory_mb,
            "memory_pressure": pressure,
            "eviction_candidates": candidates,
            "per_sequence": {
                seq_id: {
                    "token_count": token_count,
                    "memory_mb": token_count * bytes_per_token_mb,
                }
                for seq_id, token_count in token_counts.items()
            },
        }

    def _total_memory_mb_locked(self) -> float:
        return (
            sum(self._sequence_token_counts.values())
            * self.kv_cache_config.bytes_per_token_mb
        )

    def _update_peak_locked(self) -> None:
        self._peak_memory_mb = max(
            self._peak_memory_mb, self._total_memory_mb_locked()
        )
