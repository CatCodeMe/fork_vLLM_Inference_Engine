# Phase 2 学习指南：从「顺序服务」到「连续批处理」

Phase 1 你已经在 `engine/sequential.py` 里看懂了「一次 `generate()` 做了什么」。
Phase 2 要研究的是：**怎么让很多请求同时高效地跑在一张卡上**。核心是调度器
`engine/scheduler.py`。

> 配套：[`0006-scheduler-lifecycle.md`](0006-scheduler-lifecycle.md)（时序图/阶段图）、
> [`0005-blog-prefill-decode.md`](0005-blog-prefill-decode.md)（Transformer/KV cache 原理）、
> [`0002-debugging.md`](0002-debugging.md)（断点）。

---

## 0. 先跑起来（5 分钟）

```bash
# 终端 1：启动 Phase 2（:8001）。或在 VSCode F5 -> "Serve: Phase 2 continuous batching"
uv run python scripts/serve.py --phase 2

# 终端 2：并发发 8 个请求，观察批处理
uv run python scripts/demo_phase2.py --n 8 --max-new-tokens 24 --plot
# 跑完会额外拉 GET /trace，打一份带 N/8 + [step NNN] 标号的调度轨迹摘要
```

实测输出（本机 MPS）：

```
[health] model=Qwen/Qwen2-0.5B device=mps max_batch_size=4
[结果]  成功 8/8   总墙钟 9.64s   聚合吞吐 19.9 tokens/s
        TTFT min/median/max: 59 / 70 / 851 ms
[调度器] 峰值 running=4
        running 序列: ▂█████████████████████▆▂█████████████████████▆
```

**关键观察**：`max_batch_size=4`，所以 8 个请求分成**两波**、每波 4 个并发；
曲线先升到 4、保持、下降，再升到 4。这就是连续批处理——一个序列完成立刻腾位置，
而不是「8 个排队串行」。

---

## 1. 心智模型的转变

| | Phase 1 `sequential.py` | Phase 2 `scheduler.py` |
| --- | --- | --- |
| 谁在跑 | HTTP handler 直接调 `generate()` | 后台 `run_loop` 任务 |
| 并发单位 | 整个请求（跑完才换） | scheduler **iteration**（每步 1 token） |
| 请求怎么等 | `asyncio.Lock` 排队 | 入队拿 `future`，挂起等结果 |
| prefill | 一次算完整段 prompt | 分块（chunk），受预算限制 |
| KV cache | HF `past_key_values` 随手放 | 写进 paged pool，可换出到 CPU |

一句话：**Phase 1 的 `decode()` 内部 `for` 循环，在 Phase 2 里被「拆成一步」，
由 `_schedule()` 每轮推进一次。** 于是每轮结束都能插入新请求、淘汰完成的请求。

---

## 2. 六个研究阶段（建议按序）

### 2.1 调度器骨架 —— `scheduler.py`

- 读：`add_request()`、`run_loop()`、`_schedule()`（四个 Step）。
- 实验：跑 `demo_phase2.py`；在 `_schedule` 加 **Log Message 断点** `running={len(self.running)}`。
- 现象：`running` 在 0→4 之间变化；队列里多余的请求等下一波。
- 关键问题：为什么 Step 1「只登记」、Step 2 才前向？（答见下条。）

### 2.2 请求队列与背压 —— `request_queue.py`

- 读：`enqueue()` / `dequeue()` / `expire_timed_out()` / `cancel()`。
- 实验 A：一股脑发 **> max_batch_size×8 = 32** 个请求，看部分返回 **503**（`QueueFullError`）。
- 实验 B：`REQUEST_TIMEOUT_MS=2000` 重启，发大量慢请求，看部分返回 **504**（排队超时）。
- 关键问题：`future` 由谁创建、由谁解析？（`enqueue` 创建，`_resolve_sequence_future` 解析。）

### 2.3 prefill / decode 分离 + 分块 prefill

- 读：`_prefill_chunk_blocking()`、`_decode_step_single()`、`config.prefill_budget_tokens`、`prefill_chunk_size`。
- 实验：发一个很长的 prompt（几百 token），同时在别处发短请求；
  看 `/metrics` 的 `stage_breakdown.prefill.budget_utilization`。
- 现象：长 prompt 被拆成多个 chunk（默认 128 token/chunk），短请求的 TTFT 不会被它拖死。
- 关键问题：`prefill_budget_tokens` 和 `prefill_chunk_size` 的区别？（前者是每步总预算，后者是单序列每步上限。）

### 2.4 块分配器 —— `block_allocator.py`

- 读：`allocate()` / `write_token()` / `free()` / `OutOfBlocksError`。
- 实验：跑几个请求后看 `/metrics → paged_kv_cache.block_allocator`
  （`free_blocks` / `used_blocks` / `utilization` / `per_sequence`）。
- 现象：每个序列按 `ceil(tokens/16)` 个块增长；块池耗尽会抛 `OutOfBlocksError`。
- 关键问题：逻辑块号和物理块号分别存在哪？（账本在 allocator，数据在 paged pool。）

### 2.5 Paged KV cache —— `paged_kv_cache.py`

- 读：`write_kv()` 的映射公式 `token_pos // block_size` 和 `% block_size`；`read_kv_sequence()`。
- 实验：看启动日志 `[PagedKVCache] Pool size: 48.0 MB (256 blocks × 16 tokens)`，
  以及 `/metrics → paged_kv_cache`。
- **重要认知**：正常 decode 的**热路径并不读** paged pool（用的是挂在序列上的 live
  `past_key_values`），pool 只在 swap-in / 分块续 chunk 时被读。原因见
  `_decode_step_single` 的 docstring（MPS 上重建太慢）。

### 2.6 CPU swap（抢占）—— `cpu_swap_manager.py`

- 读：`swap_out()` / `swap_in()`；`scheduler._try_swap_out_victim()`。
- 实验：把块池调小，制造内存压力：
  ```bash
  KV_NUM_BLOCKS=6 KV_NUM_CPU_BLOCKS=32 uv run python scripts/serve.py --phase 2
  # 另开终端，用两个中等长度请求（每个约 4 块）并发
  uv run python scripts/demo_phase2.py --n 2 --max-new-tokens 10
  ```
- 现象：`/metrics → cpu_swap` 出现 `total_swap_outs ≥ 1`、`total_swap_ins ≥ 1`，
  且 `system.requests_oom_total == 0`（用延迟换可用性，而不是报错）。
- 关键问题：为什么抢占选「块最多的」而不是「最旧的」？

---

## 3. 实验清单速查

| # | 目的 | 命令 | 预期 |
| --- | --- | --- | --- |
| E1 | 连续批处理 | `demo_phase2.py --n 8` | 峰值 running=4，分两波 |
| E2 | 对比 Phase 1 | 起两个服务，跑 `run_validation_v2.py` | 出现 speedup |
| E3 | 背压 | `demo_phase2.py --n 40` | 部分 503 |
| E4 | 排队超时 | `REQUEST_TIMEOUT_MS=2000` + E3 | 部分 504 |
| E5 | 分块 prefill 的取舍 | `uv run python scripts/demo_chunk_sizes.py`（一条命令对比 512/128/32；**必须一长一短才测得出来**） | 短请求真实 TTFT 512→32 快 12.5×，长请求慢 2.9× |
| E6 | 块耗尽 | `KV_NUM_BLOCKS=2` | `finish_reason=oom` / 503 |
| E7 | CPU swap | `KV_NUM_BLOCKS=6` + 2 请求 | `cpu_swap.total_swap_outs≥1` |

> 改环境变量后要**重启服务**才生效（`Config` 在启动时读一次）。

---

## 4. `/metrics` 怎么读

```bash
curl -s http://127.0.0.1:8001/metrics | python3 -m json.tool
```

| 字段 | 含义 |
| --- | --- |
| `system.requests_in_flight` | 当前 `running` + `swapped` 的序列数 |
| `system.requests_waiting` | 队列深度 |
| `system.throughput_tokens_per_sec` | 滚动窗口吞吐（注意：分母固定为 60s，启动初期偏低） |
| `system.requests_oom_total` | 被判定 OOM 的请求数 |
| `system.requests_swapped_total` | 累计换出次数 |
| `e2e_latency.ttft_ms` | TTFT 的 p50/p95/p99 |
| `stage_breakdown.prefill/decode` | 分阶段时延与预算利用率 |
| `kv_cache` | 解析估算的 KV 内存（解析口径） |
| `paged_kv_cache.block_allocator` | 块池真实占用（账本口径） |
| `cpu_swap` | 换入换出统计 |
| `scheduler.batch_size_over_time` | `(t, running_size)` 采样序列 |

---

## 5. 断点

见 [`0002-debugging.md`](0002-debugging.md) 第 4 节，以及 [`0006-scheduler-lifecycle.md`](0006-scheduler-lifecycle.md)
的「建议断点」表。

---

## 6. 已知局限（读代码时留意）

见 [`0003-code-review-findings.md`](0003-code-review-findings.md)，其中与 Phase 2 最相关的是：

- **F1**：prefill 一次性分配整段 prompt 的块 → 比整个池还长的 prompt 直接 503。
- **F3**：decode 热路径不读 paged pool（设计取舍，非 bug）。
- **F4**：抢占只考虑 `decoding` 状态的序列。
- **F6/F7**：`stage_tracker` 与吞吐统计的口径问题。
