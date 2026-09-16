# PageServe 学习路线图

这是给「学习用途的 fork」准备的阅读顺序。引擎按 11 个 phase 逐步搭起来，
建议**按顺序读**，并配合调试器观察每一层的行为。

VSCode 环境与断点位置见 [`0002-debugging.md`](0002-debugging.md)。
代码里发现的问题/缺口记录在 [`0003-code-review-findings.md`](0003-code-review-findings.md)。

> 📖 配套博客草稿：[`0005-blog-prefill-decode.md`](0005-blog-prefill-decode.md) —— 用一次
> `generate()` 串讲 tokenizer / embedding / Transformer 前向 / KV cache / 采样 / 梯度。
> 公式渲染与预览方式见该文文末。
>
> 📊 解释性图表约定见 [`0004-figures.md`](0004-figures.md)（`scripts/plot_*.py` → `docs/figures/`）。
>
> 🔁 Phase 2 调度器一次请求的生命周期（含时序图/阶段图）见 [`0006-scheduler-lifecycle.md`](0006-scheduler-lifecycle.md)。
>
> 🎓 开始学 Phase 2？看这份上手指南：[`0007-phase2-guide.md`](0007-phase2-guide.md)（含实验清单）。

---

## 核心链路阅读顺序（已加中文学习注释）

> ⚠️ **这是「阅读顺序」，不是「启动顺序」。**
> 下表中的文件都是**库模块**（只有 `class` / `def`，没有 `__main__`），直接
> `python xxx.py` 不会做任何事。本项目真正的可执行入口只有一个：
> `scripts/serve.py`（即 VSCode 里的 `🌐 Serve: Phase 1/2` 配置）。
>
> 正确的「边读边看」方式：**F5 起一个服务 → 在这些文件里打断点 → 另开终端发
> `/generate` 请求**，然后 `Step Into` 顺着调用链跟下去。只想聚焦单个组件时，
> 调试上表对应的 `tests/test_*.py` 即可（用 `🧪 Pytest: current file`）。
> 详见 [`0002-debugging.md`](0002-debugging.md)。

按下面这个顺序读，一条请求的完整生命周期就串起来了。这些文件已用
`# [LEARN] / [WHY] / [TRACE] / [GOTCHA]` 标签加了中文注释：

| # | 文件 | 一句话职责 | 重点看什么 |
| --- | --- | --- | --- |
| 1 | `engine/sequence.py` | 请求状态载体 | `state` 状态机、`prefill_offset`、`past_key_values` 归属 |
| 2 | `engine/request_queue.py` | 排队 / 超时 / 背压 | FIFO、`future` 生命周期、惰性过期、`QueueFullError → 503` |
| 3 | `engine/scheduler.py` | 连续批处理调度器（核心） | `add_request` → `run_loop` → `_schedule` 四步：swap-in / admit / prefill / decode |
| 4 | `engine/block_allocator.py` | 逻辑块账本 | 逻辑块 ≠ 物理块、`write_token` 自动扩块、`OutOfBlocksError` 触发抢占 |
| 5 | `engine/paged_kv_cache.py` | KV 物理仓库 | `[num_blocks, block_size, layers, kv_heads, head_dim]`、逻辑→物理映射 |
| 6 | `engine/cpu_swap_manager.py` | GPU⇄CPU 换入换出 | `swap_out` / `swap_in` 三步顺序、`CPUSwapError` 处理 |

依赖关系（自底向上）：`sequence` ← `request_queue` ← `scheduler`；
`scheduler` 同时持有 `block_allocator` → `paged_kv_cache` → `cpu_swap_manager`。

读第 3 个文件（scheduler）时，建议配合 [`0002-debugging.md`](0002-debugging.md) 的断点地图，
按 `_schedule` 的 Step 0→3 逐步单步，观察 `self.running` 的变化效果最好。

### 组件 → 对应测试（想单独调试某个组件时用）

| 文件 | 调试这个测试（`🧪 Pytest: current file`） |
| --- | --- |
| `engine/sequence.py` | `tests/test_scheduler.py`（由 scheduler 带入） |
| `engine/request_queue.py` | `tests/test_request_queue.py` |
| `engine/scheduler.py` | `tests/test_scheduler.py` |
| `engine/block_allocator.py` | `tests/test_block_allocator.py` |
| `engine/paged_kv_cache.py` | `tests/test_paged_kv_cache.py` |
| `engine/cpu_swap_manager.py` | `tests/test_cpu_swap_manager.py` |
| `engine/attention_wrapper.py` | `tests/test_attention_wrapper.py` |
| `engine/metrics_aggregator.py` | `tests/test_metrics_aggregator.py` |
| `engine/stage_tracker.py` | `tests/test_stage_tracker.py` |
| `engine/kv_cache_tracker.py` | `tests/test_kv_cache.py` |
| `engine/sequential.py` | `tests/test_sequential.py` |

### 周边文件（同样已加中文注释）

| 文件 | 职责 | 属于 |
| --- | --- | --- |
| `config.py` | 全局配置单一来源 + 环境变量覆盖 | 基础 |
| `models/loader.py` | 设备相关（cuda/mps/cpu）的模型加载 | Phase 1 |
| `engine/sequential.py` | Phase 1 顺序推理基线（prefill/decode/generate） | Phase 1 |
| `metrics/collector.py` | 有界的结果存储 + 分位数统计 + 写 JSON | Phase 1 |
| `engine/request_queue.py` | 请求队列 | Phase 3 |
| `engine/stage_tracker.py` | prefill / decode 分阶段计时 | Phase 4 |
| `engine/kv_cache_config.py` | 解析模型结构，估算每 token KV 占用 | Phase 5 |
| `engine/kv_cache_tracker.py` | 运行时 KV 内存计量（解析估算） | Phase 5 |
| `engine/paged_kv_cache.py` | paged KV 物理池 | Phase 7 |
| `engine/attention_wrapper.py` | 从 paged pool 重建 HF `past_key_values` | Phase 8 |
| `engine/cpu_swap_manager.py` | CPU staging pool | Phase 9 |
| `engine/metrics_aggregator.py` | 统一指标汇总（/metrics 的数据源） | Phase 10 |
| `server/app.py` / `server/app_v2.py` | Phase 1 / Phase 2 HTTP 服务层 | 服务 |
| `load_test/profiles.py` / `runner.py` / `report.py` | 压测流量生成 / 执行 / 报告 | Phase 11 |

> 测试用例（`inference_engine/tests/`）按约定**不加**学习注释，当作可执行的规格说明来读。

---

## 注释约定

在源码里追加学习笔记时统一用这几个标签，方便 `rg` 检索：

| 标签 | 含义 | 例子 |
| --- | --- | --- |
| `# [LEARN]` | 解释概念 / 算法 | `# [LEARN] block size 16 摊薄映射开销` |
| `# [WHY]` | 设计取舍的理由 | `# [WHY] prefill 分块，避免把 decode 饿死` |
| `# [TRACE]` | 标注请求调用链中的一步 | `# [TRACE] step 2: 分块 prefill` |
| `# [GOTCHA]` | 容易踩的坑 / 易错点 | `# [GOTCHA] 阻塞函数不能直接 await` |
| `# [TODO-READ]` | 待深入的点 | `# [TODO-READ] 对比 vLLM 的 PagedAttention` |

检索方式：

```bash
rg -n "# \[(LEARN|WHY|TRACE|GOTCHA|TODO-READ)\]" inference_engine load_test
```

---

## 先读哪个：两种执行模式

| | Phase 1 — `engine/sequential.py` + `server/app.py` | Phase 2+ — `engine/scheduler.py` + `server/app_v2.py` |
| --- | --- | --- |
| 并发 | 一次一个请求（`asyncio.Lock` + 单 worker 线程池） | iteration-level continuous batching |
| KV cache | 每条序列一份完整 HF `past_key_values` | paged blocks + 可选 CPU swap |
| 启动入口 | `scripts/serve.py --phase 1`（`:8000`） | `scripts/serve.py --phase 2`（`:8001`） |
| 定位 | 性能对比基线 | 真正要学的引擎 |

建议先把 `sequential.py::generate` 从头到尾读一遍，再进入 scheduler。

---

## 按 Phase 阅读顺序

| Phase | 文件 | 要理解的概念 |
| --- | --- | --- |
| 1. 顺序基线 | `engine/sequential.py`、`server/app.py`、`models/loader.py` | `past_key_values` 的 prefill/decode、串行为何是瓶颈、TTFT 与 per-token latency |
| 2. 连续批处理 | `engine/scheduler.py`（`add_request`、`run_loop`、`_schedule`） | iteration-level 调度、用 `future` 表达请求生命周期 |
| 3. 请求队列 | `engine/request_queue.py` | FIFO 公平性、队列满（`429/503`）、请求超时 |
| 4. Prefill/Decode 分离 | `engine/scheduler.py`（`_prefill_chunk_blocking`、`_decode_step_single`）、`engine/stage_tracker.py`、`engine/prefill_utils.py` | compute-bound 的 prefill vs bandwidth-bound 的 decode、chunked prefill、budget |
| 5. KV 内存跟踪 | `engine/kv_cache_tracker.py`、`engine/kv_cache_config.py` | 估算每 token KV 字节数、内存压力 |
| 6. 块分配器 | `engine/block_allocator.py`、`engine/sequence.py` | 16-token 逻辑块、碎片、LRU 淘汰 |
| 7. Paged KV cache | `engine/paged_kv_cache.py` | 逻辑块 → 物理块的映射、`write_kv` / `read_kv_sequence` |
| 8. 批注意力接入 | `engine/attention_wrapper.py` | 把 paged cache 接进 forward pass |
| 9. CPU swap pool | `engine/cpu_swap_manager.py`、`scheduler._try_swap_out_victim` | 用抢占代替 OOM、swap out/in |
| 10. 指标 | `engine/metrics_aggregator.py`、`metrics/collector.py` | throughput、P50/P95/P99、SLO 合规率 |
| 11. 压测 | `load_test/`、`run_load_test.py` | constant / ramp / burst 流量、报告 |

测试与 phase 一一对应：`inference_engine/tests/test_*.py` 是每个组件紧凑的
可执行规格。想单步某个组件时，用 **🧪 Pytest: current file** 配置调试它。

---

## 术语表

- **Prefill** — 对整段 prompt 的首次 forward，产出 KV cache 和第一个 token。计算密集（compute-bound）。
- **Decode** — 每步只生成一个 token，复用 KV cache。显存带宽密集（memory-bandwidth bound）。
- **TTFT** — Time To First Token，首 token 时延；主要由排队等待 + prefill 决定。
- **Continuous batching** — 每个 iteration 都可准入/淘汰序列，而不是按请求整体调度；完成的序列立刻释放名额。
- **Paged KV cache** — 把 KV 内存切成固定大小的 block，通过 block table 映射（类似操作系统虚拟内存），消除外部碎片。
- **Preemption / swap-out** — 内存压力下把某序列的 block 拷到 CPU 并释放设备 block，之后再 swap in。
- **Chunked prefill** — 把长 prompt 拆到多个 scheduler step 处理，避免阻塞其他序列的 decode。
- **GQA / MQA** — Grouped / Multi Query Attention：KV head 数少于 attention head 数，显著降低 KV 内存。
- **Block table** — 逻辑块号到物理块号的映射表（本项目中由 `BlockAllocator` 担任）。

与异步/并发相关的术语（详细展开见 [`0012-async-and-blocking.md`](0012-async-and-blocking.md)）：

- **GIL** — **G**lobal **I**nterpreter **L**ock，全局解释器锁。CPython 里同一进程内
  同一时刻只有一个线程能执行 Python 字节码；C 扩展、内核调用、I/O 可以主动释放它。
  两个推论：纯 Python 的 CPU 活多线程**不会变快**；反过来，**释放了 GIL 的工作
  才能真并行**。
- **事件循环（event loop）** — 协程的调度器。uvicorn 默认是**一个进程一个线程跑一个循环**，
  所有 HTTP 请求和后台任务共享它；所以它被阻塞 = 全体停摆。
- **`await`** — 挂起当前协程、**把事件循环交出去**。它**不等于「不等待」**：
  调用方照样在等，但循环可以去跑别人。判据：`async def ⟺ 需要挂起点`
  （`await` / `async with` / `async for`）。
- **`asyncio.to_thread` / `run_in_executor`** — 把阻塞函数派发到线程池，
  空出事件循环。对**释放 GIL** 的工作才是真并行；对纯 Python CPU 活只有响应性收益。
- **`_blocking` 后缀** — 本仓库的命名契约：带此后缀的方法是阻塞的同步函数，
  只能由 async 调用方用 `to_thread` 派发（例：`_prefill_chunk_blocking`）。

---

## 下一步练习

1. 在 `scheduler._schedule` 打断点，观察两个 prompt 长度不同的并发请求，`self.running` 如何随 step 变化。
2. 设 `KV_NUM_BLOCKS=6`，发两个请求，观察 `_try_swap_out_victim` 抢占、以及 swap 统计。
3. 读 `baseline_metrics.json` / `continuous_batching_metrics.json`，把数字对应回 `metrics_aggregator.py`。
4. 对比 `block_allocator.py` 与 vLLM 的 `BlockSpaceManager`：prefix sharing、copy-on-write、watermark 有何不同？
5. 结合 [`0003-code-review-findings.md`](0003-code-review-findings.md)，挑一个「设计局限」想想怎么改。
