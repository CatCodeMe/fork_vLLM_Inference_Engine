# 简版 ↔ 真实版对照手册：本仓库 → vLLM / llm-d

这份文档是**从本项目走向生产级引擎的桥**。它做三件事：

1. 把本仓库的每个概念**对照到真实 vLLM / llm-d 的对应物**（含模块路径）；
2. 用**同一个框架**做一次真实量级的**容量估算**（10M DAU / 500K 峰值并发）；
3. 明确列出**本仓库简化/缺失了什么**，作为读真代码时的检查清单。

> 前面 15 篇文档是"读懂这个仓库"；这一篇是"**带着已有的心智模型去读真系统**"。
> 相关：[`0003`](0003-code-review-findings.md)（F1~F34 完整缺口清单）、
> [`0014`](0014-prefill-and-chunking.md) §8.3（生产环境的 OOM 与限流）、
> [`0015`](0015-kv-cache-lifecycle.md)（KV 的一生）。

---

## 0. 一页对照表

| 概念 | 本仓库 | 真实 vLLM（v1 架构，路径以你的版本为准） | llm-d / K8s 生态 |
| --- | --- | --- | --- |
| 服务入口 | `server/app_v2.py`（FastAPI） | `vllm/entrypoints/openai/`（OpenAI 兼容 API） | — |
| 引擎主循环 | `scheduler.run_loop()`（asyncio task） | `vllm/v1/engine/core.py` → `EngineCore`（**独立进程**） | — |
| 调度器 | `_schedule()` 的 Step 0~4 | `vllm/v1/core/sched/scheduler.py` → `Scheduler.schedule()` | — |
| 等待队列 | `RequestQueue`（有界 32 → 503） | **无界 `deque`**（默认不拒绝） | **网关侧有界队列**（EPP flow control） |
| 准入 | `len(running) < running_limit` | 按 KV 块是否够 + `max_num_seqs` | 网关按 token/租户配额 |
| KV 账本 | `engine/block_allocator.py` | `vllm/v1/core/kv_cache_manager.py` | — |
| KV 物理池 | `engine/paged_kv_cache.py` | `vllm/v1/core/block_pool.py` | — |
| 池大小 | **写死 `kv_num_blocks=256`** | **启动时按 `gpu_memory_utilization` 从显存推算** | — |
| 注意力 | **重建 HF `past_key_values` + 普通 attention** | **PagedAttention kernel 直接按块读** | FlashInfer / FlashAttention 后端 |
| 抢占 | `_try_swap_out_victim`（swap 到 CPU，**只重试一次**） | `_preempt_request`（默认 **RECOMPUTE**，FCFS 抢最后一个） | — |
| 模型执行 | `_prefill_chunk_blocking` / `_decode_step_single`（**逐序列** forward） | `vllm/v1/worker/gpu_model_runner.py`（**真 tensor 级 batching**） | — |
| 指标 | `engine/metrics_aggregator.py` + `/metrics`、`/trace`、`/spans` | Prometheus 指标 + 日志 | OTel / 集中式 tracing |
| 部署形态 | 单进程单卡 | 单进程多卡（TP/PP）+ 多副本 | PD 分离 + 智能路由 |

---

## 1. 两条主流程的对照

### 1.1 一个请求的生命周期

```
本仓库                                        vLLM
─────────────────────────────────────────    ─────────────────────────────────────────
POST /generate                               POST /v1/chat/completions
  app_v2.endpoint_generate                     entrypoints/openai/api_server.py
  scheduler.add_request                        AsyncLLM.add_request
    分词（线程池）                               tokenize（前端进程）
    Sequence.create()                           EngineCoreRequest
    RequestQueue.enqueue() → future             input_queue.put()（ZMQ/共享内存跨进程）
  await future                                 await   ← 同样用 future/异步等待
                                                 ↓（跨进程）
  后台 run_loop 任务                             EngineCore.run_busy_loop()
    _schedule() Step 0~4                          Scheduler.schedule() → 组 batch
      Step 2 prefill（逐序列，to_thread）           model_runner.execute_model()（真批处理）
      Step 3 decode（逐序列）                    → 输出 output_queue
    _resolve_sequence_future() → set_result      → 前端唤醒，流式返回
```

**关键差异**：vLLM 的 EngineCore 在**独立进程**里 —— 这正好跨过了 [`0012`](0012-async-and-blocking.md)
讲的 GIL 问题：引擎进程是纯同步的紧循环，不用跟 HTTP 事件循环抢 GIL。

### 1.2 一次调度迭代

| 步骤 | 本仓库 | vLLM |
| --- | --- | --- |
| swap-in / 恢复 | Step 0（`swapped_out` 循环） | `Scheduler` 里统一处理 preempted 请求 |
| 准入 | Step 1：`dequeue()` + stamp chunk size | `schedule()` 里按 KV 块 + `max_num_seqs` 决定 |
| prefill | Step 2：**每个序列一个 chunk**，`prefill_budget_tokens` 限总量 | `num_new_tokens` 按 `max_num_batched_tokens` 分配，**chunked prefill 是同一套机制** |
| decode | Step 3：**每个序列 1 token，串行 to_thread** | 同一个 batch 里 prefill + decode **一起前向**（tensor 级） |
| 淘汰 | Step 4：`_resolve_sequence_future` | 输出 + 释放块 + 通知前端 |

> **本仓库最核心的简化**：Step 2/3 是"逐序列 forward 轮转"，vLLM 是"**一次 forward 带 B 个序列**"。
> 这是 [`0005`](0005-blog-prefill-decode.md) §8.4 讲的"批处理是最大杀器"的落地差别，
> 也是 **F21**。

---

## 2. 逐项深挖：简化 vs 真实

### 2.1 KV 管理与注意力（最大的架构差异）

| | 本仓库 | vLLM |
| --- | --- | --- |
| KV 存哪 | paged pool（**存储**）+ live HF cache（**计算**）→ **两份** | 只有 paged blocks（**存储即计算**） |
| 计算时怎么取 | `reconstruct_*` 把块拼成 HF `past_key_values` | kernel 直接按 block table 索引 |
| 需要重建吗 | **需要**（chunked prefill 每 chunk 一次） | 不需要 |
| `block_size` 受什么约束 | **无**（纯记账参数） | **被 kernel 钉死**（kernel 按块取） |

详见 [`0015`](0015-kv-cache-lifecycle.md) §1.3 / §5。这正是
[`0009`](0009-attention-landscape.md) 里"怎么算"和"怎么存"两个正交维度的具体体现。

### 2.2 抢占

| | 本仓库 | vLLM |
| --- | --- | --- |
| 默认策略 | **SWAP 到 CPU**（唯一实现） | **RECOMPUTE**（swap 也有，但默认不用） |
| 为什么 | Phase 9 的教学主题 | swap 有"大量小 CPU↔GPU 传输"开销；且主机侧要备同量级内存（见 §3.4） |
| 选谁当受害者 | **largest-first**（占块最多） | **FCFS：抢最后到达的** |
| 重试几次 | **1 次**，失败 → `oom` → 503 | 循环抢占直到够 |
| 请求会失败吗 | **会**（`finish_reason="oom"`） | **不会**（降级为变慢） |

详见 [`0014`](0014-prefill-and-chunking.md) §8.2.0/§8.2.1，**F31**。

### 2.3 队列与限流

| | 本仓库 | vLLM | llm-d / K8s |
| --- | --- | --- | --- |
| 队列 | `RequestQueue`，`maxsize=32` → **503** | **无界**，默认不拒绝 | 网关侧**有界队列** + 优先级 |
| 过载表现 | 503（拒绝） | TTFT 无限增长 | 429 / 排队 / 按租户降级 |
| 限流在哪 | **没有**（F22） | 引擎内**也不做** | **网关做 token-aware 准入** |

详见 [`0014`](0014-prefill-and-chunking.md) §8.3。

### 2.4 指标口径

| 指标 | 本仓库 | 注意 |
| --- | --- | --- |
| `ttft_ms` | `first_token − prefill_start` → **不含排队**（**F34**） | 真实 TTFT = `queue_wait_time_ms + ttft_ms` |
| `kv_cache` vs `paged_kv_cache` | 两套口径：**解析估算** vs **块账本** | 约束用后者（**F29** 那条 O(n) 扫描也在这里） |

---

## 3. 容量估算（摘要）

> 📖 **完整版见 [`0017-capacity-planning.md`](0017-capacity-planning.md)** ——
> 那一篇是按"可以拿去做选型"的完整度写的（含方法、假设清单、敏感性分析、调参启示）。
> 本节只留结论，方便和 §2 的架构差异对着看。

以 **32B dense + 平均 32K 上下文**为基准（10M DAU / 500K 峰值在途 / 35K req/s）：

| 项 | 数量级 |
| --- | --- |
| 每会话 KV | **8 GB**（32K × 256 KB/token） |
| 单 8×H100 节点能同时服务 | **67 个**会话 |
| **GPU 显存总量** | **≈ 4,400 TB**（7,000 节点 × 640 GB） |
| ├ 权重池 | ≈ 444 TB |
| └ **KV 预算** | **≈ 3,660 TB**（≈ 468K 个 32K 会话） |
| **主机内存**（排队） | **≈ 20 GB** |
| **主机内存**（若开 swap） | **≈ 3,660 TB** ← 和 KV 池同量级 |

**三个和本仓库直接相关的判断**：

1. **池大小是容量规划问题，不是工程细节** —— 本仓库把 `kv_num_blocks` 写死成 256（≈48 MB），
   而真实系统必须从显存推算（`gpu_memory_utilization`）。**F 系列里没有这一条，因为它是"配置方式"而不是 bug。**
2. **swap 抢占在容量上不可行** —— 开 swap 等于要求主机侧备一份 KV（PB 级）。
   这是 vLLM 默认 RECOMPUTE 的**容量理由**（延迟理由见 [`0014`](0014-prefill-and-chunking.md) §8.2.0）。
3. **排队/准入不是优化项** —— 500K 在途里能"在算"的只有一小部分，
   所以 §2.3 说的"限流在网关"是**唯一可行的工作方式**。

---

## 4. 本仓库简化/缺失了什么（读真代码时的检查清单）

| # | 本仓库 | vLLM 的做法 | 相关 |
| --- | --- | --- | --- |
| 1 | 逐序列 forward（**无 tensor 级 batching**） | 一次 forward 带 B 个序列 | **F21** |
| 2 | 没有 PagedAttention kernel（重建 HF cache） | kernel 直接按块算 | §2.1，**F26 相关** |
| 3 | 池大小写死 256 块 | 启动时按 `gpu_memory_utilization` 推算 | §3.4 |
| 4 | 抢占只重试一次，且会 `oom` 失败 | 循环抢占（RECOMPUTE），不失败 | **F31** |
| 5 | 只实现 SWAP | 默认 RECOMPUTE | §2.2 |
| 6 | 有界队列 → 503 | 无界队列（限流在网关） | §2.3，**F22** |
| 7 | `ttft_ms` 不含排队 | 指标口径明确 | **F34** |
| 8 | 无流式输出（SSE） | 流式 + abort | **F19** |
| 9 | 断线不取消已 admit 的请求 | abort 集合 + step 边界丢弃 | **F20** |
| 10 | 无 `stop` 序列 / 上下文窗口检查 | 都有 | **F27 / F28** |
| 11 | 单卡单进程 | TP / PP / 多副本 / **PD 分离** | — |
| 12 | 无 prefix caching（`copy_blocks` 写了但没调用） | 有（自动前缀复用） | **F 系列** |
| 13 | 只做 greedy argmax | temperature / top-k / top-p / 惩罚 | — |

> 完整 34 条见 [`0003`](0003-code-review-findings.md)。

### 4.1 本仓库**完全没有**、但你读 vLLM 一定会遇到的 5 个主题

这几个不是"本仓库实现得简单"，而是**压根没有对应物**。它们的共同点是
**不改变调度器骨架**，所以这个仓库没有它们也能跑通 —— 但生产系统一个都少不了：

| 主题 | 一句话 | 为什么重要 | 在 vLLM 哪里 |
| --- | --- | --- | --- |
| **投机解码**（speculative decoding） | 用一个小"草稿模型"一次猜 K 个 token，大模型并行验证，接受多少个算多少个 | 解码是 memory-bound（[`0005`](0005-blog-prefill-decode.md) §8），**一次 forward 验证 K 个 token ≈ 免费**，端到端常快 2~3× | `vllm/v1/spec_decode/`、`--speculative-model` |
| **约束解码 / 结构化输出** | 每步把不合法的 token 的 logits 掩掉（JSON schema、正则、grammar） | **Codex 这类产品的刚需**（工具调用要吐合法 JSON）；也顺手解决 [`0011`](0011-stopping-and-cancellation.md) 提到的"停止序列"一类问题 | `vllm/v1/structured_output/`、`--guided-decoding-backend` |
| **LoRA / adapter** | 一份基座权重上挂多个小 adapter，按请求切换 | 多租户/多客户场景的成本关键 —— 不用为每个客户复制一份全量权重 | `vllm/lora/`、`--enable-lora` |
| **多模态** | 图像/音频/视频 token 与文本 token 拼在一起 | 不改变调度器骨架，但**改变"一个 token 多少钱"**：一张图的 token 数可能抵几千个文本 token，直接影响 §3 的容量估算 | `vllm/multimodal/`、`--limit-mm-per-prompt` |
| **embedding / rerank 模型** | 只做一次 prefill、不要 decode 循环 | 服务形态完全不同（无 KV 增长、无逐 token 循环），但常和生成模型**共用同一套部署/路由** | `--task embed`、`PoolingModelRunner` |

> **它们和本仓库的关系**：
>
> - **投机解码** 改的是"一步生成几个 token"→ 对应本仓库 `_decode_step_single` 的位置（如果要做，就得在 Step 3 里加"验证草稿"的逻辑）；
> - **约束解码** 改的是"argmax 之后/之前怎么选 token"→ 本仓库只有 `logits[:, -1, :].argmax()` 一行（[`0011`](0011-stopping-and-cancellation.md) §1 讨论的 `finish_reason` 就在同一层）；
> - **LoRA / 多模态** 改的是"forward 吃什么"→ 本仓库的 `self.model(input_ids=..., past_key_values=...)` 那个调用点；
> - **embedding** 改的是"要不要 decode"→ 本仓库的整个 Step 3 都不需要。
>
> 也就是说：**这四个方向都能挂到你已经理解的那几个函数上**，不需要推翻调度器的心智模型。

### 4.2 展开讲两个最实用的：约束解码 + 投机解码

§4.1 的表格是一行一个，这两个值得展开 —— 它们是**生产系统里最常被问到、也最容易在本仓库上验证的**。

#### ⑴ 约束解码 / 结构化输出（constrained decoding）

> 📖 **完整原理见 [`0018-constrained-decoding.md`](0018-constrained-decoding.md)**
> —— 那一篇用 ~150 行可运行代码回答了"正则怎么可能作用在神经网络上"。

**一句话**：每步选 token 之前，把"会导致输出不合法"的 token 的 logits 设成 `-inf`，
**让模型在物理上无法输出非法内容**。

**具体例子**：让模型做工具调用，要求符合 schema：

```json
{"city": string, "unit": "celsius" | "fahrenheit"}
```

| | 结果 |
| --- | --- |
| **不加约束** | 可能吐 `{"city": "Beijing",}`（**多一个逗号 → JSON 非法**），或干脆不按格式；你只能事后校验 + 重试（多花几百 ms 和 token） |
| **加约束** | 模型写到 `{"city": "Beijing",` 时，"接下来输出 `}`" 这个选择**被 mask 掉了**（grammar 当前状态要求"逗号后必须跟新 key"）→ 只能走合法路径 → **结构上保证可解析** |

**机制 —— 就一行插入点**：

```
step t:
  logits[152K]                      ← 模型输出，全词表分布
    ↓ 用 grammar 的当前状态算出 allowed（152K 位的 bitmask）
  logits[~allowed] = -inf           ← ★ 约束就发生在这一行
    ↓
  argmax / sample                   ← 正常采样
    ↓
  grammar 状态机前进一步
```

**难点是 tokenization 边界**：`{"name": "` 这种片段可能跨好几个 token，
而一个 token 里也可能含跨界字符。所以状态机建在**字节/字符**上（正则 DFA 或 CFG），
并为每个状态**预计算词表级 mask 并缓存**（否则每步重算太慢）。
`outlines` / `xgrammar` / `llguidance` 做的就是这件事（C++/Rust + 缓存）。

**和微调的根本区别**：

| | 微调 | 约束解码 |
| --- | --- | --- |
| 效果 | **倾向**输出 JSON（概率上） | **不可能**输出非法内容（硬保证） |
| 代价 | 训练成本 | 每步一次 mask（缓存后 ~µs） |

**挂在本仓库哪里**：选 token 只有两处，约束就是在这两行的 `argmax` **之前**插一个 mask：

```python
first_token_id = int(logits[:, -1, :].argmax(dim=-1).item())            # prefill 最后一个 chunk
next_token_id  = int(outputs.logits[:, -1, :].argmax(dim=-1).item())    # decode 每步
```

**不用改调度器、KV、批处理** —— 这也是为什么它值得先学。

> 顺带区分：**`stop` 序列管"什么时候停"**（[`0011`](0011-stopping-and-cancellation.md)），
> **约束解码管"允许输出什么"**。两者互补。

#### ⑵ 投机解码（speculative decoding）

**先看它解决什么**：decode 每步只生成 1 个 token，而这一步要把**全部权重读一遍**
（32B fp16 ≈ 65 GB）—— 算术强度 ≈ 1，**纯 memory-bound**（[`0005`](0005-blog-prefill-decode.md) §8.3）。
换句话说：**算力大量闲置，瓶颈在"把权重搬进计算单元"**。

**想法**：既然一趟搬运能算 B 个 token 的 batch，那**让一个小模型先猜 K 个 token**，
再让大模型**一次 forward 验证这 K 个**，接受前缀匹配的部分 ——
**验证 K 个 token 的成本 ≈ 生成 1 个**（因为瓶颈是权重读取，不是 token 数）。

```
草稿模型（小）  →  猜 t1 t2 t3 t4 t5
大模型（一次 forward，输入 t1..t5）→ 得到 5 个位置的分布
逐个比对：t1 ✅ t2 ✅ t3 ❌ → 接受前 3 个，用大模型在第 3 位的结果换掉 t4，从 t4 重新开始
```

**关键指标是 acceptance rate**（接受率）：接受 3 个就意味着**一次 forward 顶 3 步**。
端到端常见快 **2~3×**。

**代价**：多一份草稿模型权重 + 实现复杂度；接受率低时反而更慢。
所以有"自适应"版本（根据历史接受率动态调 K）。

**挂在本仓库哪里**：`_decode_step_single` ——
现在的逻辑是"喂 1 个 token、取最后一位、argmax"；投机解码要把它改成
"喂 K 个候选、取 K 个位置的 logits、逐个验证"。**KV 写入逻辑也要跟着改成一次写 K 个 token。**

> 两项的共同点：**都不动调度器骨架**，而是改"一步生成什么 / 一步允许什么"。
> 这正是你已经有心智模型之后，读 vLLM 时最该先抓的两个点。

---

## 5. 建议的读码顺序（带着本仓库的心智模型去读）

| 顺序 | 本仓库里你已经懂的 | 去 vLLM 找什么 |
| --- | --- | --- |
| 1 | `run_loop` / `_schedule` | `vllm/v1/engine/core.py` 的 `EngineCore.run_busy_loop()` —— **看它怎么把"同步紧循环"和"跨进程通信"分开**（这就解决了 0012 的 GIL 问题） |
| 2 | `_schedule` 的 Step 1~4 | `vllm/v1/core/sched/scheduler.py` 的 `schedule()` —— **看它怎么把 prefill 和 decode 塞进同一个 batch**（F21 的答案） |
| 3 | `BlockAllocator` + `PagedKVCacheManager` | `kv_cache_manager.py` + `block_pool.py` —— **看它怎么从显存推算池大小**（§3.4 的答案） |
| 4 | `_prefill_chunk_blocking` / `_decode_step_single` | `gpu_model_runner.py` —— **看 tensor 级 batching 和 CUDA Graph** |
| 5 | 你**没有**的东西 | `vllm/v1/attention/backends/` —— **PagedAttention kernel**：这里能看到"pool 参与计算"长什么样（§2.1 的答案） |
| 6 | `_try_swap_out_victim` | `_preempt_request` —— 看 RECOMPUTE 怎么实现、FCFS 怎么选受害者 |
| 7 | 无 | **llm-d / K8s Gateway API Inference Extension** 的 flow control —— 看"网关怎么只放被批准的负载进来"（§3 的答案） |

---

## 6. 一句话总结每个主题

| 主题 | 本仓库（简版） | 真实系统（生产版） |
| --- | --- | --- |
| 批处理 | 逐序列轮转 | 一次 forward 带 B 个序列 |
| KV | 两份（pool + live cache），计算靠重建 | 一份（paged blocks），计算靠 kernel 直接寻址 |
| 池大小 | 写死常量 | 从显存推算 |
| 内存压力 | 抢占 → 失败（`oom`） | 抢占 → 变慢（不失败） |
| 过载 | 拒绝（503） | 引擎排队 + **网关限流** |
| 规模 | 单卡单进程 | 多卡多副本 + PD 分离 |
