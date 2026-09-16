# PageServe 代码梳理：问题与缺口

> 目的：**帮助理解这个推理引擎**，把阅读过程中发现的核心局限、可疑点和缺失功能沉淀下来。
> **不是**要重构/升级它，也不是说作者写得差——这是一个教学项目，很多取舍是有意的。
> 每条都标注了证据位置和严重度，方便你自己去验证、判断。

## 本次梳理方法

- 通读 `inference_engine/` 全部非测试代码 + `load_test/`。
- 跑通：`pytest`（98 passed）、Phase 1 / Phase 2 服务、普通生成、611-token 长 prompt
  （chunked prefill）、`KV_NUM_BLOCKS=6` 下的 swap out/in。
- 对可疑点做了针对性实验，证据写在各条目里。

严重度：🔴 明显影响正确性/可用性 · 🟡 设计局限或指标口径 · 🔵 小问题/整洁 · ⚪ 缺失功能

## TL;DR：最值得注意的 3 点

1. **F1**：prefill 会一次性分配「整段 prompt」所需的 block，因此**比整个 KV 池还长的
   prompt 永远无法被服务**——即使引擎号称支持 chunked prefill。这是最容易让人困惑的
   行为（表现为直接 HTTP 503）。
2. **F3**：正常 decode 的**热路径并没有读 paged pool**，用的是挂在序列上的 live
   HF `past_key_values`；paged KV 实际上只在 swap-in / 分块续 chunk 时被读。所以
   「paged attention」在本项目里更像「可换出的 KV 存储」，而不是 vLLM 那种注意力实现。
3. **F6 / F7**：`stage_tracker` 和 `compute_throughput` 的统计口径会让指标被稀释/低估，
   看 `/metrics` 的数字时要心里有数。

---

## A. 核心实现局限

### F1 🔴 Prefill 一次性分配整段 prompt 的 block

- **位置**：`engine/scheduler.py` → `_prefill_chunk_blocking()`（首个 chunk）；
  `engine/scheduler.py` → `_prefill_sequence()`（遗留路径同理）。
- **现象**：首 chunk 里 `blocks_needed = ceil(prompt_len / block_size)`，把**整段 prompt**
  的块一次分配掉，之后 `set_token_count(..., 0)` 再随 chunk 增长。
- **后果**：只要 `prompt_len` 超过整个 block 池容量，请求必然 `OutOfBlocksError` →
  `finish_reason="oom"` → HTTP 503，**没有任何 chunked prefill 的机会**。
- **实测证据**：
  - `KV_NUM_BLOCKS=4`（=64 token 容量）+ 611-token prompt → 3 个并发请求全部
    `{"detail":"Insufficient KV-cache capacity"}`。
  - `KV_NUM_BLOCKS=8`（=128 token）+ ~153-token prompt → 同样全部 503。
- **对照**：真正的 vLLM 是按 chunk 增量分配 block，长 prompt 可以分块塞进小池子。
- **学习点**：chunked prefill 在这里解决的是「公平性/延迟」，不是「内存上限」。

### F2 🟡 Chunked prefill 每块都重建完整 KV → 总量 O(n²)

- **位置**：`engine/scheduler.py` → `_prefill_chunk_blocking()` 非首 chunk 调
  `build_past_key_values()`；`engine/attention_wrapper.py` → `reconstruct_*()` 内部
  `read_kv_sequence()` 会把**此前所有** KV 拼接回来。
- **后果**：处理第 k 个 chunk 时要读回前 k-1 个 chunk 的 KV，长 prompt 的 TTFT 明显偏大。
- **实测证据**：611-token prompt（5 个 chunk）TTFT ≈ **1921 ms**。
- **学习点**：这就是为什么生产引擎要把「缓存管理」和「注意力 kernel」耦合起来（否则
  每次都要物化完整 KV）。

### F3 🔵 正常 decode 热路径不读 paged pool（设计取舍）

- **位置**：`engine/scheduler.py` → `_decode_step_single()`。
- **现象**：`past_kv = seq.past_key_values`（沿用上一轮 forward 返回的 live cache）；
  同时每步仍把新 token 的 KV `write_kv` 进 paged pool，但**不读**。
- **作者理由**（docstring 已写明）：MPS 上「读 pool + permute + 重建 cache」约
  240 ms/token，而一次 forward 只要 ~24 ms，所以热路径保留 live cache。
- **影响**：paged pool 的读取只发生在 swap-in 和分块续 chunk。所谓 paged attention
  并未在常规 decode 中生效。**理解这点很关键**，否则会误以为 paged cache 是每步都用。

### F4 🟡 抢占只考虑 `decoding` 序列

- **位置**：`engine/scheduler.py` → `_try_swap_out_victim()` 的 candidates 过滤条件
  `sequence.state == "decoding"`。
- **后果**：一个正在 `chunked_prefilling`、且已经占了不少块的序列不可被换出；新请求
  可能因此被判 OOM，而池子里其实有可回收的块。
- **学习点**：生产中通常按「可抢占 + 回收收益」排序，而不只看状态。

### F5 🟡 无 prefix caching

- **位置**：`engine/paged_kv_cache.py` → `copy_blocks()` 已实现，但全仓库无生产调用
  （仅测试用到）。
- **后果**：相同前缀的多个请求无法共享 block；每个请求各自占满前缀。
- **学习点**：`Block` 里的 `ref_count` / `is_dirty` 就是为 copy-on-write 型 prefix
  caching 预留的，属于「接口就绪、功能未接」。

---

## B. 潜在正确性 / 边界

### F6 🟡 StageTracker 把「空转 step」也计入平均

- **位置**：`engine/scheduler.py` → `_schedule()` 无条件调用
  `stage_tracker.record_prefill(...)` / `record_decode(...)`，即使该 step 没有任何
  prefill/decode。
- **后果**：`stage_tracker.full_report()` 的 `avg_latency_ms`、`avg_budget_utilization`、
  `avg_batch_size` 被大量 0 记录拉低，只能看趋势，不能当绝对值。

### F7 🟡 吞吐分母用固定窗口长度，而非真实经过时间

- **位置**：`engine/metrics_aggregator.py` → `compute_throughput()`：
  `tokens_per_sec = total_tokens / self.history_window_seconds`（默认 60 s）。
- **后果**：服务器刚启动（窗口未填满）或长时间空闲后，吞吐被系统性低估。
- **学习点**：更稳的做法是除以「窗口内首末记录的实际时间差」。

### F8 🔵 `extract_new_token_kv` 忽略 `token_position`

- **位置**：`engine/attention_wrapper.py` → `extract_new_token_kv()` 固定取 `-1`。
- **影响**：参数具有误导性，调用方以为能指定位置。当前逻辑正确（新 token 总在最后），
  但接口应删掉该参数或真正使用它。

### F9 🔵 `BlockAllocator.evict_*` 非原子

- **位置**：`engine/block_allocator.py` → `evict_lru()` / `evict_largest()`。
- **现象**：先在锁内取候选，释放锁后再逐个 `free()`，并发场景下存在竞态（候选可能已变）。
- **补充**：当前 scheduler 不调用它们，风险低；调用的是 `_try_swap_out_victim`。

### F10 🔵 在已运行的 loop 内用 `asyncio.get_event_loop()`

- **位置**：`engine/request_queue.py` → `enqueue()`；`server/app.py`（lifespan + endpoint）；
  `server/app_v2.py`（lifespan）；`engine/scheduler.py` → 遗留 `_decode_step()`。
- **影响**：Python 3.10+ 已 deprecated，未来版本会报错；应改为 `get_running_loop()`。
  注意 `scheduler.add_request()` 已经用的是正确写法。

### F11 🔵 executor 注释与实现不一致

- **位置**：`engine/scheduler.py` → `__init__` 注释「Single shared executor — one warm
  thread」；但热路径 prefill/decode 实际走 `asyncio.to_thread`（默认 executor），
  `self._executor` 只用于 `add_request` 的分词和未使用的 `_decode_step`。
- **影响**：不是 bug（因为 `_schedule` 内是顺序 `await`，仍串行），但注释会误导读者。

---

## C. 指标口径

### F12 MPS 的 `gpu_memory_reserved_mb` 用进程 RSS

- **位置**：`engine/sequential.py` → `get_memory_stats()`。
- **现象**：MPS 分支的 `reserved` = `psutil.Process().memory_info().rss`，**包含模型权重本身**，
  与 CUDA 的 `memory_reserved()`（PyTorch 显存池）语义不同。
- **后果**：跨设备对比 `gpu_memory_*` 字段没有可比性。

### F13 SLO 默认阈值在 MPS/CPU 上普遍达不到

- **位置**：`engine/metrics_aggregator.py` → `compute_slo_compliance()`（TTFT ≤ 200 ms）。
- **现象**：MPS 上 TTFT 常见 400 ms+，`ttft_compliance_pct` 会长期偏低。
- **结论**：是阈值不匹配硬件，不是计算错误。

---

## D. 测试覆盖缺口

### F14 swap-in 的**数据完整性**未被断言

- **位置**：`inference_engine/tests/test_scheduler.py` →
  `test_swap_out_triggers_on_oom()` 只断言「至少发生过一次 swap，或者两个序列不是都 OOM」。
- **缺口**：换回设备后 KV 是否正确、生成是否连续，没有数值级校验。
  （本次手动验证：`KV_NUM_BLOCKS=6` 下 `total_swap_outs=1, total_swap_ins=1, oom=0`，
  两个请求都正常返回，但这是「能跑通」而非「数值正确」的证明。）

### F15 缺少长 prompt 的 chunked prefill 正确性测试

- 现有测试用的都是短 prompt，多 chunk 的「负索引取新 KV」逻辑
  （`source_position = token_pos_in_chunk - chunk_len`）没有专门的正确性断言。

### F16 三个死代码方法无测试、也无调用

- `engine/scheduler.py` → `_prefill_sequence()`、`_decode_step()`、
  `_decode_one_step_blocking()` 均无生产调用（仅注释提及）。
- 建议阅读时把它们当「历史版本对照」，不要误以为是当前路径。

---

## E. 依赖 / 版本

### F17 `transformers>=4.40.0` 的下限与代码不符

- **证据**：
  - `models/loader.py` 用 `from_pretrained(..., dtype=...)`——这是 transformers 5.x 的参数名；
    4.x 用 `torch_dtype=`。
  - `engine/scheduler.py` → `_extract_kv_layer()` 依赖 `.layers[i].keys` 形式的
    `DynamicCache`（较新版本才有）。
- **后果**：若按声明装 4.40，可能加载失败或静默回退 fp32。
- **实测**：本环境锁定 `transformers==5.16.1`，一切正常。
- **学习点**：这个项目的 requirements 写得偏宽松，涉及 transformers cache API 的代码
  对版本很敏感。

### F18 `matplotlib` 未列入依赖

- **位置**：`run_validation_v2.py` → 绘图函数。
- **行为**：`ImportError` 时打印 `[WARN] ... skipping`，**优雅跳过**，不是报错。
  只是文档里说会生成 `batch_size_over_time.png`，实际没装就看不到。

---

## F. 缺失功能（部分作者已在 README 承认）

| 编号 | 缺失 | 说明 |
| --- | --- | --- |
| F19 | ⚪ 无 streaming | `/generate` 阻塞到整段生成完才返回，没有 SSE / 增量输出。 |
| F20 | ⚪ 断线不取消已 admit 的请求 | `server/app_v2.py` 的 `CancelledError` 只取消「排队中」的请求；已进 running 的会继续跑完。 |
| F21 | ⚪ 无 tensor-level batching | 每个序列单独 forward，只是在 step 上轮转（`scheduler` docstring 明说）。 |
| F22 | ⚪ 无鉴权 / 限流 | 任何客户端都能调用 `/generate`。 |
| F23 | ⚪ `decode_batch_limit` 默认不起作用 | `_schedule()` 里 `running_limit = min(max_batch_size=4, decode_batch_limit=8) = 4`，默认下该参数被 `max_batch_size` 盖住。 |
| F25 | ⚪ 未套用 chat template | `scheduler.add_request()` / `sequential.prefill()` 直接 `tokenizer(prompt)`，不做 system/user 角色包装。默认模型 `Qwen/Qwen2-0.5B` 又是 **base（续写）模型**，所以问它会得到“续写”式的奇怪回答。这是**模型类型 + prompt 格式**问题，不是 bug；想问答需换 `-Instruct` 模型并套 `tokenizer.apply_chat_template`。 |
| F26 | ⚪ `compute_kv_cache_config` 不支持 MLA | `engine/kv_cache_config.py` 按 `bytes = 2·layers·n_kv·head_dim·dtype_bytes` 估算，对 MHA/GQA/MQA 正确，但 **MLA（DeepSeek）缓存的是低秩隐向量**，公式完全不同，用这个模块会高估/算错。若要支持 MLA 模型需单独处理。 |
| F27 | ⚪ 无 `stop` 序列支持 | 只有 EOS 和 `max_new_tokens` 两个停止条件，`GenerateRequest` 不接受 `stop` 字符串/ids。后果：套 chat template 后模型会在 `"<\|im_end\|>"` 之类边界**自己扮演下一轮角色继续编**；多轮对话/Agent 场景不可用。`finish_reason` 也缺 `"stop"` 取值。实现要点与取舍见 [`0011`](0011-stopping-and-cancellation.md) §9。 |
| F28 | ⚪ 无上下文窗口检查 | 引擎里没有任何 `max_position_embeddings` / `position_ids` 检查，`prompt_tokens + max_new_tokens` 可能越过模型位置上限（RoPE 外推或报错）。注意 `kv_num_blocks` 管的是**显存**不是**位置**，两者不同。 |
| F29 | ⚪ `expire_timed_out` 是 O(队列深度) | `RequestQueue.enqueue()` 与 `dequeue()` **各调用一次** `expire_timed_out()`，而它是全表扫描。实测 `enqueue+dequeue`：深度 0 → 7.5 µs，深度 100 → 23.9 µs（3.2x）。当前 `maxsize = max_batch_size*8 = 32` 有上界，扫 32 项无所谓；但若为「多缓冲请求」把 `maxsize` 调到几万，`enqueue` 会变成 O(N) 热点。改法：只在 `dequeue` 扫描，或改用按到期时间排序的结构。详见 [`0012`](0012-async-and-blocking.md) §9。 |
| F31 | ⚪ **抢占实现过简** | Phase 9 的 CPU swap 有 6 个具体问题：① **只重试一次**（`try allocate → swap → try → oom`），一个受害者不够就放弃，应循环抢多个；② **只从 `state=="decoding"` 挑受害者**，running 里全是 `chunked_prefilling` 时无候选 → 直接 `oom`；③ **largest-first 惩罚最长的请求**，对公平性/尾延迟不利（vLLM 是 FCFS，抢占**最后到达**的）；④ **CPU 池只有设备池的一半**（`kv_num_cpu_blocks=128` vs `kv_num_blocks=256`），抢占有硬上限，超了 `CPUSwapError → oom`；⑤ 换回条件严（`空闲块 ≥ 原始块数`）+ 新请求紧接着抢块 → 被抢占者可能**长期饿死**（实测见过暂停 5 个 step）；⑥ **无防抖动**，理论上可反复拆换。详见 [`0014`](0014-prefill-and-chunking.md) §8.2.1。 |
| F32 | ✅ **已修复**：非最后一个 chunk 白算 `lm_head` | `_prefill_chunk_blocking` 里 `outputs = self.model(...)` 对**每个** chunk 都算完整 logits，但只有 `is_last_chunk` 才读它（`scheduler.py:875`），中间的 logits 直接被丢。代价（chunk_len=128）：logits 张量 `[1,128,151936]` fp16 = **38.9 MB**，`lm_head` FLOPs **34.9 GFLOP**，占这个 chunk 全部线性算力的 **27.6%**。修法：`model(..., logits_to_keep=1)`（transformers 5.16.1 的 `Qwen2ForCausalLM.forward` 支持；decode 路径本来就只喂 1 个 token，不受影响）。 |
| F30 | ⚪ `run_loop` 是轮询而非事件驱动 | `scheduler.run_loop()` 在完全空闲时 `await asyncio.sleep(1ms)` 轮询。实测（macOS）：空闲 CPU **1.39%**、**867 次唤醒/秒**；且**每个新请求最多等 1ms 才被 admit**，这段直接加在 TTFT 上（≈TTFT 的 1.5%）。两者都可以是 0：改成「新请求用 `asyncio.Event` 唤醒 + 超时/swap-in 用 `wait_for` 定时兜底」。注意 `run_loop` 有**三个**唤醒源，不能只换成 `asyncio.Queue`。详见 [`0013`](0013-polling-and-os-scheduling.md) §9。 |
| F33 | ⚪ **prefill 与 decode 对 live cache 的处理不对称** | decode 保留 `seq.past_key_values`（live HF cache，因为 MPS 上从 pool 重建约 240 ms/token），但 **chunked prefill 每个中间 chunk 都从 pool 重建一次**（`seq.past_key_values` 在 prefill 阶段始终是 `None`）→ 总拷贝量 O(n²/chunk)。**但实测差异测不出来**（`PREFILL_CHUNK_SIZE=512` vs `64`：TTFT 916/931 ms vs 954/907 ms，在噪声内）—— 因为重建是**一次批量 `torch.cat`+`permute`**，常数极小。属于"整洁性问题"而非性能问题，优先级低。改法：prefill 也保留 live cache（与 decode 一致）。详见 [`0015`](0015-kv-cache-lifecycle.md) §4。 |
| F34 | ⚪ **`ttft_ms` 名不副实：不含排队时间** | `scheduler.py:905` 是 `seq.ttft_ms = first_token_time - prefill_start_time`，而 `queue_wait_time_ms = prefill_start_time - arrival_time`。**所以 API 返回的 `ttft_ms` 只是"prefill 耗时"，真正的 TTFT（到达→首 token）= `queue_wait_time_ms + ttft_ms`。** 后果：① 只看 `ttft_ms` 会**低估**长排队场景（实测同一请求 `ttft_ms` 34.5 ms 而排队 1568 ms）；② `metrics_aggregator.compute_slo_compliance()` 用 `r.ttft_ms <= 200` 判定，**忽略了排队，SLO 偏乐观**。修法：`GenerationResult` 增加 `queue_wait_ms` 或在聚合时相加；文档/响应字段名建议改为 `prefill_ms`。 |

---

## G. 附带的运维脚枪

### F24 Phase 1 服务关闭会覆盖 `baseline_metrics.json`

- **位置**：`server/app.py` lifespan shutdown → `MetricsCollector.dump_to_json(config.metrics_output_path)`；
  `config.metrics_output_path` 默认 `"baseline_metrics.json"`（仓库已提交的基准文件）。
- **后果**：起一次 Phase 1 服务再关掉，基准就被真实（或空）结果覆盖，后续
  `run_validation_v2.py` 的对比会失真。
- **处理**：调试 launch 配置已改为写 `dev_metrics.json`；还原用
  `git checkout -- baseline_metrics.json`。

---

## 已验证的正面结论

- `uv run pytest inference_engine/tests -q` → **98 passed**。
- Phase 1（`:8000`）与 Phase 2（`:8001`）均能启动，`/health` 200，`device=mps`。
- 普通 `/generate` 正常，返回 TTFT / tokens_per_second / per-token 延迟。
- 611-token prompt 的 chunked prefill 正常完成（无异常）。
- `KV_NUM_BLOCKS=6` 下成功触发抢占：swap out 1 次、swap in 1 次、`oom_total=0`，
  两个并发请求都正常返回。

---

## 建议的自我验证方式

1. 复现 F1：`KV_NUM_BLOCKS=4` 起服务，发一个几百 token 的 prompt，观察 503。
2. 复现 F2：对比 50-token 与 600-token prompt 的 `ttft_ms`，看增长是否超线性。
3. 复现 F6：发几个请求后看 `/metrics` 的 `stage_breakdown`，注意 `total_iterations`
   远大于实际有活动的 step 数。
4. 复现 F14：给 swap 测试加一个「换入后生成的 token 与不 swap 时一致」的断言（先别改，
   想想为什么这在小模型 + greedy 下其实容易成立）。
