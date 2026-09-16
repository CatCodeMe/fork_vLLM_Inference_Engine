# Phase 索引：编号 ↔ 文件 ↔ 代码位置

源码里的学习注释大量出现 `Phase 2` / `Phase 6` / `Phase 9` 这样的编号，但在
`scheduler.py` 里从上往下读，编号会来回跳（`6 → 7 → 9 → 3 → 10 → 6 → 9 …`）。
这不是注释写错了，是**三个视角错位**。本页给出权威对照，方便按编号反查代码。

> 编号的权威定义在 [`README.md`](../README.md) 的 *Development Phases* 一节
> （Phase 1~11）。本页与之一致，补充了「代码在哪里」这一层。
> 阅读顺序请不要按编号走，按 [`0001-learning-roadmap.md`](0001-learning-roadmap.md) 走。

---

## 0. 一句话结论

| 视角 | 顺序由什么决定 |
| --- | --- |
| **Phase 编号** | 功能增量被引入的**时间顺序**（上游作者一步步加的） |
| **文件** | **组件**划分，一个文件会被多个 Phase 反复修改 |
| **函数体内** | **数据流 / 执行流**（先接线、再调度、最后收尾） |

三者互不相干，所以同一个文件里 Phase 编号必然乱序。**Phase 编号在这类位置只是
「这块代码是哪个阶段引入的」的溯源标签，不是阅读顺序。**

---

## 1. 权威编号表

| Phase | 主题 | 主要文件 | 代码入口 | 测试 |
| --- | --- | --- | --- | --- |
| 1 | 顺序服务基线 | `engine/sequential.py`、`server/app.py`、`metrics/collector.py`、`models/loader.py` | `sequential.generate()` (`:356`)、`collector.compute_summary()` (`:70`) | `test_sequential.py` |
| 2 | 连续批处理调度 | `engine/scheduler.py`、`server/app_v2.py`、`engine/sequence.py` | `scheduler.add_request()` (`:544`)、`run_loop()` (`:1452`)、`_schedule()` (`:1243`) | `test_scheduler.py` |
| 3 | 请求队列 | `engine/request_queue.py` | `RequestQueue.enqueue/dequeue`、`expire_timed_out` | `test_request_queue.py` |
| 4 | Prefill / Decode 分离 | `engine/stage_tracker.py`、`engine/prefill_utils.py`、`scheduler` | `_prefill_chunk_blocking()` (`:721`)、`_decode_step_single()` (`:953`) | `test_stage_tracker.py` |
| 5 | KV 内存跟踪 | `engine/kv_cache_config.py`、`engine/kv_cache_tracker.py` | `compute_kv_cache_config()` (`:37`)、`KVCacheTracker.stats()` | `test_kv_cache.py` |
| 6 | 块分配器 | `engine/block_allocator.py` | `allocate()` (`:129`)、`write_token()` (`:217`)、`free()` (`:165`) | `test_block_allocator.py` |
| 7 | Paged KV cache | `engine/paged_kv_cache.py` | `write_kv()` (`:81`)、`read_kv_sequence()` (`:125`)、`clear_sequence()` (`:188`) | `test_paged_kv_cache.py` |
| 8 | 批注意力接入 | `engine/attention_wrapper.py` | `build_past_key_values()` (`:154`)、`extract_new_token_kv()` (`:194`) | `test_attention_wrapper.py` |
| 9 | CPU swap pool | `engine/cpu_swap_manager.py`、`scheduler` | `swap_out()` (`:151`)、`swap_in()` (`:247`)、`_try_swap_out_victim()` (`scheduler:1172`) | `test_cpu_swap_manager.py` |
| 10 | 统一指标 | `engine/metrics_aggregator.py` | `record_token_generated()` (`:146`)、`full_report()` (`:319`)、`scheduler.get_metrics()` (`:1510`) | `test_metrics_aggregator.py` |
| 11 | 压测工具 | `load_test/profiles.py`、`runner.py`、`report.py`、`run_load_test.py` | `run_load_test.py --profile constant` | `test_load_test_profiles.py`、`test_load_test_report.py` |

> 行号会随注释增删漂移，**以函数名定位更稳**（`rg -n "def _schedule"`）。

---

## 2. 为什么看起来是乱的

### 2.1 `scheduler.py` —— 最乱的地方

`scheduler.py` 是 Phase 2 出生、之后被 3/4/5/6/7/8/9/10 反复改的文件。按行号列出
全部 Phase 标记：

| 行号 | 标记 | 这段代码实际在干什么 |
| --- | --- | --- |
| 2–48 | 2, 3, 4, 8, 9 | 模块 docstring：讲「本文件后来长成了什么」 |
| 226 | **6** | `__init__`：BlockAllocator（账本） |
| 232 | **7** | `__init__`：PagedKVCacheManager（仓库） |
| 244 | **9** | `__init__`：CPUSwapManager（临时仓） |
| 257 | **3** | `__init__`：RequestQueue |
| 270 | **10** | `__init__`：MetricsAggregator |
| 667 / 694 / 708 | **6 / 7 / 8** | `_prefill_sequence`：分配块 → 写 KV → 释放 HF tensor |
| 674 | **9** | `_prefill_sequence`：块不够时先尝试 swap out |
| 970 | **9** | `_decode_step_single`：注释说明「池子每步都更新，是为了将来能淘汰」 |
| 1063 / 1095–1096 | **10 / 2, 8** | `_decode_step`：记吞吐；注释解释「批处理是 Phase 2 的约束」 |
| 1156–1167 | **10 / 7 / 6** | `_resolve_sequence_future`：清池 → 还块 → 记完成 |
| 1282 | **9** | `_schedule` Step 0：swap-in |
| 1508 | **10** | metrics API 区段 |

`__init__` 的顺序是 **6 → 7 → 9 → 3 → 10**：因为它是按**依赖接线**排的
（allocator 是账本 → paged cache 是仓库 → swap 是临时仓 → 队列 → 指标），
跟编号无关。`_schedule` 同理，它按**一次调度的执行顺序**排：

```
Step 0 swap-in (9) → Step 1 admit (3) / chunk-size (4) → Step 2 chunked prefill (4,6,7)
→ Step 3 decode (2,8,9) → Step 4 evict + 收尾 (7,6,10)
```

### 2.2 `config.py` —— 唯一整齐的文件

`config.py:66-96` 是 **2 → 3 → 4 → 5 → 6 → 9** 升序，因为它按「配置项归属的
子系统」分段，后来新增的 Phase 9 段直接追加在末尾，恰好没破坏顺序。

### 2.3 `app_v2.py` —— 服务层混了三代

`app_v2.py:2` 是 Phase 2，`:4` 在对比 Phase 1，`:245` 的响应结构是 Phase 10。
服务层是「横切」的：每个 Phase 加功能时都要动它一点。

---

## 3. 注释标签约定

在源码里追加学习笔记时统一用这几个标签（检索方式见
[`0001-learning-roadmap.md`](0001-learning-roadmap.md) 的「注释约定」）：

| 标签 | 含义 |
| --- | --- |
| `# [LEARN]` | 解释概念 / 算法 |
| `# [WHY]` | 设计取舍的理由 |
| `# [TRACE]` | 标注请求调用链中的一步 |
| `# [GOTCHA]` | 容易踩的坑 |
| `# [TODO-READ]` | 待深入的点 |

**建议**：`Phase N` 只写在模块头 docstring（「本文件主要由 Phase N 引入」）；
函数体内的跨阶段接线改用 `# [PHASE 6]` 形式，这样 `rg "# \[PHASE "` 能按编号聚合，
且视觉上和 `[LEARN]/[WHY]` 区分开。（尚未执行，见文末。）

---

## 4. 常用检索

```bash
# 某个 Phase 在代码里出现的所有位置
rg -n "Phase 9" inference_engine load_test

# 只要学习标签，不要 Phase 溯源
rg -n "# \[(LEARN|WHY|TRACE|GOTCHA|TODO-READ)\]" inference_engine load_test

# 某一层“接线”长什么样
rg -n "Phase \d" inference_engine/engine/scheduler.py

# 某 Phase 的测试
uv run pytest inference_engine/tests/test_cpu_swap_manager.py -v
```

---

## 5. 如果只记一件事

`scheduler.py`（1512 行）分三层，认清这三层就不会被编号带跑：

![scheduler.py 三层结构：接线层 / 执行层 / 收尾层](figures/scheduler-layers.drawio.svg)

> 图的源文件就是 `docs/figures/scheduler-layers.drawio.svg` 本身（SVG 外壳 +
> 内嵌 draw.io XML），双击用 draw.io 扩展即可改。参见 `0004-figures.md` §一。

| 层 | 位置 | 作用 |
| --- | --- | --- |
| 接线层 | `__init__` (`:198`) | 把 6/7/9/3/10 各组件组装起来，只跑一次 |
| 执行层 | `_schedule()` (`:1243`) → `run_loop()` (`:1452`) | 每个 iteration 推进一次 |
| 收尾层 | `_resolve_sequence_future()` (`:1117`) | 清池、还块、resolve future |

---

## 待办

- [x] ~~统一源码 Phase 标注格式为 `# [PHASE N]`~~ —— **评估后决定不做**，理由见下。

### 为什么不改 `Phase N` 的写法

现有的 `# Phase 6: Block allocator` 已经能被 `rg "Phase 6"` 精确检索到，
改成 `# [PHASE 6]` 只多两个方括号，**信息量零增益**，却要动 8+ 个文件、
让这些学习注释的行号集体漂移（而我们刚刚才用行号建了对照表）。

真正的痛点不是写法，而是**函数体里的 Phase 编号是「历史」而不是「结构」**
—— 这个问题已经由本页（编号反查表）+ `0006` 的时序图解决，不需要改源码。
