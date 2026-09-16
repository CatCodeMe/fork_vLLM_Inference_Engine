# 在 VSCode 中调试 PageServe

这份文档面向**用于学习的 fork**：介绍如何用 `uv` 准备环境、如何在 VSCode 里
运行/调试，以及为了看懂内部实现应该在哪里打断点。

> 是否需要 GPU？**不需要。** `config.py` 会自动选择设备（`cuda > mps > cpu`）。
> Apple Silicon 用 MPS，没有独显就纯 CPU（`float32`，能跑但慢）。本项目已在
> macOS + MPS 上实测通过。

---

## 1. 一次性环境准备（uv）

上游仓库只提供了 `requirements.txt`。我们新增了 [`pyproject.toml`](../pyproject.toml)，
让 `uv` 管理解释器和项目级 `.venv`。

```bash
# 在仓库根目录执行
uv python pin 3.10      # 可选：固定解释器（本机已装 3.10）
uv sync                 # 创建 .venv 并安装 runtime + dev 依赖
uv sync --no-dev        # 只装 runtime 依赖
```

`uv sync` 安装的内容：

- runtime：`torch`、`transformers`、`fastapi`、`uvicorn[standard]`、`psutil`、
  `numpy`、`httpx[socks]`、`accelerate`
- dev（dependency-group `dev`）：`pytest`、`pytest-asyncio`

一次性命令统一用 `uv run` 前缀，确保走 `.venv`：

```bash
uv run python scripts/serve.py --phase 2
uv run pytest inference_engine/tests -v
```

上游 `requirements.txt` 保留，供参考 / 纯 `pip` 用户使用。

> 关于 `httpx[socks]`：如果机器配置了 SOCKS 代理（`ALL_PROXY`），HuggingFace Hub
> 通过 `httpx` 下载模型会因缺少 `socksio` 而失败。加上 `[socks]` extra 即可。

---

## 2. 选择解释器

`.vscode/settings.json` 已把 VSCode 指向 `${workspaceFolder}/.venv/bin/python`。
如果状态栏仍显示系统 Python：

1. `Cmd+Shift+P` → **Python: Select Interpreter**
2. 选择 `./.venv/bin/python`

---

## 3. Launch 配置

打开 **Run and Debug** 面板（`Cmd+Shift+D`），从下拉框选配置。
完整列表见 [`.vscode/launch.json`](launch.json)。

| 配置 | 端口 / 模式 | 需要先起服务吗 |
| --- | --- | --- |
| 🌐 Serve: Phase 1 sequential | `:8000` | — |
| 🌐 Serve: Phase 2 continuous batching | `:8001` | — |
| 🐍 Debug current file | 由 `${file}` 决定 | — |
| 🧪 Pytest: current file / all tests | pytest | — |
| 📊 Validate Phase 1 | `run_validation.py` | Phase 1 |
| 📊 Validate Phase 2 vs baseline | `run_validation_v2.py` | Phase 2 |
| 🚦 Load test: constant / ramp / burst | `run_load_test.py` | Phase 2 |
| ⚡ Bench: direct / batched decode / per-phase | 直接加载模型 | — |

### 为什么用 wrapper 脚本而不是 `uvicorn ...`？

`inference_engine/server/app.py` 和 `app_v2.py` 都没有
`if __name__ == "__main__"`，无法直接对着文件按 F5。
[`scripts/serve.py`](../scripts/serve.py) 是可调试入口，它在**同一进程**里调用
`uvicorn.run(app_path, reload=False)`，因此 scheduler 里的断点能正常命中。

> ⚠️ 调试时**不要**开 `--reload`：reloader 会 fork 子进程，debugger 会挂到父进程上。

### 推荐工作流

1. `F5` → **Serve: Phase 2 continuous batching**，等待
   `Application startup complete.`（首次运行会下载 `Qwen/Qwen2-0.5B`）。
2. 设置断点（见下文）。
3. 在另一个终端发请求：
   ```bash
   curl -s http://127.0.0.1:8001/generate \
     -H 'content-type: application/json' \
     -d '{"prompt": "Explain paged attention.", "max_new_tokens": 20}' | python -m json.tool
   ```
4. 检查变量、单步跟踪 scheduler。

### 不用写代码的调试入口：Swagger UI

FastAPI 自带交互式接口文档（本项目没有关掉它）， **不用 curl 也能直接发测试请求**：

| 地址 | 是什么 |
| --- | --- |
| **http://127.0.0.1:8001/docs** | **Swagger UI** —— 展开 `POST /generate` → `Try it out` → 填 prompt → `Execute` |
| http://127.0.0.1:8001/redoc | ReDoc（只读、排版更好看的同一份文档） |
| http://127.0.0.1:8001/openapi.json | 原始 OpenAPI schema（可用 `jq` 看请求体约束） |

> 端口跟着服务走：Phase 1 是 `:8000`，Phase 2 是 `:8001`。
> `scripts/serve.py --port 8002` 起的服务就用 `:8002/docs`。

### `Connection refused` 与"服务没起"不是一回事

这是重启后最常撞的一个坑。uvicorn 0.52 的 `Server.startup()`：

```python
async def startup(self, sockets=None):
    await self.lifespan.startup()             # ← 模型加载在这里（阻塞几秒~几十秒）
    if self.lifespan.should_exit: ...
    server = await loop.create_server(...)    # ← 端口到这里才 bind
```

**端口是加载完成后才打开的。** 实测时间线（`HF_HUB_OFFLINE=1`，模型已缓存）：

```
21:49:17.125  Phase 2 server starting: model=Qwen/Qwen2-0.5B device=mps
21:49:17.125  Loading model 'Qwen/Qwen2-0.5B' targeting device 'mps'
21:49:19.055  Scheduler run_loop started
              INFO:     Application startup complete.
              INFO:     Uvicorn running on http://127.0.0.1:8003    ← 此端口才打开
              （从启动到端口可连 = 5.28s）
```

所以两种失败要分清 —— **它们的含义完全不同**：

| 你看到的 | 含义 | 怎么办 |
| --- | --- | --- |
| **`Connection refused`（errno 61）** | 端口**没在监听**：服务没起，或**正在加载模型** | 等 `Uvicorn running on ...` 出现；`demo_phase2.py` 默认帮你等（`--wait 60`） |
| 超时 / 挂住 | 端口在监听，但应用忙或没响应 | 看服务端在干什么（`/metrics`、日志） |

两个小工具已经处理了这件事：

- `scripts/serve.py` 启动时会先打印一行说明（端口要等加载完才监听）；
- `scripts/demo_phase2.py` 有 `--wait`（默认 60s），会轮询 `/health` 直到就绪，
  并且把 `Connection refused` 和"超时"分开报；`--wait 0` 可关掉直接报错。

> 顺带：**首次运行还要先下载模型**（~1GB），那段时间同样是 `Connection refused`，
> 而且窗口可能几分钟。想跳过联网检查：`HF_HUB_OFFLINE=1`。

### 跟一个请求走完全程：`[REQ]` / `[step NNN]` 轨迹日志

`app_v2.py` 与 `scheduler.py` 里已经埋好一条带**步标号**的日志线，
把整个生命周期串起来。**两个标号各管一事，别看混**：

| 标号 | 含义 |
| --- | --- |
| **`N/8`** | 在**一个请求的生命周期**里的位置（固定 8 步，看它此刻在干什么） |
| **`[step NNN]`** | 在**调度器**里的第几次循环（自己递增，看谁跟谁落在同一次调度里） |

| `N/8` | 位置 | 内容 |
| --- | --- | --- |
| 1/8 收到 | `app_v2.endpoint_generate` | 收到 POST |
| 2/8 入队 | `scheduler.add_request` | 入队（分词完成 + 队列深度） |
| 3/8 step | `_schedule` 开头 | step 开始：`running/waiting/swapped` 三个尺寸 |
| 3/8 swap-in | Step 0 | 把被抢占的序列换回设备 |
| 4/8 admit | Step 1 | admit（含 frozen 的 chunk 大小、chunk 数）；批满时打"跳过" |
| 5/8 prefill | Step 2 / `_prefill_chunk_blocking` | **每个 chunk 一行** + offset 一致性自检 |
| 5/8 swap-out | `_try_swap_out_victim` | 抢占：把某个 decoding 序列换到 CPU |
| 6/8 decode | Step 3 | 每个序列推进 1 个 token |
| 7/8 finish | Step 4 | finish（`finish_reason` + tokens） |
| 8/8 返回 | `app_v2.endpoint_generate` | 返回 200（端到端耗时） |

**颜色**：阶段（stage）在打印时会按 `N/8` 上色 —— prefill 青、decode 绿、finish 黄、
admit 洋红、step/swap 蓝、抢占红、HTTP 层暗色。**只在输出到终端时上色**：
重定向到文件、管道、或设了 `NO_COLOR` 就自动关，所以日志文件和 `/trace` 的 JSON
里不会有 ANSI 转义码。

**`/trace` 与 `/spans` 是两种视角，配着看：**

| 接口 | 形态 | 适合 |
| --- | --- | --- |
| `GET /trace` | **事件日志**（时间顺序，一行一条 + `N/8` 标号） | 排查"某一步发生了什么"、grep |
| `GET /spans` | **span 树**（OTel 风格：父子 + start/duration） | 看"整体形状"、画瀑布图、导进 Jaeger 类工具 |

```bash
# span 树：request → queue_wait / prefill(+chunk#i) / decode(+token#i)
curl -s 'localhost:8001/spans?limit=5&max_children=64' | python -m json.tool
```

`demo_phase2.py` 会把 span 树画成**文本瀑布图**：

```
[请求瀑布图]  横轴 = 相对 arrival；最长 2412ms；每格 40.2ms
  图例：q=排队  P=prefill（| 为 chunk 边界）  d=decode
  #1  655b73b8  qqqqqqqqqqPP|P|P|PPPPPPPPPPPPPPddddddddddddddddddddddddddddd  queue 438ms  prefill 837ms/4chunk  decode 1137ms/12tok  总 2412ms
  #4  077c55a7  qqqP|ddddddddddd                                              queue 158ms  prefill 52ms/1chunk   decode 473ms/6tok   总 683ms
```

一眼能看出：长 prompt 前面有长 `q`（排队）和 4 段 `P`（分块 prefill），短请求几乎不排队。

**⚠️ 缓冲区是「服务端累积」的，not「本次运行」的**

`/trace` 和 `/spans` 的数据源是 scheduler 实例上的环形缓冲区
（`_trace_log` / `self.finished`），**活得和服务器进程一样久**。所以：

- 不重启服务，历史记录就一直累积；
- `--n 1` 也可能在输出里看到**别的** seq —— 那不是脚本发重了，是上一次跑留下的；
- 同一台服务连跑三次，第二次会看到第一、二次的记录。

`demo_phase2.py` 已经**默认按本次返回的 `seq_id` 过滤**，并打印忽略了多少条历史：

```
（轨迹已按本次 2 个请求过滤，忽略了服务端缓冲区里的 46 条历史记录；--trace-all 可看全部）
```

想看原始缓冲区（含历史）：`--trace-all`。想彻底清空：**重启服务**。

> ⚠️ span 视图**不新增热路径埋点** —— 所有时间点都是 `Sequence` 上已有的字段
> （`arrival_time` / `prefill_start_time` / `first_token_time` / `finish_time`
> + `prefill_chunk_latencies_ms` / `per_token_latencies_ms`）推导出来的，
> 所以打开它对引擎性能零影响。

**两种取轨迹的方式**（内容一样，都来自同一个环形缓冲区）：

| 方式 | 命令 | 适合 |
| --- | --- | --- |
| 服务端日志 | `grep -E '\[(REQ|step [0-9]+)\]' server.log` | 服务在前台/重定向到文件时 |
| **`GET /trace`** | `curl -s 'localhost:8001/trace?limit=200' \| python -m json.tool` | 服务在别处跑，或想程序化处理 |

**`scripts/demo_phase2.py` 已经自动接上了 `/trace`** —— 跑完直接打一份摘要：

```bash
uv run python scripts/demo_phase2.py --n 4 --max-new-tokens 4
#   → [本批请求] 表格（# 序号 + seq 前缀 + 排队/TTFT/总时延）
#   → [调度轨迹] 带 N/8 + [step NNN] 标号的逐行摘要（长 prompt 的中间 chunk 自动折叠）
# 关掉：--no-trace   看每个 chunk：--trace-raw   条数：--trace-limit 500
```

**一条命令抽出全部轨迹（不用翻 server.log）：**

```bash
# 服务端日志重定向到文件时
grep -E '\[(REQ|step [0-9]+)\]' server.log

# 或者直接在终端里看（服务前台运行时）
#   把 log level 调到 DEBUG 还会多出每 token 的细节
uv run python scripts/serve.py --phase 2 --log-level debug
```

**想看到多个 chunk（触发分块 prefill），用一个长 prompt + 小的 chunk 大小：**

```bash
# serve 端：把 chunk 调小，让 600 token 的 prompt 切成 10 段
PREFILL_CHUNK_SIZE=64 uv run python scripts/serve.py --phase 2

# 客户端：造一个 ~600 token 的长 prompt
python3 -c "
import json
json.dump({'prompt': 'Explain paged attention and continuous batching in detail. ' * 62,
           'max_new_tokens': 4}, open('/tmp/req.json', 'w'))"
curl -s http://127.0.0.1:8001/generate -H 'content-type: application/json' \
  -d @/tmp/req.json | python -m json.tool
```

实测输出（622 token，chunk=64，节选）：

```
[REQ]      1/8 收到       POST /generate prompt=3658 字符 max_new_tokens=4
[REQ]      2/8 入队       seq=a328f4b9 prompt_len=622 max_new=4 队列深度=1/16
[step 001] 3/8 step      running=0 waiting=1 swapped=0 上限=2
[step 001] 4/8 admit     seq=a328f4b9 prompt_len=622 chunk_size=64 chunk数=10 (running 1/2)
[step 001] 5/8 prefill   seq=a328f4b9 首个 chunk：按整段 prompt 分配 39 个块（622 token / block=16），分配后池内剩余 217 块；排队等待 1.4ms
[step 001] 5/8 prefill   seq=a328f4b9 chunk=[0,64) 64 token → 写 pool 位置 0–63；offset 0→64；prompt 剩余 558
[step 002] 3/8 step      running=1 waiting=0 swapped=0 上限=2
[step 002] 5/8 prefill   seq=a328f4b9 从 pool 重建 past_kv：64 条 KV == prefill_offset=64 ✓
[step 002] 5/8 prefill   seq=a328f4b9 chunk=[64,128) 64 token → 写 pool 位置 64–127；offset 64→128；prompt 剩余 494
... （共 10 个 chunk）...
[step 010] 5/8 prefill   seq=a328f4b9 ✅ 最后一个 chunk 完成：共 10 个 chunk，采样首个 token=16，ttft=2031.1ms，state → decoding
[step 010] 6/8 decode    seq=a328f4b9 → 2/4 token
[step 012] 6/8 decode    seq=a328f4b9 → 4/4 token
[step 012] 7/8 finish    seq=a328f4b9 reason=length tokens=4/4 ttft=2031.1ms → 清 pool + 还块 + future.set_result
[REQ]      8/8 返回       seq=a328f4b9 tokens=4 finish_reason=length ttft=2031.1ms 端到端=2177.0ms
```

**这张日志里值得盯的四个变化点：**

1. `3/8 step` 那行的 `running/waiting/swapped` —— 连续批处理的“进进出出”全在这里。
2. `5/8 prefill` 的 `从 pool 重建 past_kv：N 条 KV == prefill_offset=N ✓` ——
   **这就是“三个 offset 一致”的可观测证据**（原理见 `0014` §3）。一旦不一致会打 `ERROR`。
3. `5/8 prefill` 的 `chunk=[a,b)` 区间逐段右移，且 `剩余` 单调递减到 0 —— 游标没重叠没遗漏。
4. `6/8 decode` 的 `seq=X → n/上限 token` —— 多个序列时能看到它们各推各的、互不干扰。

> 想看 **swap/抢占**（`⚡ swap-out` + `② swap-in`），需要同时满足三个条件，
> 否则你只会看到 `reason=oom`：
>
> 1. **先有一个正在 `decoding` 的序列占着块** —— 如果新请求自己就超过全池，
>    它会直接 `oom`（`_try_swap_out_victim` 找不到候选：它只挑 `state=="decoding"` 的）；
> 2. **全池容量 ≥ 两个序列的块数之和** —— 否则换出受害者后重试仍然失败；
> 3. **新请求要的块数 > 当时的空闲块数** —— 这才是触发 `OutOfBlocksError` 的条件。
>
> 已验证的配方（本机实测）：
>
> ```bash
> # 池子 45 块 = 720 token
> KV_NUM_BLOCKS=45 MAX_BATCH_SIZE=2 uv run python scripts/serve.py --phase 2
>
> python3 -c "
> import json
> b = 'Explain paged attention and continuous batching in detail. '
> json.dump({'prompt': b*40, 'max_new_tokens': 64}, open('/tmp/A.json','w'))  # ≈402 token → 26 块
> json.dump({'prompt': b*50, 'max_new_tokens': 4},  open('/tmp/B.json','w'))  # ≈502 token → 32 块
> "
> # A 先跑起来（它会 decoding 很久，给我们留出窗口），2.5 秒后再发 B
> curl -s localhost:8001/generate -H 'content-type: application/json' -d @/tmp/A.json > /tmp/A.out &
> sleep 2.5
> curl -s localhost:8001/generate -H 'content-type: application/json' -d @/tmp/B.json > /tmp/B.out &
> ```
>
> 关键几行（实测输出）：
>
> ```
> [step 058] 4/8 admit     seq=b0618855 prompt_len=502 chunk_size=128 chunk数=4 (running 2/2)
> [step 058] 5/8 swap-out  seq=b5796f53 换出 29 个块到 CPU（largest-first）　→ running=1 swapped=1 空闲块=45
> [step 063] 7/8 finish    seq=b0618855 reason=length tokens=4/4 ttft=830.4ms
> [step 064] 3/8 swap-in   seq=b5796f53 块=29 → 回到 decoding（swapped 还剩 0）
> [step 072] 7/8 finish    seq=b5796f53 reason=length tokens=64/64 ttft=650.1ms   ← 受害者没被丢，跑完了
> ```
>
> **两个请求都返回 200** —— 这就是抢占的设计意图：**不杀请求**。
> 代价转移给被抢占者：A 的端到端耗时 4163ms（中间被搬去 CPU 停了几步），
> 而它的 TTFT 仍然只有 650ms。
>
> 完整的八步标号说明见 `0014` §9。

校验 / 压测配置都声明了 `preLaunchTask`（`wait: phase N health`），会轮询 `/health`
直到模型加载完成。所以你可以先 F5 起服务，再立刻启动其中一个，VSCode 会并行跑两个
调试会话。

---

## 4. 请求生命周期 —— 在哪里打断点

> 配套的**时序图**与 **`_schedule` 阶段图**见
> [`0006-scheduler-lifecycle.md`](0006-scheduler-lifecycle.md)。

借助 **Call Stack** 面板，重点锚点如下。

### 入队路径（async，在 FastAPI 事件循环里）

```
POST /generate
└─ inference_engine/server/app_v2.py  endpoint_generate
   └─ ContinuousBatchingScheduler.add_request()      engine/scheduler.py
      ├─ tokenizer(prompt)  (在线程池里跑)
      ├─ Sequence.create()                            engine/sequence.py
      └─ RequestQueue.enqueue()                       engine/request_queue.py
   └─ await future          # handler 挂起，直到 scheduler 解析该 future
```

### 执行路径（后台 `run_loop` 任务）

```
ContinuousBatchingScheduler.run_loop()               engine/scheduler.py
└─ _schedule()                                       engine/scheduler.py
   ├─ Step 0  swap-in   → CPUSwapManager.swap_in()            engine/cpu_swap_manager.py
   ├─ Step 1  admit     → RequestQueue.dequeue()              engine/request_queue.py
   ├─ Step 2  prefill   → _prefill_chunk_blocking()           engine/scheduler.py
   │                       ├─ BlockAllocator.allocate()        engine/block_allocator.py
   │                       ├─ _try_swap_out_victim()           engine/scheduler.py  (OOM 时)
   │                       └─ PagedKVCacheManager.write_kv()   engine/paged_kv_cache.py
   ├─ Step 3  decode    → _decode_step_single()               engine/scheduler.py
   │                       ├─ BlockAllocator.write_token()     engine/block_allocator.py
   │                       └─ PagedKVCacheManager.read/write_kv
   └─ 淘汰 finished      → _resolve_sequence_future()          engine/scheduler.py
                            └─ future 解析 → endpoint_generate 组装 JSON 返回
```

> 由于加了注释，具体行号请以编辑器为准，上面用函数名定位更稳。

### 建议的第一批断点

| 目标 | 断点 |
| --- | --- |
| 看请求如何进入系统 | `scheduler.py: add_request` |
| 看 iteration-level 批处理 | `scheduler.py: _schedule` |
| 看 chunked prefill 与块分配 | `scheduler.py: _prefill_chunk_blocking` |
| 看单个 decode step | `scheduler.py: _decode_step_single` |
| 看内存压力 / 抢占 | `scheduler.py: _try_swap_out_victim` |
| 看 future 解析 → HTTP 响应 | `scheduler.py: _resolve_sequence_future` |

小技巧：右键断点 → **Log Message**，可以不停下来地打印，例如
`step: running={len(self.running)}`。

---

## 5. 独立 benchmark

`bench_direct.py`、`bench_batched_decode.py`、`bench_phases.py` 直接加载模型
（不需要 HTTP 服务）。想观察最原始的 `transformers` forward pass，从这里下手最直接。

---

## 6. 实测结论（本机 macOS + MPS）

| 验证项 | 结果 |
| --- | --- |
| 单元测试 | `uv run pytest inference_engine/tests -q` → **98 passed** |
| Phase 1 服务 | `:8000` `/health` 200，`device=mps` |
| Phase 2 服务 | `:8001` `/health` 200，`device=mps` |
| 普通生成 | `/generate` 正常返回 TTFT、tokens/s |
| Chunked prefill | 611-token prompt（5 个 chunk）正常完成 |
| CPU swap | `KV_NUM_BLOCKS=6` 下触发 1 次 swap out + 1 次 swap in，`oom_total=0` |

---

## 7. 常见问题

| 现象 | 处理 |
| --- | --- |
| `Address already in use` | 已有服务在跑；停掉它，或用 `scripts/serve.py --port` 换端口。 |
| **刚起服务就 `Connection refused (errno 61)`** | **不是"起不来"** —— uvicorn 的顺序是 `lifespan.startup()`（加载模型，阻塞几秒~几十秒）**先**、`create_server()`（bind 端口）**后**，所以模型加载期间端口根本没在监听。实测本机（模型已缓存）≈ **5s**，首次下载更长。详见 §3 里的「`Connection refused` 与"服务没起"不是一回事」。 |
| 断点不命中 | 确认是用调试配置启动（不是终端里的 `uvicorn`），且 `reload=False`。 |
| `ModuleNotFoundError: inference_engine` | 从仓库根目录运行；`launch.json` / `tasks.json` 已设 `PYTHONPATH`，根目录 `conftest.py` 负责 pytest。 |
| 模型下载慢 / gated | 在 `.env` 或环境变量里设 `MODEL_NAME`；SOCKS 代理需 `httpx[socks]`（已加）。 |
| 想换加速设备 | 设 `DEVICE=cuda`/`mps`/`cpu`（不设则自动检测）。 |
| 首个请求很慢 | 模型在 warmup，后续会快。 |
| `baseline_metrics.json` 被改了 | Phase 1 服务关闭时会覆盖 `metrics_output_path`；调试配置已改为写 `dev_metrics.json`。用 `git checkout -- baseline_metrics.json` 还原。 |

VSCode 会读取仓库根目录的 `.env`（可参考 `.env.example`）。当前配置未强制
`envFile`，可用 `env` 块或直接在 shell 里设置。
