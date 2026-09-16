# 异步、阻塞与 GIL：本仓库的 async 边界

这篇文档是 `scheduler.py` / `request_queue.py` 里那几条 `[LEARN]` 注释的展开版。
起因是三个很自然的问题：

1. `add_request` 里分词用了 `await`，可它明明还在等分词完成 —— 那放线程池图什么？
2. `enqueue` 只是往列表里追加一项，为什么也要 `await`？
3. GIL 是什么的缩写？它和上面两个问题什么关系？

> 代码里的注释位置：`scheduler.py:519-526`（`add_request`）、`scheduler.py:724`
> 与 `:655`（`_blocking` 契约）、`request_queue.py:136-152`（`enqueue` 为什么 async）。
> 相关：[`0011-stopping-and-cancellation.md`](0011-stopping-and-cancellation.md)（取消与阻塞的关系）、
> [`0013-polling-and-os-scheduling.md`](0013-polling-and-os-scheduling.md)（事件循环与 OS 调度、1ms 轮询间隔）。

---

## 0. 一页速查

| 概念 | 全称 / 含义 | 一句话 |
| --- | --- | --- |
| **GIL** | **G**lobal **I**nterpreter **L**ock，全局解释器锁 | CPython 里同一进程同一时刻只有一个线程能跑 Python 字节码 |
| **事件循环** | event loop | 协程调度器。uvicorn 默认**一进程一线程一循环**，所有请求共享它 |
| **`await`** | — | 挂起当前协程、**把循环交出去**。**不等于「不等待」** |
| **`async def` 判据** | — | `async def` ⟺ 需要挂起点：`await` / `async with` / `async for` |
| **`to_thread`** | `asyncio.to_thread` / `run_in_executor` | 把阻塞函数丢到线程池，空出事件循环 |
| **`_blocking` 后缀** | 本仓库命名契约 | 阻塞的同步函数，**只能**由 async 调用方 `to_thread` 派发 |

**三条最重要的结论**：

1. `await` 的收益**不是「调用方不用等」**，而是「**事件循环不用等**」。调用方照样在等。
2. `to_thread` 的收益分两层：**响应性**（人人有份）+ **真并行**（仅当工作释放 GIL）。
3. `enqueue` 的 `async` **不是性能优化**（实测净开销 418 ns vs 24 ms 的 step），
   它是「访问边界的声明」。

---

## 1. GIL 是什么

**GIL = Global Interpreter Lock，全局解释器锁。**

它是 **CPython 的实现细节**（不是 Python 语言规范）：同一进程内，**同一时刻只有一个
线程能执行 Python 字节码**。想跑 Python 代码的线程必须先拿到这把锁。

### 为什么会有它

1. **让内存管理变简单**：CPython 用引用计数管理对象。如果没有 GIL，每次 `obj.refcount += 1`
   都要加锁 —— 那会慢得离谱。GIL 把「整个解释器」当成一个粗粒度临界区，
   引用计数就完全不用加锁。
2. **让 C 扩展好写**：扩展作者默认「我持有 GIL，没人会并发改我的对象」。
   这是生态能长起来的关键原因之一。

代价就是：**多线程无法并行跑 Python 代码**。

### 两个必须记住的推论

| # | 推论 | 后果 |
| --- | --- | --- |
| 1 | 纯 Python 的 CPU 活，多线程**不会变快** | 8 线程 ≈ 1 线程，只是轮流持有 GIL |
| 2 | **C 扩展可以主动释放 GIL** | 那段代码执行期间，别的线程能跑 → 这才是真并行 |

**推论 2 是 `to_thread` 有意义的前提。** 谁释放、谁不释放：

| 操作 | 释放 GIL？ | 说明 |
| --- | --- | --- |
| 纯 Python 循环 / 计算 | ❌ | 推论 1 适用 |
| `time.sleep` | ✅ | 睡眠期间当然不需要锁 |
| 文件 / 网络 I/O | ✅ | 等待内核时释放 |
| numpy / torch 的大矩阵运算 | ✅ | 计算在 C/C++/CUDA/Metal 里跑 |
| HF **fast** tokenizer（Rust `tokenizers`） | ✅ | **本文实测确认**（§4 的 2.31x / 3.18x） |
| `hashlib` / `zlib` / `sqlite3` | ✅ | 常见 C 扩展都显式释放 |
| `json.dumps` 大对象 | ⚠️ | C 实现部分释放 |

> ⚠️ 对本项目的直接意义：decode 的 forward 绝大部分时间在 C++/Metal kernel 里、
> **会释放 GIL**，所以 `to_thread` 能用。但如果哪天把热点换成纯 Python 逻辑
> （比如手写 attention loop），多线程收益会立刻消失 —— 只剩响应性。

### 版本演进（生产环境还没普及）

- **3.12**（PEP 684）：支持 **per-interpreter GIL**（子解释器各自拿锁，仍是共享 GIL 的变体思路）。
- **3.13**（PEP 703）：提供 `--disable-gil` 的 **free-threaded 实验构建**，
  首次真正去掉 GIL。生态兼容性仍在推进中。
- 本仓库 `requires-python = ">=3.10"`，用的是**有 GIL** 的普通构建。

### 对比 Java / Go

Java、Go **没有 GIL**，多线程真并行共享堆 —— 代价是内存模型复杂（`volatile`、
`synchronized`、happens-before）、GC 要处理并发。CPython 选择了「单线程快 + 生态好写」，
把并行的问题推给 `multiprocessing` 或 C 扩展。

> 这也是为什么**真实推理引擎把 engine core 放进独立进程**：跨过 GIL 的最彻底办法
> 就是别在同一个解释器里。本仓库「引擎同步、外壳异步」是这个方向上的轻量版。

---

## 2. `async def` 的判据：需要挂起点

Python 里 `async def` 的**唯一必要性**是函数体里需要**挂起点**（suspension point）：

| 挂起点 | 形式 |
| --- | --- |
| `await` | `await some_coroutine()` |
| 异步上下文管理器 | `async with some_lock:` |
| 异步迭代 | `async for x in some_aiter:` |

> ⚠️ **一个容易漏的点**：`async with` / `async for` **不是** `Await` 节点
> （AST 里分别是 `AsyncWith` / `AsyncFor`）。所以只用
> `any(isinstance(x, ast.Await))` 去扫，会把「只有 `async with` 的 async 函数」
> 误判成「不该是 async」。本文最早就是这么错的，被 `request_queue.py` 打脸。

### 在 `scheduler.py` 里：恰好就是 `async ⟺ await`

`engine/scheduler.py` 里没有 `async with` / `async for`，所以简单规则在这里
**精确成立**（可用 AST 扫，见 §7.1）：

| 函数 | async? | 含 await? | 为什么 |
| --- | --- | --- | --- |
| `add_request` | ✅ | ✅ | `run_in_executor`（分词）+ `enqueue` |
| `_decode_step` | ✅ | ✅ | `run_in_executor` |
| `_schedule` | ✅ | ✅ | `expire_timed_out` / `dequeue` / `to_thread` |
| `run_loop` | ✅ | ✅ | `asyncio.sleep` + `_schedule` |
| `stop` | ✅ | ✅ | `asyncio.wait_for` |
| `start` | ❌ | ❌ | 只有 `asyncio.create_task()`，不 await |
| `_prefill_chunk_blocking` | ❌ | ❌ | 纯阻塞计算，由**调用方** `to_thread` 派发 |
| `_decode_step_single` | ❌ | ❌ | 同上 |
| `get_metrics` | ❌ | ❌ | 纯拼 dict |

### 在 `request_queue.py` 里：有两个「只有 `async with`」的反例

| 函数 | async? | await | async with | 说明 |
| --- | --- | --- | --- | --- |
| `enqueue` | ✅ | ✅ | ✅ | 先 `await expire_timed_out()` 再加锁 |
| `dequeue` | ✅ | ✅ | ✅ | 同上 |
| **`expire_timed_out`** | ✅ | **❌** | ✅ | **只靠 `async with self._lock` 才需要 async** |
| **`cancel`** | ✅ | **❌** | ✅ | 同上 |
| `stats` / `__len__` / `mark_admitted` | ❌ | ❌ | ❌ | 纯读/写计数器 |

这两个反例正好印证了 §8 的结论：那把锁存在的意义是**声明访问边界**，
而 `asyncio.Lock` 只能在 async 代码里用 —— 于是这两个方法也就跟着成了 async。

---

## 3. `await` ≠ 「不等待」

这是最容易搞反的一点。

> `await` 的意思是：**挂起当前协程，把事件循环让出去**。

所以 `add_request` 自己**确实在等**分词完成 —— 但**事件循环没有在等**。
区别不在「调用方等不等」，而在「**那个唯一的共享资源有没有被占住**」。

### 为什么这件事在本项目里很重要

uvicorn 默认 **一个进程、一个线程、一个事件循环**。这个循环上同时挂着：

- 所有 HTTP handler（每个 `/generate` 请求一个）
- 后台的 `run_loop` 任务（就是真正的推理）
- `/health`、`/metrics` 的 handler

**循环被阻塞 = 这三样全体停摆。** 具体表现：

| 被阻塞期间 | 后果 |
| --- | --- |
| `/health` | 超时 → 编排层的健康检查会认为实例挂了 |
| 新的 `/generate` | 连 TCP 连接都不被 accept |
| `run_loop` 的 decode step | 被推迟 → **正在生成的其他用户一起变慢** |
| `expire_timed_out` | 被推迟 → 超时请求继续占着队列名额 |

所以「放线程池」换来的不是"调用方不等"，而是**"循环不受影响"**。

---

## 4. 实测：`to_thread` 到底换来了什么

用**真实的 Qwen2 tokenizer**（本机已缓存）测。prompt = 3002 token，8 核机器，
心跳协程每 1 ms 醒一次用来观测事件循环是否还活着。

| 并发 | 写法 | 总墙钟 | 加速比 | 心跳 p50 | 心跳 max |
| --- | --- | --- | --- | --- | --- |
| 1 | 同步 | 4 ms | | 4.34 ms | 4.3 ms |
| 1 | `to_thread` | 5 ms | **~1x** ※ | 1.17 ms | 1.3 ms |
| 8 | 同步 | 31 ms | | **31.29 ms** | 31.3 ms |
| 8 | `to_thread` | 13 ms | **2.31x** ↑ | 1.27 ms | 6.9 ms |
| 32 | 同步 | 117 ms | | **117.33 ms** | 117.3 ms |
| 32 | `to_thread` | 37 ms | **3.18x** ↑ | 1.27 ms | 16.5 ms |

※ **只有 n=1 那一行有抖动**：不同轮次跑出过 0.85x（慢 15%）也跑出过 1.1x（快 10%），
都在噪声范围内。所以那一行只能得出「**低并发下没有收益**」，不能得出「一定更慢」。
心跳那一列则是稳定的、每次都复现的。

读法：**同步写法下「心跳 p50 = 总墙钟」**，意思是整个分词期间事件循环完全停摆。

### 收益一：响应性（主要目的）

`to_thread` 之后心跳 p50 恒定在 1.27 ms，**几乎不随并发变化**。
而同步写法是 4 ms → 31 ms → 117 ms 线性恶化。

### 收益二：真并行（只在释放 GIL 时有）

2.31x / 3.18x 说明 **Rust 版 tokenizer 释放了 GIL**，多线程是真并行。
（如果它不释放，这里只会看到 ~1x —— 那就只剩响应性收益。）

注意 32 并发时只到 3.18x：受 8 核 + 线程池 + **GIL 争抢**共同限制。
同一个表里 `to_thread` 的心跳 max 从 1.3 → 6.9 → 16.5 ms 也在涨 —— 因为 32 个线程
抢 GIL 时，连事件循环线程也要排队。

### 代价：并发低时没有收益

并发 = 1 时线程池**没有收益**（实测在 0.85x ~ 1.1x 之间抖，见上表 ※）：
线程调度 + 跨线程唤醒的开销没有任何并发来摊平。

> 所以「有什么优化」的完整答案是：**收益与并发度正相关**。
> 在一个要同时服务 N 个请求 + 1 个后台循环的进程里，它把「循环停摆」换成
> 「循环不受影响」；在单请求脚本里它什么也不换。

---

## 5. 两种静默故障（都不要犯）

`_blocking` 那类函数**必须保持同步**。把被调方改成 `async def`，或者调用方漏掉
`to_thread`，会产生两种**不报错**的退化：

### 模式 A：async 包装 + 直接 await

```python
async def bad_wrapper(ms):
    return sleep_blocking(ms)        # 阻塞体留在事件循环线程里

await bad_wrapper(100)               # 结果完全正确，但循环被独占
```

实测（5 次 × 100 ms 阻塞）：

| 写法 | 心跳 p50 | 心跳 p95 | 心跳 max |
| --- | --- | --- | --- |
| `await asyncio.to_thread(sleep_blocking, 100)` ✅ | 1.2 ms | 1.2 ms | 3.8 ms |
| `await bad_wrapper(100)` ❌ | **262.8 ms** | **315.3 ms** | **315.3 ms** |
| `await asyncio.to_thread(cpu_blocking, 100)` ⚠️ | 13.7 ms | 43.7 ms | 50.5 ms |

⚠️ 第三行是 GIL 边界：`time.sleep` 释放 GIL，所以 `to_thread` 完美解决；
换成**纯 Python CPU 忙等**后，即使丢到线程里也会因抢 GIL 拖慢循环（p50 13.7 ms）。

### 模式 B：`run_in_executor` 传了 async 函数

```python
await loop.run_in_executor(executor, async_worker, 5)
# 实测返回：<coroutine object async_worker at 0x...>   ← 不是结果
# RuntimeWarning: coroutine 'async_worker' was never awaited
```

线程里只拿到一个 **coroutine 对象**，**函数体从未执行**。表现是「序列永远跑不完」，
而那句 warning 要到 **GC 时**才出现，栈定位在 `asyncio/base_events.py` —— 离 bug 很远。

---

## 6. `_blocking` 契约与本仓库的 async 边界

### 命名契约

```
def _prefill_chunk_blocking(self, seq):  ...   # 阻塞的同步函数
def _decode_step_single(self, seq):      ...   # 名字里没有 _blocking，但同样是阻塞体
```

调用点（`async def _schedule` 内）：

```python
await asyncio.to_thread(self._prefill_chunk_blocking, seq)   # scheduler.py:1378
await asyncio.to_thread(self._decode_step_single, seq)       # scheduler.py:1408
```

**两侧必须成对改**：被调方变 async → 模式 B；调用方去掉 `to_thread` → 模式 A。

### 边界清单

| 位置 | 归属 | 理由 |
| --- | --- | --- |
| `add_request` 的分词 | **循环外**（`run_in_executor`） | CPU 密集 + 释放 GIL |
| `_schedule` 的两处计算 | **循环外**（`to_thread`） | 同上 |
| `RequestQueue` 全部方法 | **循环内**（async + lock） | 只碰内存、无阻塞；async 是访问边界声明 |
| `_resolve_sequence_future` / `get_metrics` | **循环内**（同步） | 纯内存操作，快 |
| 模型加载（`app_v2` lifespan） | **循环外**（`run_in_executor`） | 要读几百 MB 权重 |

---

## 7. 自己复现本文的测量

### 7.1 验证「async def ⟺ 需要挂起点」

```bash
cd <fork 的仓库根目录>          # 本 fork 的工作目录
uv run python -c "
import ast, pathlib
for f in ['inference_engine/engine/scheduler.py', 'inference_engine/engine/request_queue.py']:
    print('===', f)
    for n in ast.walk(ast.parse(pathlib.Path(f).read_text())):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_async = isinstance(n, ast.AsyncFunctionDef)
            # 注意：async with / async for 不是 Await 节点！
            need = any(isinstance(x, (ast.Await, ast.AsyncWith, ast.AsyncFor))
                       for x in ast.walk(n))
            mark = 'OK  ' if is_async == need else '!!  '
            print(f'{mark}{n.name:22} async={is_async!s:5} 需要挂起点={need}')
"
```

### 7.2 复现 §4 的响应性/并行对比

```python
import asyncio, time, statistics
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
PROMPT = "Explain paged attention and continuous batching in detail. " * 300

# 两种 handler：唯一区别就是分词在哪个线程里跑
async def handler_sync(p):
    return tok(p)["input_ids"]                      # 就地在事件循环线程里分词

async def handler_thread(p):
    return await asyncio.to_thread(lambda: tok(p)["input_ids"])

async def heartbeat(ticks, stop):
    while True:
        t0 = time.perf_counter()
        await asyncio.sleep(0.001)
        ticks.append((time.perf_counter() - t0) * 1000)
        if stop.is_set():
            return

async def run(handler, n):
    stop, ticks = asyncio.Event(), []
    hb = asyncio.create_task(heartbeat(ticks, stop))
    await asyncio.sleep(0.05); ticks.clear()        # 预热后再开始计数
    t0 = time.perf_counter()
    await asyncio.gather(*[handler(PROMPT) for _ in range(n)])
    wall = (time.perf_counter() - t0) * 1000
    stop.set(); await hb
    s = sorted(ticks)
    return wall, statistics.median(s), max(s)

async def main():
    for n in (1, 8, 32):
        for name, h in (("同步", handler_sync), ("to_thread", handler_thread)):
            w, p50, mx = await run(h, n)
            print(f"{n:>3} {name:>9} 墙钟={w:7.0f}ms  心跳p50={p50:7.2f}ms  心跳max={mx:7.1f}ms")

asyncio.run(main())
```

> 用 `HF_HUB_OFFLINE=1` 跑可避免联网检查（本机已缓存 `Qwen/Qwen2-0.5B`）。

### 7.3 测 `enqueue` 的 async + lock 开销

```python
import asyncio, time
lock = asyncio.Lock()
N = 200_000

lst = []
t0 = time.perf_counter()
for i in range(N):
    lst.append(i)
append_ns = (time.perf_counter() - t0) / N * 1e9

async def locked():
    for i in range(N):
        async with lock:
            lst.append(i)

t0 = time.perf_counter()
asyncio.run(locked())
locked_ns = (time.perf_counter() - t0) / N * 1e9

print(f"裸 append           {append_ns:7.1f} ns")
print(f"async+lock+append   {locked_ns:7.1f} ns")
print(f"净开销              {locked_ns - append_ns:7.1f} ns")
```

（`RequestQueue.enqueue` 本身还要加上 `create_future` ≈ 1.7 µs。）

### 7.4 验证文档里的行号锚点

文档里引用了很多 `文件:行号`。加注释后行号会漂移，用**函数名/特征串**重新核对：

```bash
uv run python -c "
import pathlib
src = pathlib.Path('inference_engine/engine/scheduler.py').read_text().splitlines()
for n, want in [(568,'self.model('), (771,'eos_hit'), (1032,'to_thread'), (1065,'to_thread')]:
    ok = want in src[n-1]
    print(('OK  ' if ok else 'DRIFT ') + f':{n} {src[n-1].strip()[:60]}')
"
```

---

## 8. `enqueue` 的 async：实测与结论

### 开销

| 项 | 数值 |
| --- | --- |
| 裸 `list.append` | 32 ns |
| `async with asyncio.Lock` + append | 451 ns |
| → **async + lock 的净开销** | **≈ 418 ns** |
| `Sequence.create()`（含在 `add_request` 里） | 3 365 ns |
| `await RequestQueue.enqueue()`（队列空） | 2 176 ns（其中 ~1.7 µs 是 `create_future`） |
| 参照：一次 decode forward | **24 000 000 ns** |
| **enqueue 占一步的比例** | **0.009%（0.03‰）** |
| 即使 1000 req/s，enqueue 总 CPU 占用 | 0.218% |

**结论：它不是为了性能。**

### 那为什么还是 async？

代码里只有两个理由：

```python
await self.expire_timed_out()      # ①
async with self._lock:             # ②
```

而 `enqueue` / `dequeue` / `expire_timed_out` / `cancel` **四个方法的临界区里一个
`await` 都没有** —— 所有 await 都在 `async with self._lock` **之前**。

于是关键结论：**单事件循环下这把锁永远不会真正竞争**（acquire→release 之间没有挂起点，
其他协程没机会插进来）。它是「**访问边界的声明**」，不是「互斥实现」。

保留 async 的三个理由（按重要性）：

1. **约定 / 防未来 bug** —— 一旦有人往临界区里加 `await`（写盘、上报指标、换成
   Redis/磁盘队列），当前写法**自动正确**，同步写法会**静默出错**（被交错执行）。
2. **标准模式的遗产** —— `asyncio.Queue.put()` 之所以 async，是因为它**在队列满时会挂起
   等空位**（背压）。本实现选择抛 `QueueFullError` → HTTP 503，所以「等待语义」没了，
   只留下了 async 的外形。
3. **签名稳定性** —— 将来要改成背压语义，不用改所有调用点。

### 那改成同步会怎样？

| 改动 | 结论 |
| --- | --- |
| 只把 `enqueue` 改成同步 `def` | **在当前代码下是正确的**（临界区无 await、队列只在循环里被访问）。省 418 ns |
| 换成 `threading.Lock` | **保护不了任何东西** —— 队列从未被 `to_thread` 访问过 |
| 把整个 `add_request` 改成同步（含分词） | ❌ **这才是真会出事的那个** —— 就是 §4 里「同步」那一列 |

---

## 9. 顺带发现：`expire_timed_out` 是 O(队列深度)

`expire_timed_out()` 会**全表扫描**队列，而 `enqueue` 和 `dequeue` **各调用它一次**：

| 队列深度 | `enqueue` + `dequeue` 耗时 |
| --- | --- |
| 0 | 7.5 µs |
| 10 | 8.8 µs |
| 100 | **23.9 µs（3.2x）** |

好消息：本仓库 `maxsize = max_batch_size * 8 = 32`（`scheduler.py:257`），深度有上界，
扫 32 个元素无所谓。**但如果为了「多缓冲请求」把 `maxsize` 调到几万，`enqueue` 就变成
O(N)**，每秒几千请求时这里会成为热点。（记为 `0003` 的 **F29**。）

---

## 10. 一句话口诀

| 场景 | 怎么写 |
| --- | --- |
| 要等 I/O、要 `await` 别的东西 | `async def` + `await` |
| 纯计算 / 会阻塞几十 ms 以上 | **同步 `def`**（名字可带 `_blocking`），**调用方** `to_thread` |
| 只碰内存、很快 | 同步 `def` |
| 想写 `async def` 但里面没有 `await` | 停下，想想 §2 的判据 —— 大概率不该是 async |
