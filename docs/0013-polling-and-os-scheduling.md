# 1 毫秒之谜：轮询间隔、sleep/yield 与操作系统调度

这篇文档回答一串从 `scheduler.run_loop()` 里那行 `await asyncio.sleep(0.001)` 引发的疑问：

1. 为什么是 **1ms**？为什么不是 0.5ms 或 5ms？是不是约定俗成？
2. 这个值有没有硬约束？**Linux / Windows / macOS 各是什么情况**？最新的 Windows 修了没？
3. `sleep` 这种"让出"动作在实际中**真的有用吗**？操作系统不是按时间片 + 优先级抢占的吗？
   我 `sleep(0)` 一直占着线程，别的任务就真跑不了吗？
4. Java 的 `Thread.sleep` 和 `Object.wait` 到底差在哪？

> 相关：[`0012-async-and-blocking.md`](0012-async-and-blocking.md)（`await` / `to_thread` / GIL）、
> [`0010-phase-index.md`](0010-phase-index.md)（`run_loop` 在哪一层）。

---

## 0. 一页速查

| 问题 | 答案 |
| --- | --- |
| 1ms 的硬约束来自哪 | **Linux 的 `epoll_wait` 只有 1ms 分辨率**，CPython 会把小于 1ms 的请求**向上取整** |
| 0.5ms 有意义吗 | 在 Linux 上**没有**（被取整成 1ms）；在 macOS 上有，但收益看不见 |
| 5ms 为什么不选 | CPU 只省 0.77 个百分点，但**每个新请求**的准入延迟从 1ms 变成 5ms |
| Windows 修了吗 | **部分修了**：Win10 2004 起 `timeBeginPeriod` 变成**按进程**生效（不再拖累全系统）；但默认节拍仍是 ~15.6ms，要 1ms 必须显式申请 |
| `sleep` 真的有用吗 | 有用，但不是"让 OS 有机会调度别人"（OS 本来就会抢占），而是**让自己离开 runqueue**：CPU 占用归零、不跟别人竞争、不吃 cgroup 配额 |
| `sleep(0)` 和 `sleep(1ms)` 差别 | `sleep(0)` = 留在 runqueue 里空转（实测 **22.45% CPU**）；`sleep(1ms)` = 离开 runqueue（**1.39%**） |
| **最关键的一点** | **OS 调度的是线程，不是协程。** 单线程事件循环里，一个不让出的协程会卡死同线程上所有协程，**OS 帮不上任何忙** |
| Java `wait` vs `sleep` | **反直觉**：`wait()` 释放锁 + 让出 CPU；`sleep()` **只让出 CPU、不释放锁** |

---

## 1. 1ms 的三个来源

### 1.1 Linux：`epoll_wait` 的 1ms 分辨率（决定性约束）

CPython 源码 `selectors.py`（`EpollSelector.select`）里就有原文。
文件位置用 `python3 -c "import selectors; print(selectors.__file__)"` 查：

```python
def select(self, timeout=None):
    if timeout is None:
        timeout = -1
    elif timeout <= 0:
        timeout = 0
    else:
        # epoll_wait() has a resolution of 1 millisecond, round away
        # from zero to wait *at least* timeout seconds.
        timeout = math.ceil(timeout * 1e3) * 1e-3      # ← 向上取整到 1ms
```

**这是 Linux 上"1ms"的真正来源**：`epoll_wait()` 的超时参数单位就是毫秒（API 层面），
所以任何小于 1ms 的请求都会被 `math.ceil` 提升到 1ms。

`asyncio` 在 Linux 上默认用 `EpollSelector`，而 `BaseEventLoop._run_once()` 的等待
就是 `self._selector.select(timeout)` —— **整个事件循环的定时器精度在 Linux 上就是 1ms**。

于是：

| 你写 | Linux 实际等 |
| --- | --- |
| `await asyncio.sleep(0.0005)` | **1 ms** |
| `await asyncio.sleep(0.0009)` | **1 ms** |
| `await asyncio.sleep(0.001)` | 1 ms |

想在 Linux 上比 1ms 更细，**只能忙等**（`sleep(0)` 循环）—— 也就是拿 CPU 换精度。

### 1.2 Windows：默认 ~15.6ms，且 2004 之前会拖累全系统

Windows 的默认定时器节拍长期是 **15.6ms**（≈64 Hz）。`Thread.sleep(1)` 在这上面
实际会睡 15ms 左右，除非进程显式调用 `timeBeginPeriod(1)` 把节拍提到 1ms。

**2020 年有个重要变化**：Windows 10 **2004**（2020-04）静默改了 `timeBeginPeriod` 的语义 ——

| | 2004 之前 | 2004 之后 |
| --- | --- | --- |
| 一个进程调用 `timeBeginPeriod(1)` | 提升**全系统**定时器精度，**所有进程**都受益（也都被降频/耗电影响） | **只影响调用进程**；其他进程由内核"模拟"出高频 tick |

这个改动**没有公告、文档也是后来才补**的，直接搞坏了一批依赖"我提频，别人受益"的程序。

> 所以"最新的 Windows 有没有修复这个问题"的准确回答是：
> **修的是"副作用"（不再拖累全系统），不是"默认精度"** —— 默认仍是 ~15.6ms，
> 你自己的进程要 1ms 精度，仍然必须自己申请。

Python 侧还有一个相关事实：Windows 上 `asyncio` 默认用 **`ProactorEventLoop`**（自 3.8 起），
Python 官方文档 *asyncio-platforms* 提到它的 monotonic 时钟分辨率**通常约 15.6ms**
（最好情况 0.5ms，取决于 HPET 与系统配置）。也就是说：

> **Linux 的事件循环精度下限是 1ms，Windows 是 ~15.6ms（要显式申请才能到 1ms）。**
> 跨平台代码里"1ms"是能拿到的**最细的通用值**。

### 1.3 macOS：其实是 ns 级（所以 1ms 是"可移植性选择"）

macOS 用 `kqueue`，超时是 **timespec（纳秒精度）**，没有 Linux 那个取整。
`KqueueSelector.select` 直接把浮点秒传给 `select.kqueue.control(..., timeout)`。
本机实测（§3）确认：请求 0.1ms 真能睡到 0.135ms。

所以 **1ms 不是物理极限，是"可移植性下限"** —— 在 macOS 上你可以更细，
但那会让代码在 Linux 上表现不一致（被静默提升到 1ms）。

---

## 2. 版本时间线（本文结论依赖的版本）

本文的实测都在下面这台机器上跑；跨平台的结论都标了来源版本。

| 项 | 版本 | 说明 |
| --- | --- | --- |
| 本机 OS | **macOS 26.6.2**（Darwin 25.6.0，arm64 / Apple M1） | 提供了 §3 的实测数据 |
| 本机 Python | **3.10.20**（Homebrew） | 实测用；`selectors.py` 引用自这一版 |
| 本机 JDK | **Temurin 21.0.11+10 LTS** | 提供了 §8 的 Java 实测 |
| Linux 调度器 | **EEVDF，自 Linux 6.6 起取代 CFS** | §7 的 base slice 等参数按 EEVDF |
| Linux 实时抢占 | **PREEMPT_RT 于 Linux 6.12 合入主线**（x86_64/arm64/RISC-V 起步） | 不影响本文结论，但说明"抢占粒度"仍在演进 |
| Windows 定时器 | **10 2004 起 `timeBeginPeriod` 按进程生效** | §1.2 |
| Java 规范 | 引用 **Java SE 26** 的 `Object` / JLS 17 文档（当前最新） | `wait` 语义 |

> ⚠️ 一处**可能过时的说法**要避开：讲 Linux 调度时别再说"CFS 的 `sched_latency` 是 6ms"
> 作为**当前**行为 —— 6.6 之后 fair class 是 **EEVDF**，参数也改名了：
> `sched_min_granularity_ns` → **`sched_base_slice_ns`（默认 3ms）**，
> 它会直接决定任务何时被 tick 抢占。

---

## 3. 实测：sleep 精度与空闲轮询代价

### 3.1 `asyncio.sleep` 请求值 vs 实际值（macOS / kqueue）

| 请求 | 实际中位 | 实际最大 |
| --- | --- | --- |
| 0.1 ms | 0.135 ms | 0.190 ms |
| 0.25 ms | 0.305 ms | 0.455 ms |
| 0.5 ms | 0.588 ms | 0.675 ms |
| **1 ms** | **1.152 ms** | 1.198 ms |
| 2 ms | 2.279 ms | 2.325 ms |
| 5 ms | 5.668 ms | 5.751 ms |
| 10 ms | 11.057 ms | 11.109 ms |
| 20 ms | 21.066 ms | 21.124 ms |

**实际总是 ≥ 请求值，约 +10~15%**（macOS 的定时器合并 / timer coalescing）。
所以 `scheduler_poll_interval_ms = 1.0` 的**有效间隔其实接近 1.15ms**。

### 3.2 空闲轮询的 CPU 代价

模拟 `run_loop` 的空闲分支，跑 2 秒，测进程自身 CPU 时间：

| 间隔 | CPU 占用 | 唤醒次数/秒 |
| --- | --- | --- |
| `sleep(0)`（纯让出） | **22.45%** | 71,039 |
| 0.2 ms | 5.12% | 4,011 |
| 0.5 ms | 2.38% | 1,699 |
| **1 ms（本仓库现状）** | **1.39%** | 867 |
| 2 ms | 0.86% | 438 |
| 5 ms | 0.62% | 177 |
| 20 ms | 0.26% | 47 |

复现脚本（见 §10）。

---

## 4. 为什么是 1ms，而不是 0.5ms 或 5ms

先说一个诚实的观察：**这张表是干净的 1/x 曲线，没有明显的"拐点"**。
所以不能说"1ms 是某个最优解"，准确的说法是**它被两个方向的约束夹出来**：

### 往下走（比 1ms 更小）—— 被硬约束挡住

- **Linux 上根本做不到**（§1.1 的 `math.ceil`）：写 0.5ms 得到的就是 1ms。
- 在能做得更细的平台上（macOS），代价陡增：**0.2ms 的 CPU 占用是 1ms 的 3.7 倍**
  （5.12% vs 1.39%），而换来的收益是"把准入延迟从 1ms 降到 0.2ms" ——
  相对 **24ms 的 decode step** 和 **60ms+ 的 TTFT** 完全看不见。

### 往上走（比 1ms 更大）—— 省得少，赔得多

- CPU 只省 **0.77 个百分点**（5ms：1.39% → 0.62%）；
- 但**准入延迟地板线性增长**，而且这是**每个新请求都要付**的：

| | 数值 |
| --- | --- |
| 一次 decode forward | ~24 ms |
| 本仓库实测 TTFT | 59 / 70 / 851 ms（min / median / max） |
| 1ms 轮询的额外准入延迟 | ≤1 ms ≈ **TTFT 的 1.5%** |
| 5ms 轮询 | ≤5 ms ≈ **TTFT 的 8%** |

**结论**：1ms = "Linux 允许的最小值" ∩ "CPU 代价已经不高（1.39%）"。
它不是算出来的，是**默认值**。而 `sleep(0)` 的 22.45% 说明"干脆不睡"是不行的。

---

## 5. 关键：两个层次的调度器

这是你问题里最核心的一点。**必须把两个调度器分开看**：

```
┌──────────────────────────────────────────────────────────┐
│  用户态：事件循环（asyncio 的 Task 调度器）                │
│  ─ 协作式（cooperative）：只有协程主动 await 才会切换      │
│  ─ 运行在一个 OS 线程里                                   │
├──────────────────────────────────────────────────────────┤
│  内核态：操作系统调度器（Linux: EEVDF）                    │
│  ─ 抢占式（preemptive）：按时间片强制切换                 │
│  ─ 调度单位是【线程】，它看不见协程                        │
└──────────────────────────────────────────────────────────┘
```

### 5.1 操作系统调度的是线程，不是协程

内核**完全不知道**协程的存在。从内核看，uvicorn 就是**一个线程**在跑。
这个线程是否让出 CPU，由 EEVDF 按时间片决定；**这个线程内部有几个协程在等，
内核一无所知**。

### 5.2 asyncio 是协作式的：不让出 = 全停

于是就有了这个结论：

> 在一个事件循环线程里，**一个不让出（不 `await`）的协程，会卡死同线程上所有协程，
> 而操作系统帮不上任何忙** —— 因为从内核看，这个线程正在**忙着干活**（CPU 100%），
> 它没有任何理由去抢占它。

这正是 `0012` 里那个实验的含义：`await bad_wrapper(100)` 让心跳从 1ms 变成 315ms。
**OS 抢占救不了它。** 抢占只能到"线程"粒度。

### 5.3 所以 `to_thread` 的真正作用

`asyncio.to_thread` 把一个协程的活儿**变成另一个 OS 线程**。这样：

- 内核现在**看得见**两份工作了 → 它可以真的并行调度它们；
- 事件循环线程可以继续跑别的协程。

> 换句话说：**`to_thread` 的作用是把"用户态调度问题"转化成"内核态调度问题"** ——
> 后者内核才会帮你。

---

## 6. `sleep` 在操作系统里到底做了什么

### 6.1 线程状态机

Linux 里线程（task）的关键状态：

| 状态 | 含义 | 在 runqueue 里？ |
| --- | --- | --- |
| `TASK_RUNNING` | 正在跑 **或** 就绪等 CPU | ✅ 在 |
| `TASK_INTERRUPTIBLE` | 睡着，等事件（信号可唤醒） | ❌ 不在 |
| `TASK_UNINTERRUPTIBLE` | 睡着，不可被信号打断（如等磁盘） | ❌ 不在 |

**关键就是"在不在 runqueue 里"。**

### 6.2 `nanosleep` 的完整路径

```
thread: nanosleep(1ms)
   ↓ 系统调用
kernel: 把 thread 状态设为 TASK_INTERRUPTIBLE
       把它从 runqueue 摘出来
       挂一个 hrtimer（高精度定时器，ns 级）到 1ms 后
   ↓
CPU:    【这个线程此刻完全不在 CPU 上，也不参与调度竞争】
       别的线程随便用这个核
   ↓ 1ms 后
kernel: hrtimer 到期 → 唤醒线程 → 重新放回 runqueue
   ↓
thread: 变成 TASK_RUNNING，等 EEVDF 挑中它
```

所以 **sleep 期间 CPU 占用是真的 0**（不是"低"，是不在 runqueue 里）。

### 6.3 `sleep(0)` / `yield()` 的区别 —— 这解释了 22.45%

`sleep(0)`（或 `sched_yield()`）**不改变状态**：线程**仍然留在 runqueue 里**，
只是"让一下"。效果是：

| | `sleep(0)` / `yield()` | `sleep(1ms)` |
| --- | --- | --- |
| 是否离开 runqueue | ❌ 留下 | ✅ 离开 |
| CPU 占用 | **实测 22.45%** | **1.39%** |
| 是否和别人竞争 CPU | ✅ 一直竞争 | ❌ 不竞争 |
| 别的线程看到的 runqueue 长度 | 一直 +1 | 短暂 +1 |

注意 22.45% 不是 100%，因为 `yield` 之后马上又被挑中，中间还夹着事件循环自己的开销 ——
但它是**实打实的 CPU 时间**。这就是"1ms 的存在意义"：**把线程彻底移出竞争**。

---

## 7. Linux 到底怎么调度（回答"抢占不就行了吗"）

### 7.1 完全抢占 ≠ 零延迟

Linux 自 2.6 起就是**完全抢占内核**（`CONFIG_PREEMPT` 系）。所以是的，
**即使你 `sleep(0)` 死循环，别的线程也能跑** —— 你的直觉在这一点上是对的。

但"能跑"和"跑得好"是两回事。抢占的粒度由时间片决定，而 **Linux 6.6+ 的 EEVDF 里，
fair class 任务的默认 base slice 是 `sched_base_slice_ns = 3ms`**：

- 一个一直 `TASK_RUNNING` 的线程，可以连续占用 CPU **最多约 3ms** 才被 tick 抢占；
- 而且它一直待在 runqueue 里 → 每次调度都要参与竞争 → 别人被挑中的**延迟和抖动都变大**。

在旧 CFS 下这个参数叫 `sched_latency`（按 `nr_running` 缩放）/
`sched_min_granularity_ns`；**讲"当前 Linux"时应该用 EEVDF 的说法**（见 §2 的提醒）。

### 7.2 那为什么还是要 sleep —— 四个真实理由

| # | 理由 | 具体后果 |
| --- | --- | --- |
| 1 | **CPU 时间是共享资源** | 空闲时烧 1.39% 或 22.45%，就是白占；多租户机器上直接变成邻居的延迟 |
| 2 | **runqueue 竞争 → 抖动** | 始终 runnable 的线程抬高别人的调度延迟，尤其是 RT 任务和延迟敏感的请求 |
| 3 | **cgroup 配额**（容器场景） | K8s 里 `cpu.max` 是按**整个 cgroup** 算的；一个空转线程吃的配额会让**同容器里所有线程**被 throttling |
| 4 | **功耗 / 频率 / 成本** | 空闲空转阻止 CPU 进 C-state，云上按核计费更是直接花钱 |

所以"OS 会抢占，所以 sleep 无所谓"这个推理的漏洞是：
**抢占保证的是"别人最终能跑"，不是"别人跑得好"，也不是"你不浪费资源"。**

### 7.3 但抢占对"单线程事件循环"完全无效

回到 §5：抢占的粒度是**线程**。如果那一份工作是一个**不让出的协程**，
内核看到的是"这个线程在认真干活" —— 它会**继续给它时间片**。
同线程上其他协程只能等。

> **这就是为什么 `sleep()` 在单线程事件循环里有意义，但它治不了"协程不让出"的病。**
> 治那个病只有两条路：协程主动 `await`（协作式），或者 `to_thread`（变成多线程，交给内核）。

---

## 8. Java 对照：`wait` vs `sleep`（含本地实测）

你的记忆是**反的** —— `wait()` 让出的东西**比 `sleep()` 更多**：

| | `Thread.sleep(ms)` | `Object.wait()` |
| --- | --- | --- |
| 释放 CPU | ✅ 进 `TIMED_WAITING` | ✅ 进 `WAITING` |
| **释放 monitor 锁** | ❌ **不释放** | ✅ **释放**（被 notify 后重新获取） |
| 调用前置条件 | 无 | **必须持有该对象的 monitor**，否则 `IllegalMonitorStateException` |
| 谁来唤醒 | 定时器到点 | `notify()` / `notifyAll()` / 超时 / 中断 |
| 位置 | `Thread` 静态方法 | `Object` 实例方法 |

本机 **Temurin 21.0.11 LTS** 实测（源码见 §10）：

```
① sleep(500) 期间，别人拿锁耗时 = 450 ms   (≈450 说明锁没释放)
② wait()  期间，别人拿锁耗时   =   0 ms   (≈0  说明锁已释放)
③ Thread.sleep(1) 实际中位 = 1.256 ms, 最大 = 1.308 ms
```

另外两个 Java 侧的"等待间隔"参考：

- **`LockSupport.parkNanos(1)`** —— 常常**几乎立即返回**（内部有短暂自旋），
  所以它常被用作"廉价让出"，而不是"精确睡 1ns"。
- **LMAX Disruptor 的等待策略表** —— Java 世界对"等待间隔怎么选"的系统化回答：

  | 策略 | 机制 | 延迟 | CPU |
  | --- | --- | --- | --- |
  | `BusySpinWaitStrategy` | 死循环 | 最低 | 烧满一核 |
  | `YieldingWaitStrategy` | `Thread.yield()` 循环 | 低 | 高 |
  | `SleepingWaitStrategy` | 自旋 → yield → `parkNanos(1)` 渐进退避 | 中 | 低 |
  | `BlockingWaitStrategy` | `Condition` 等待（**事件驱动**） | 有抖动 | **最低** |

- **Netty 的 `NioEventLoop`** 是这几条路里"最优解"的样板：
  `selector.select(timeout)` **配一个 `wakeup()`** —— 有人提交任务就唤醒它。
  **因为有唤醒机制，它的 timeout 可以设到 1 秒**，完全不必靠轮询。

---

## 9. 对本仓库的结论

```python
# scheduler.py  run_loop()
while not self._stop_event.is_set():
    if not self.running and not self.swapped_out and len(self.request_queue) == 0:
        await asyncio.sleep(idle_sleep_s)     # 默认 1ms → 空闲时 1.39% CPU + 867 次/秒唤醒
        continue
    await self._schedule()
```

两个代价：

1. **空闲时纯烧 CPU**（1.39%，零产出）；
2. **每个新请求最多等 1ms 才被 admit** —— 这段直接加在 TTFT 上。

而这两件事**都可以是 0**：把轮询换成"事件驱动 + 定时器兜底"。

```python
# 事件驱动版（等价于 Netty 的 wakeup()）
async def run_loop(self):
    while not self._stop_event.is_set():
        if self._idle():
            try:
                await asyncio.wait_for(self._work.wait(), timeout=idle_sleep_s)
            except asyncio.TimeoutError:
                pass                      # 定时兜底：检查超时 / 试 swap-in
            self._work.clear()
        await self._schedule()

# add_request 末尾
self._work.set()                          # 叫醒
```

**为什么不能简单换成 `asyncio.Queue` + `await queue.get()`？**
因为 `run_loop` 需要被**三个**来源唤醒，不只是新请求：

| 唤醒源 | 谁触发 | 定时兜底能覆盖吗 |
| --- | --- | --- |
| 新请求入队 | `add_request` | ❌ 必须事件唤醒（否则又变轮询） |
| 请求超时过期 | `expire_timed_out` | ✅ 定时器 |
| swap-in 重试 | 设备块腾出 | ✅ 定时器 |

所以正确写法是 **"新请求用事件唤醒 + 超时/swap-in 用定时器兜底"** ——
也就是上面那个 `wait_for(event, timeout=...)` 的版本。
收益：空闲 CPU → ~0，准入延迟 → ~0。（记为 `0003` 的 **F30**。）

---

## 10. 复现本文的测量

### 10.1 Python：sleep 精度 + 空闲轮询代价

```python
import asyncio, time

async def measure_sleep(req):
    errs = []
    for _ in range(200):
        t0 = time.perf_counter()
        await asyncio.sleep(req)
        errs.append((time.perf_counter() - t0) * 1000)
    errs.sort()
    return errs[len(errs)//2], errs[-1]

async def idle_cost(interval, seconds=2.0):
    end = time.perf_counter() + seconds
    wakes = 0
    c0 = time.process_time()
    while time.perf_counter() < end:
        if interval > 0:
            await asyncio.sleep(interval)
        else:
            await asyncio.sleep(0)
        wakes += 1
    return (time.process_time() - c0) / seconds * 100, wakes / seconds

async def main():
    for req in (0.0005, 0.001, 0.005):
        p50, mx = await measure_sleep(req)
        print(f"请求 {req*1000:5.2f}ms → 实际 {p50:6.3f}ms (max {mx:6.3f})")
    for iv in (0, 0.001, 0.005):
        cpu, wps = await idle_cost(iv)
        print(f"间隔 {iv*1000:5.2f}ms → CPU {cpu:5.2f}%  唤醒 {wps:,.0f}/s")

asyncio.run(main())
```

### 10.2 Java：`wait` vs `sleep`

```java
public class SleepVsWait {
    static final Object LOCK = new Object();
    static long timeToGrabLock() {
        long t0 = System.nanoTime();
        synchronized (LOCK) { }
        return (System.nanoTime() - t0) / 1_000_000;
    }
    public static void main(String[] a) throws Exception {
        Thread sleeper = new Thread(() -> {
            synchronized (LOCK) { try { Thread.sleep(500); } catch (InterruptedException e) { } }
        });
        sleeper.start(); Thread.sleep(50);
        System.out.println("① sleep 期间拿锁 = " + timeToGrabLock() + " ms");   // ≈450
        sleeper.join();

        Thread waiter = new Thread(() -> {
            synchronized (LOCK) { try { LOCK.wait(); } catch (InterruptedException e) { } }
        });
        waiter.start(); Thread.sleep(50);
        long t0 = System.nanoTime();
        synchronized (LOCK) { LOCK.notifyAll(); }
        System.out.println("② wait 期间拿锁 = " + (System.nanoTime()-t0)/1_000_000 + " ms");  // ≈0
        waiter.join();
    }
}
```

```bash
javac SleepVsWait.java && java SleepVsWait
```

### 10.3 看 Linux 上的真实调度参数

```bash
# EEVDF 的 base slice（6.6+；旧内核里这里叫 sched_min_granularity_ns）
cat /proc/sys/kernel/sched_base_slice_ns 2>/dev/null || \
  cat /proc/sys/kernel/sched_min_granularity_ns

# 观察一个进程的状态（R=在 runqueue, S=interruptible sleep）
ps -eo pid,stat,comm | head

# 看自己被 cgroup 节流了多少（容器里很关键）
cat /sys/fs/cgroup/cpu.stat 2>/dev/null | grep -E "throttled|nr_throttled"
```

---

## 11. 一句话口诀

| 你想说的 | 应该说 |
| --- | --- |
| "1ms 是拍脑袋定的" | 不是：**Linux 的 `epoll_wait` 只给到 1ms**，CPython 会向上取整 |
| "0.5ms 更精确" | 在 Linux 上**是同一个值**（被 `math.ceil` 抬到 1ms） |
| "OS 会抢占，所以 sleep 无所谓" | 抢占保证"别人最终能跑"，不保证"别人跑得好"，也不阻止你**白烧 CPU 和 cgroup 配额** |
| "单线程里 OS 会公平调度我的协程" | **不会**。内核只看见一个线程；**协程不让出 = 全停，抢占救不了** |
| "`wait` 不让出控制权" | 反了：**`wait` 让出 CPU 且释放锁**；`sleep` 只让出 CPU、**不释放锁** |
| "`sleep(0)` 等于 `sleep(1ms)`" | 差别巨大：22.45% vs 1.39% CPU，前者**一直待在 runqueue 里竞争** |
