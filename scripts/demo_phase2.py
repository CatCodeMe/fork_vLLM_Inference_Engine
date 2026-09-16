"""
scripts/demo_phase2.py — Phase 2 连续批处理实验/演示脚本。

对正在运行的 Phase 2 服务器并发发送 N 个请求，测量：
  - 总墙钟时间、聚合吞吐（tokens/s）
  - 每个请求的 TTFT 与总时延
  - 从 /metrics 读取 batch_size_over_time，直观看到调度器“同时跑多个序列”

前置：先启动 Phase 2 服务器（:8001）
    uv run python scripts/serve.py --phase 2
    # 或在 VSCode 里 F5 -> "Serve: Phase 2 continuous batching"

用法：
    uv run python scripts/demo_phase2.py
    uv run python scripts/demo_phase2.py --n 8 --max-new-tokens 32
    uv run python scripts/demo_phase2.py --n 16 --plot   # 额外保存 batch size 曲线图
    uv run python scripts/demo_phase2.py --trace-raw     # 轨迹不折叠（看每个 chunk）

跑完会自动拉取服务端的 `GET /trace`（调度轨迹环形缓冲区）并打一份带标号的摘要
—— 不需要去翻 server.log。两个标号各管一事：

    N/8         在【请求生命周期】里的位置（固定 8 步）：
                1/8 收到 → 2/8 入队 → 3/8 step/swap → 4/8 admit
                → 5/8 prefill → 6/8 decode → 7/8 finish → 8/8 返回
    [step NNN]  在【调度器】里的第几次循环（自己递增，看谁跟谁在同一次调度里）

阶段会按 N/8 上色（prefill 青、decode 绿、finish 黄……）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from datetime import datetime

import httpx

from _plot_common import save, setup_style

PROMPTS = [
    "Explain the concept of entropy in thermodynamics.",
    "What are the key differences between supervised and unsupervised learning?",
    "Describe the role of the prefrontal cortex in decision making.",
    "Summarize the causes of World War I in three sentences.",
    "How does a transformer architecture work in natural language processing?",
    "What is the significance of the Pythagorean theorem in geometry?",
    "Explain how HTTPS certificates establish trust on the internet.",
    "Describe the process of photosynthesis at the molecular level.",
    "Why is the speed of light constant in all reference frames?",
    "What is the difference between a process and a thread?",
    "Explain the CAP theorem in distributed systems.",
    "How do vaccines train the immune system?",
]


# 用于造长 prompt 的模板句：实测在 Qwen2 tokenizer 下 ≈ 10 个 token
_LONG_BASE = "Explain paged attention and continuous batching in detail. "


def build_prompts(n: int, long_tokens: int | None) -> list[str]:
    """生成本次要发的 n 个 prompt。

    long_tokens=None → 用内置的 12 个短 prompt（实测每个只有 8~13 个 token）
    long_tokens=N    → 用模板句重复到约 N 个 **token**

    [LEARN] 注意 `prefill_chunk_size` 的**单位是 prompt token 的个数**，
            不是字符数、也不是词数。所以内置短 prompt（8~13 token）永远只会有
            1 个 chunk；想让分块 prefill 跑起来，prompt 必须超过 chunk_size 个 token。
    """
    if not long_tokens:
        return [PROMPTS[i % len(PROMPTS)] for i in range(n)]
    reps = max(1, round(long_tokens / 10))          # 模板句 ≈ 10 token
    approx = reps * 10
    print(f"[prompt] 构造长 prompt：模板句 ×{reps} ≈ {approx} 个 token"
          f"（真实 token 数由服务端 tokenizer 决定，看轨迹里的 prompt_len）")
    return [_LONG_BASE * reps for _ in range(n)]


async def wait_for_server(client: httpx.AsyncClient, url: str, wait_s: float) -> dict | None:
    """等 /health 返回 200；超时则打印排查步骤并返回 None。

    [GOTCHA] 「Connection refused (errno 61)」**不等于服务没起**。
    uvicorn 的 Server.startup() 是：

        await self.lifespan.startup()      # ← 模型加载在这里（阻塞几秒~几十秒）
        ...
        server = await loop.create_server(...)   # ← 端口到这里才 bind

    所以**模型加载期间端口根本没打开**，curl 得到的是 refused 而不是超时。
    实测本机 Qwen2-0.5B（模型已缓存、HF_HUB_OFFLINE=1）这个窗口 ≈ 5s；
    首次下载模型、或走代理时会显著更长。

    两种失败要分清：
      · Connection refused  → 端口没监听（没起 / 还在加载）
      · 超时 / 卡住         → 端口在监听，但应用忙或没响应
    """
    t0 = time.perf_counter()
    last = ""
    while True:
        try:
            r = await client.get(f"{url}/health", timeout=5.0)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}（503=还在加载模型）"
        except httpx.ConnectError:
            last = "Connection refused：端口未监听（没起，或模型还在加载）"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"

        waited = time.perf_counter() - t0
        if waited >= wait_s:
            print(f"\nERROR: 等了 {waited:.0f}s 仍连不上 {url}")
            print(f"  最后一次失败：{last}")
            print("  排查：")
            print("    1) 服务起了吗？模型加载中端口不监听 → Connection refused 属正常，再等等")
            print("    2) 端口对吗？Phase 1=8000，Phase 2=8001；serve.py --port N 就对应 N")
            print("    3) 服务端日志该出现 'Application startup complete' / 'Uvicorn running on'")
            print("    4) 手动确认：curl -s " + url + "/health")
            print("  → 或加大等待：--wait 180")
            return None
        print(f"  … 等待服务就绪（{waited:4.1f}s / {wait_s:.0f}s）：{last}")
        await asyncio.sleep(1.0)


async def fire_one(client: httpx.AsyncClient, url: str, prompt: str, max_new: int) -> dict:
    t0 = time.perf_counter()
    resp = await client.post(
        f"{url}/generate",
        json={"prompt": prompt, "max_new_tokens": max_new},
        timeout=300.0,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if resp.status_code == 200:
        body = resp.json()
        return {
            "ok": True,
            "seq_id": body.get("seq_id"),
            "finish_reason": body.get("finish_reason"),
            "ttft_ms": body.get("ttft_ms"),
            "latency_ms": body.get("total_latency_ms", elapsed_ms),
            "tokens": body.get("generated_tokens"),
            "queue_wait_ms": body.get("queue_wait_time_ms"),
        }
    return {"ok": False, "status": resp.status_code, "latency_ms": elapsed_ms}


async def run(url: str, n: int, max_new: int, model: str | None,
              with_trace: bool = True, trace_limit: int = 200,
              trace_raw: bool = False, color: bool = True,
              with_spans: bool = True, wait_s: float = 60.0,
              long_tokens: int | None = None, show_all: bool = False) -> None:
    print(f"\n{'='*64}")
    print(f"  Phase 2 continuous batching demo")
    print(f"  server={url}  concurrent_requests={n}  max_new_tokens={max_new}")
    if model:
        print(f"  model={model}")
    print(f"{'='*64}\n")

    # 健康检查（会等：模型加载期间端口是不监听的，见 wait_for_server 的注释）
    async with httpx.AsyncClient() as client:
        info = await wait_for_server(client, url, args.wait)
        if info is None:
            return
        print(f"[health] model={info.get('model')} device={info.get('device')} "
              f"max_batch_size={info.get('max_batch_size')}")

        prompts = build_prompts(n, long_tokens)
        print(f"[benchmark] 并发发送 {n} 个请求 …")
        t_wall = time.perf_counter()
        results = await asyncio.gather(
            *(fire_one(client, url, p, max_new) for p in prompts)
        )
        wall_s = time.perf_counter() - t_wall

        # 读取调度器遥测
        metrics = (await client.get(f"{url}/metrics", timeout=10.0)).json()
        # 读取调度轨迹（事件日志）与 span 树
        trace = spans = None
        if with_trace:
            try:
                trace = (await client.get(
                    f"{url}/trace", params={"limit": trace_limit}, timeout=10.0
                )).json()
            except Exception as exc:  # noqa: BLE001
                print(f"  （/trace 拉取失败：{exc}）")
        if with_spans:
            try:
                spans = (await client.get(
                    f"{url}/spans", params={"limit": 50}, timeout=10.0
                )).json()
            except Exception as exc:  # noqa: BLE001
                print(f"  （/spans 拉取失败：{exc}）")

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    total_tokens = sum(r["tokens"] or 0 for r in ok)

    # [TRACE] 每请求摘要：seq 前缀和 [Snnn]/[REQ] 轨迹里的完全一致，可以对着 grep
    if ok:
        print(f"\n[本批请求] 按完成顺序给出 seq 前缀（可直接对照下面的调度轨迹）")
        # [GOTCHA] 服务端的 ttft_ms **不含排队时间**（它 = 首 token − prefill 开始），
        #          所以这里额外算一列"真实 TTFT" = 排队 + ttft（= 到达 → 首 token）。
        print(f"  {'#':>2}  {'seq':>9}  {'排队':>8}  {'prefill':>8}  {'真实TTFT':>9}  "
              f"{'总时延':>9}  {'tokens':>6}  finish_reason")
        for i, r in enumerate(sorted(ok, key=lambda x: x["latency_ms"] or 0), 1):
            q, t = r["queue_wait_ms"], r["ttft_ms"]
            real = (q or 0) + (t or 0) if (q is not None and t is not None) else None
            print(f"  {i:>2}  {(r['seq_id'] or '?')[:8]:>9}  "
                  f"{_fmt_ms(q):>8}  {_fmt_ms(t):>8}  {_fmt_ms(real):>9}  "
                  f"{_fmt_ms(r['latency_ms']):>9}  {r['tokens']:>6}  {r['finish_reason']}")

    print(f"\n[结果]")
    print(f"  成功 / 失败        : {len(ok)} / {len(failed)}")
    print(f"  总墙钟时间          : {wall_s:.2f} s")
    if total_tokens:
        print(f"  聚合吞吐            : {total_tokens / wall_s:.1f} tokens/s")
    if ok:
        ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
        lats = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]
        if ttfts:
            print(f"  TTFT  min/median/max: "
                  f"{min(ttfts):.0f} / {statistics.median(ttfts):.0f} / {max(ttfts):.0f} ms")
        if lats:
            print(f"  总时延 min/median/max: "
                  f"{min(lats):.0f} / {statistics.median(lats):.0f} / {max(lats):.0f} ms")
    if failed:
        codes: dict[int, int] = {}
        for r in failed:
            codes[r.get("status")] = codes.get(r.get("status"), 0) + 1
        print(f"  失败状态码          : {codes}")

    # 调度器批次曲线
    sched = metrics.get("scheduler", {})
    bso = sched.get("batch_size_over_time", [])
    sizes = [int(s) for _, s in bso]
    system = metrics.get("system", {})
    print(f"\n[调度器 /metrics]")
    print(f"  requests_finished_total : {system.get('requests_finished_total')}")
    print(f"  throughput_tokens_per_sec: {system.get('throughput_tokens_per_sec'):.1f}")
    print(f"  requests_swapped_total  : {system.get('requests_swapped_total')}")
    if sizes:
        # 只看请求发生这段时间窗内的采样
        print(f"  batch_size_over_time    : {len(sizes)} 个采样, "
              f"峰值 running={max(sizes)}")
        print(f"  running 序列（ascii）    : {ascii_bar(sizes)}")
    else:
        print("  batch_size_over_time    : （无数据）")

    # [TRACE] 只保留本次运行的记录（服务端缓冲区是累积的，见 _filter_trace）
    wanted = {r["seq_id"][:8] for r in ok if r.get("seq_id")}
    if spans and not show_all:
        spans = dict(spans)
        spans["spans"] = [x for x in spans["spans"] if x["seq"] in wanted]
    if trace and not show_all:
        kept, dropped = _filter_trace(trace["entries"], wanted, len(ok), show_all)
        if dropped:
            print(f"  （轨迹已按本次 {len(ok)} 个请求过滤，"
                  f"忽略了服务端缓冲区里的 {dropped} 条历史记录；--trace-all 可看全部）")
        trace = dict(trace, entries=kept, returned=len(kept))

    # [TRACE] span 树 → 文本瀑布图（先看"形状"）
    if spans:
        print_waterfall(spans.get("spans", []), color=color)
    # [TRACE] 事件日志 → 逐行标号（再看"细节"）
    if trace:
        print_trace(trace, raw=trace_raw, color=color)

    return sizes


def _fmt_ms(v) -> str:
    return "-" if v is None else f"{v:.0f}ms"


# ── 轨迹着色（只影响控制台；重定向/管道/NO_COLOR 时自动关）───────────────────

_ANSI = {"dim": "2", "red": "31", "green": "32", "yellow": "33",
         "blue": "34", "magenta": "35", "cyan": "36"}
# 按“在生命周期里的第几步”上色
_STAGE_COLOR = [
    ("swap-out", "red"), ("swap-in", "blue"),
    ("3/8", "blue"), ("4/8", "magenta"), ("5/8", "cyan"),
    ("6/8", "green"), ("7/8", "yellow"),
    ("1/8", "dim"), ("2/8", "dim"), ("8/8", "dim"),
]


def _use_color(flag: bool | None) -> bool:
    if flag is False:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if flag is True:
        return True
    return sys.stdout.isatty()


def _paint(text: str, color: str | None, enabled: bool) -> str:
    if not enabled or not color:
        return text
    return f"\033[{_ANSI[color]}m{text}\033[0m"


def _stage_color(stage: str) -> str | None:
    for prefix, color in _STAGE_COLOR:
        if prefix in stage:
            return color
    return None


def _filter_trace(entries: list[dict], wanted: set[str], n_requests: int,
                  show_all: bool) -> tuple[list[dict], int]:
    """只保留"本次运行"的轨迹条目。

    [GOTCHA] `/trace` 和 `/spans` 的数据源是 scheduler 实例上的环形缓冲区，
             **活得和服务器进程一样久** —— 不重启服务，历史记录就一直累积。
             所以 --n 1 也会看到上一次跑留下的 seq（不是脚本发重了）。
             这里按本次返回的 seq_id 过滤；想看原始缓冲区用 --trace-all。

    保留规则：
      · 带 seq 的条目      → seq 前缀在本次集合里
      · `[REQ]` 无 seq（⓪ 收到）→ 只保留最后 n_requests 条
      · `[step NNN]` 无 seq（② step 开始 → step 级事件）→ 保留 step 落在本次区间内的
    """
    if show_all:
        return entries, 0

    marked = set()
    for i in range(len(entries) - 1, -1, -1):
        if len(marked) >= n_requests:
            break
        e = entries[i]
        if e.get("seq") is None and e["stage"].startswith("1/8"):
            marked.add(i)

    steps = [e["step"] for e in entries
             if e.get("seq") and e["seq"][:8] in wanted and e.get("step") is not None]
    lo, hi = (min(steps), max(steps)) if steps else (None, None)

    out = []
    for i, e in enumerate(entries):
        if i in marked:
            out.append(e)
        elif e.get("seq") is not None:
            if e["seq"][:8] in wanted:
                out.append(e)
        elif e.get("step") is not None and lo is not None and lo <= e["step"] <= hi:
            out.append(e)
    return out, len(entries) - len(out)


def print_waterfall(spans: list[dict], width: int = 60, color: bool = True) -> None:
    """把 /spans 的 span 树画成文本瀑布图（OTel trace 的那种观感）。

    每行一个请求，横轴是相对该请求 arrival 的时间：
        q = queue_wait   P = prefill（`|` 标出 chunk 边界）   d = decode   · = 空白
    """
    if not spans:
        print("\n[请求瀑布图]  （无数据）")
        return
    tmax = max(s["dur_ms"] for s in spans) or 1.0
    print(f"\n[请求瀑布图]  横轴 = 相对 arrival；最长 {tmax:.0f}ms；每格 "
          f"{tmax / width:.1f}ms")
    print("  图例：q=排队  P=prefill（| 为 chunk 边界）  d=decode")

    for i, s in enumerate(spans, 1):
        cells: list[tuple[str, str | None]] = [("·", None)] * width

        def put(start_ms: float, dur_ms: float, ch: str, col: str | None) -> None:
            a = int(start_ms / tmax * width)
            b = max(a + 1, int((start_ms + dur_ms) / tmax * width))
            for k in range(a, min(b, width)):
                cells[k] = (ch, col)

        for seg in s["children"]:
            if seg["name"] == "queue_wait":
                put(seg["start_ms"], seg["dur_ms"], "q", "dim")
            elif seg["name"] == "prefill":
                put(seg["start_ms"], seg["dur_ms"], "P", "cyan")
                off = seg["start_ms"]
                for c in seg["children"]:
                    off += c["dur_ms"]
                    k = int(off / tmax * width)
                    if 0 <= k < width:
                        cells[k] = ("|", "blue")
            elif seg["name"] == "decode":
                put(seg["start_ms"], seg["dur_ms"], "d", "green")

        # 合并连续同色段，减少转义码
        bar, run, run_col = "", "", None
        for ch, col in cells + [("", None)]:
            if col != run_col:
                bar += _paint(run, run_col, color)
                run, run_col = "", col
            run += ch

        parts = []
        for seg in s["children"]:
            if seg["name"] == "prefill":
                parts.append(f"prefill {seg['dur_ms']:.0f}ms/{seg['attrs']['chunks']}chunk")
            elif seg["name"] == "decode":
                parts.append(f"decode {seg['dur_ms']:.0f}ms/{seg['attrs']['tokens']}tok")
            else:
                parts.append(f"queue {seg['dur_ms']:.0f}ms")
        print(f"  #{i:<2} {s['seq']}  {bar}  {'  '.join(parts)}  总 {s['dur_ms']:.0f}ms")


def print_trace(trace: dict, raw: bool = False, fold_after: int = 3,
                color: bool = True) -> None:
    """把 /trace 的条目打成人能读的轨迹。

    两个标号各管一事：
      N/8         —— 在【一个请求的生命周期】里的位置（固定 8 步）
      [step NNN]  —— 在【调度器】里的第几次循环（自己递增）

    折叠规则：连续的「同一序列的中间 chunk」行（5/8）超过 fold_after 条时，
    只留前后各一条，中间用一行省略号代替 —— 长 prompt 会刷屏。
    """
    entries = trace.get("entries", [])
    if not entries:
        print("\n[调度轨迹]  （缓冲区为空）")
        return
    print(f"\n[调度轨迹]  共 {trace.get('returned')} 条 / 服务端缓冲 {trace.get('buffered')} 条"
          f"（当前 step={trace.get('step')}）")
    print("  编号 N/8    = 在【请求生命周期】里的位置（固定 8 步，看它在干什么）")
    print("  [step NNN]  = 在【调度器】里的第几次循环（看谁跟谁在同一次调度里）")
    print("  1/8 收到  → 2/8 入队 → 3/8 step → 4/8 admit → 5/8 prefill "
          "→ 6/8 decode → 7/8 finish → 8/8 返回")

    def line(e: dict) -> str:
        tag = f"{e['tag']:>12}"
        stage = _paint(f"{e['stage']:<12}", _stage_color(e["stage"]), color)
        return f"  {tag} {stage} {e['msg']}"

    def is_mid_chunk(e: dict) -> bool:
        return "chunk=[" in e["msg"] and "✅" not in e["msg"]

    i, printed = 0, 0
    while i < len(entries):
        e = entries[i]
        if not raw and is_mid_chunk(e):
            j = i
            while (j + 1 < len(entries) and is_mid_chunk(entries[j + 1])
                   and entries[j + 1]["seq"] == e["seq"]):
                j += 1
            n_run = j - i + 1
            if n_run > fold_after:
                print(line(e))
                print(f"        {'':>12} … 另 {n_run - 2} 段同类 chunk 已省略"
                      f"（--trace-raw 看全部）…")
                print(line(entries[j]))
                printed += 2
            else:
                for k in range(i, j + 1):
                    print(line(entries[k]))
                    printed += 1
            i = j + 1
            continue
        print(line(e))
        printed += 1
        i += 1
    if printed != len(entries):
        print(f"  （折叠后 {printed} 行）")


def ascii_bar(values: list[int], width: int = 72) -> str:
    """把一串 batch size 采样压成一行字符示意。"""
    if not values:
        return ""
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    max_v = max(values) or 1
    blocks = " ▁▂▃▄▅▆▇█"
    return "".join(blocks[min(len(blocks) - 1, round(v / max_v * (len(blocks) - 1)))]
                   for v in values)


async def main_async(args: argparse.Namespace) -> None:
    sizes = await run(args.url, args.n, args.max_new_tokens, args.model,
                      with_trace=args.trace, trace_limit=args.trace_limit,
                      trace_raw=args.trace_raw, color=_use_color(args.color),
                      with_spans=args.spans, wait_s=args.wait,
                      long_tokens=args.long_prompt, show_all=args.trace_all)
    if args.plot and sizes:
        setup_style()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(range(len(sizes)), sizes, drawstyle="steps-post", color="#2ca02c")
        ax.set_xlabel("scheduler step (sample index)")
        ax.set_ylabel("running sequences")
        ax.set_title(f"Phase 2 batch size over time (n={args.n})")
        ax.grid(True, alpha=0.3)
        save(fig, "demo_batch_size_over_time")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2 continuous batching demo")
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--n", type=int, default=8, help="并发请求数")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--model", default=None, help="仅用于显示")
    p.add_argument("--plot", action="store_true", help="保存 batch size 曲线到 docs/figures/")
    p.add_argument("--trace", action=argparse.BooleanOptionalAction, default=True,
                   help="跑完拉取服务端 /trace 并打印调度轨迹（默认开；--no-trace 关）")
    p.add_argument("--trace-limit", type=int, default=200,
                   help="最多拉取多少条轨迹（默认 200）")
    p.add_argument("--trace-raw", action="store_true",
                   help="不折叠长 prompt 的中间 chunk 行")
    p.add_argument("--trace-all", action="store_true",
                   help="不按本次运行过滤，直接显示服务端缓冲区里的全部历史轨迹")
    p.add_argument("--long-prompt", type=int, default=None, metavar="TOKENS",
                   help="用长 prompt 替代内置短句，长度约 TOKENS 个 token"
                        "（内置短句只有 8~13 token，永远只切 1 个 chunk）")
    p.add_argument("--wait", type=float, default=60.0,
                   help="启动时最多等服务器就绪多少秒（默认 60；模型加载中端口不监听，"
                        "所以刚起服务就跑本脚本会先 Connection refused）")
    p.add_argument("--spans", action=argparse.BooleanOptionalAction, default=True,
                   help="跑完拉取服务端 /spans 并画文本瀑布图（默认开；--no-spans 关）")
    p.add_argument("--color", action=argparse.BooleanOptionalAction, default=None,
                   help="轨迹着色（默认：输出到终端时开，重定向/管道时关）")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    asyncio.run(main_async(args))
