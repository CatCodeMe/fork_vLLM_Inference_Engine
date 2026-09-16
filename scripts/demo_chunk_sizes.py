"""
scripts/demo_chunk_sizes.py — 一条命令看清 prefill_chunk_size 的取舍。

为什么需要这个脚本：**`PREFILL_CHUNK_SIZE` 是服务端配置**，不能通过 HTTP 按请求改，
所以想对比就必须"改配置 → 重启服务 → 跑实验"。这个脚本把这件事自动化了：

    对每个 chunk size：
        1) 用对应的环境变量启动一个临时服务（端口默认 8011，不动你的 :8001）
        2) 等 /health 就绪（含模型加载窗口，见 0002）
        3) 发 1 个长 prompt，隔一小段时间再发 1 个短 prompt
        4) 收两个请求的指标，收工杀进程
    最后打一张对比表 + 条形图。

⚠️ 测这个取舍**必须用「一长一短」**：chunk 的收益是"让长 prefill 别独占"，
   同长度的 prompt 一起发是测不出来的（详见 docs/0014 §9.2）。

用法：
    uv run python scripts/demo_chunk_sizes.py                     # 默认 512 / 128 / 32
    uv run python scripts/demo_chunk_sizes.py --chunk-sizes 512 32
    uv run python scripts/demo_chunk_sizes.py --long-tokens 4000 --delay 0.5
    uv run python scripts/demo_chunk_sizes.py --keep-server       # 跑完不杀（调试用）
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
LONG_BASE = "Explain paged attention and continuous batching in detail. "  # ≈10 token


def spawn_server(port: int, chunk_size: int, budget: int, max_batch: int,
                 log_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env.update({
        "PREFILL_CHUNK_SIZE": str(chunk_size),
        "PREFILL_BUDGET_TOKENS": str(budget),
        "MAX_BATCH_SIZE": str(max_batch),
        "PYTHONUNBUFFERED": "1",
    })
    log = log_path.open("w")
    return subprocess.Popen(
        [sys.executable, "scripts/serve.py", "--phase", "2", "--port", str(port)],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
    )


def wait_ready(client: httpx.Client, url: str, timeout_s: float = 180.0) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            if client.get(f"{url}/health", timeout=2.0).status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    return False


def one_trial(client: httpx.Client, url: str, long_prompt: str, short_prompt: str,
              max_new: int, delay: float) -> dict:
    """发一长一短，返回两个请求的关键指标。

    [GOTCHA] 服务端的 `ttft_ms` **不含排队时间**（F34），所以这里算 `real_ttft`。
    """
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        f_long = pool.submit(
            client.post, f"{url}/generate",
            json={"prompt": long_prompt, "max_new_tokens": max_new}, timeout=600.0)
        time.sleep(delay)
        f_short = pool.submit(
            client.post, f"{url}/generate",
            json={"prompt": short_prompt, "max_new_tokens": max_new}, timeout=600.0)
        r_long, r_short = f_long.result(), f_short.result()

    out = {}
    for tag, resp in (("long", r_long), ("short", r_short)):
        if resp.status_code != 200:
            out[tag] = {"error": f"HTTP {resp.status_code}: {resp.text[:80]}"}
            continue
        d = resp.json()
        q, t = d.get("queue_wait_time_ms") or 0.0, d.get("ttft_ms") or 0.0
        out[tag] = {
            "prompt_tokens": d["prompt_tokens"],
            "queue_ms": q,
            "prefill_ms": t,
            "real_ttft_ms": q + t,
            "total_ms": d["total_latency_ms"],
            "tokens": d["generated_tokens"],
        }
    return out


def bar(v: float, vmax: float, width: int = 32) -> str:
    n = 0 if vmax <= 0 else max(1, round(v / vmax * width))
    return "█" * n


def main() -> None:
    p = argparse.ArgumentParser(description="prefill_chunk_size 取舍对比")
    p.add_argument("--chunk-sizes", type=int, nargs="+", default=[512, 128, 32])
    p.add_argument("--port", type=int, default=8011, help="临时服务端口（默认 8011，避开 :8001）")
    p.add_argument("--budget", type=int, default=512,
                   help="固定 PREFILL_BUDGET_TOKENS（单配置模式用，默认 512）")
    p.add_argument("--budgets", type=int, nargs="+", default=None, metavar="B",
                   help="给出即进入【二维扫描】模式：对 --chunk-sizes × --budgets 的每个组合都跑一遍，"
                        "最后打一张矩阵表。注意每个组合都要重启一次服务，组合多了会很慢")
    p.add_argument("--max-batch", type=int, default=2, help="MAX_BATCH_SIZE（默认 2）")
    p.add_argument("--long-tokens", type=int, default=2500, help="长 prompt 的 token 数")
    p.add_argument("--short-prompt", default="Say hi.")
    p.add_argument("--max-new-tokens", type=int, default=2)
    p.add_argument("--delay", type=float, default=0.4, help="长请求发出后多久发短请求（秒）")
    p.add_argument("--repeats", type=int, default=2, help="每个配置重复几次（取中位数）")
    p.add_argument("--keep-server", action="store_true", help="跑完不杀服务（调试用）")
    args = p.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    long_prompt = LONG_BASE * max(1, round(args.long_tokens / 10))
    outdir = Path("/tmp/pageserve_chunk")
    outdir.mkdir(exist_ok=True)

    print(f"\n{'=' * 78}")
    print(f"  prefill_chunk_size 取舍对比")
    print(f"  长 prompt ≈{args.long_tokens} token + 短 prompt {args.short_prompt!r}"
          f" | budget={args.budget} max_batch={args.max_batch} delay={args.delay}s")
    print(f"{'=' * 78}\n")

    # 组装要跑的配置列表
    pairs: list[tuple[int, int]] = []
    if args.budgets:
        for ch in args.chunk_sizes:
            for b in args.budgets:
                pairs.append((ch, b))
    else:
        for ch in args.chunk_sizes:
            pairs.append((ch, args.budget))

    if len(pairs) > 4:
        print(f"  注意：共 {len(pairs)} 个组合，每个都要重启服务（约 15~30s/个）"
              f" → 预计 {len(pairs) * 20 // 60}~{len(pairs) * 30 // 60} 分钟\n")

    rows: list[dict] = []
    procs: list[subprocess.Popen] = []
    try:
        with httpx.Client() as client:
            for chi, (ch, bud) in enumerate(pairs, 1):
                log = outdir / f"server_chunk{ch}_budget{bud}.log"
                print(f"  ▸ [{chi}/{len(pairs)}] chunk={ch} budget={bud}"
                      f"（一步最多照顾 {bud // ch} 个 prefill 序列）"
                      f"：启动服务（日志 {log}）…", flush=True)
                proc = spawn_server(args.port, ch, bud, args.max_batch, log)
                procs.append(proc)
                if not wait_ready(client, url):
                    print(f"    ✗ 服务 {args.port} 起不来，跳过（看 {log}）")
                    proc.terminate()
                    continue
                client.post(f"{url}/generate",
                            json={"prompt": "warm up", "max_new_tokens": 1}, timeout=300.0)

                trials = []
                for _ in range(max(1, args.repeats)):
                    trials.append(one_trial(client, url, long_prompt,
                                            args.short_prompt, args.max_new_tokens,
                                            args.delay))

                def med(tag: str, key: str) -> float:
                    vals = [t[tag][key] for t in trials if "error" not in t[tag]]
                    return statistics.median(vals) if vals else float("nan")

                rows.append({
                    "chunk": ch, "budget": bud,
                    "slots": bud // ch,
                    "s_queue": med("short", "queue_ms"),
                    "s_prefill": med("short", "prefill_ms"),
                    "s_ttft": med("short", "real_ttft_ms"),
                    "l_total": med("long", "total_ms"),
                    "l_prompt": med("long", "prompt_tokens"),
                    "err": [t["short"].get("error") for t in trials
                            if "error" in t["short"]][:1],
                })
                print(f"    ✓ 短请求真实 TTFT = {rows[-1]['s_ttft']:.0f} ms"
                      f"（排队 {rows[-1]['s_queue']:.0f} + prefill {rows[-1]['s_prefill']:.0f}）"
                      f"；长请求总耗时 {rows[-1]['l_total']:.0f} ms")

                if not args.keep_server:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    time.sleep(1.0)
    finally:
        if not args.keep_server:
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()

    if not rows:
        print("\n没有拿到任何数据。")
        return

    # ── 渲染 ──────────────────────────────────────────────────────────────────
    ok_rows = [r for r in rows if not r["err"]]
    if args.budgets and ok_rows:
        budgets = sorted({r["budget"] for r in ok_rows})
        print(f"\n{'=' * 78}")
        print(f"  二维扫描：单元格 = 短请求真实 TTFT(ms)，越小越好"
              f"（长 {ok_rows[0]['l_prompt']:.0f} token + 短 prompt）")
        print(f"{'=' * 78}")
        print(f"  {'chunk ↓ / budget →':>20}" + "".join(f"{b:>14}" for b in budgets))
        for ch in args.chunk_sizes:
            cells = []
            for b in budgets:
                hit = next((r for r in ok_rows if r["chunk"] == ch and r["budget"] == b), None)
                cells.append(f"{hit['s_ttft']:>13.0f} " if hit else f"{'—':>13} ")
            print(f"  {ch:>20}" + "".join(cells))
        print(f"\n  括号里的关键量是 budget/chunk（一步能照顾几个 prefill 序列）：")
        print(f"  {'chunk ↓ / budget →':>20}" + "".join(f"{b:>14}" for b in budgets))
        for ch in args.chunk_sizes:
            cells = [f"{b // ch:>14}" for b in budgets]
            print(f"  {ch:>20}" + "".join(cells))
        print(f"\n  长请求总耗时（代价面）：")
        print(f"  {'chunk ↓ / budget →':>20}" + "".join(f"{b:>14}" for b in budgets))
        for ch in args.chunk_sizes:
            cells = []
            for b in budgets:
                hit = next((r for r in ok_rows if r["chunk"] == ch and r["budget"] == b), None)
                cells.append(f"{hit['l_total']:>13.0f} " if hit else f"{'—':>13} ")
            print(f"  {ch:>20}" + "".join(cells))
        print(f"\n  读法：同一行往右（budget ↑）= 短请求变快、长请求也变快；"
              f"同一列往下（chunk ↓）= 短请求更快但长请求更慢。")
        print(f"  真正的旋钮是 budget/chunk —— 它才是「一步照顾几个 prefill 序列」。")
        print(f"  ⚠️ 它会【饱和】：一旦 budget/chunk ≥ 并发序列数，再加大就没有收益了"
              f"（本例并发 {args.max_batch}，所以 slots≥{args.max_batch} 之后短请求 TTFT 基本不变）。")
        print()
        return

    # ── 对比表 ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  对比结果（长 prompt 约 "
          f"{rows[0]['l_prompt']:.0f} token；数值为 {args.repeats} 次的中位数）")
    print(f"{'=' * 78}")
    print(f"  {'chunk':>6}  {'短请求:排队':>12}  {'短请求:prefill':>15}  "
          f"{'短请求:真实TTFT':>16}  {'长请求:总耗时':>14}")
    for r in rows:
        if r["err"]:
            print(f"  {r['chunk']:>6}  ⚠️ {r['err'][0]}")
            continue
        print(f"  {r['chunk']:>6}  {r['s_queue']:>10.0f}ms  {r['s_prefill']:>13.0f}ms  "
              f"{r['s_ttft']:>14.0f}ms  {r['l_total']:>12.0f}ms")

    tmax = max(r["s_ttft"] for r in rows if not r["err"]) or 1.0
    print(f"\n  短请求真实 TTFT（越短越好；这才是 chunk 大小真正影响的东西）")
    for r in rows:
        if r["err"]:
            continue
        print(f"    chunk={r['chunk']:<5} {bar(r['s_ttft'], tmax):<33} {r['s_ttft']:>8.0f}ms")

    lmax = max(r["l_total"] for r in rows if not r["err"]) or 1.0
    print(f"\n  长请求总耗时（这是代价：chunk 小 → 长请求每步份额变小 → 变慢）")
    for r in rows:
        if r["err"]:
            continue
        print(f"    chunk={r['chunk']:<5} {bar(r['l_total'], lmax):<33} {r['l_total']:>8.0f}ms")

    print(f"\n  读法：")
    print(f"    · 真正起作用的是 budget / chunk_size —— 也就是「一个 step 能照顾几个")
    print(f"      prefill 序列」。budget={args.budget} 时：chunk=512 → 1 个，chunk=32 → 16 个。")
    print(f"    · chunk 小 = 用「长请求变慢」换「短请求不用排队」。这是公平性取舍，不是提速。")
    print(f"    · ⚠️ 服务端的 ttft_ms 不含排队（F34），所以上面用的是 排队+prefill = 真实 TTFT。")
    print()


if __name__ == "__main__":
    main()
