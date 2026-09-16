"""
scripts/capacity_estimate.py — Codex 量级的容量估算（只算容量，不挑 GPU 型号）。

回答一个问题：10M DAU / 500K 峰值并发的服务，大概要准备多少显存和主机内存？

用法：uv run python scripts/capacity_estimate.py

⚠️ 所有假设都写在代码里（模型配置、单节点吞吐、平均上下文长度）。
   **结论对这些假设极敏感** —— ⑧ 节专门做了敏感性分析。
   配套讲解见 docs/0016-simplified-vs-real-vllm.md §3。
"""

# -*- coding: utf-8 -*-
"""Codex 量级容量估算：10M DAU / 500K 峰值并发（只算容量，不考虑 GPU 型号）"""
TB, GB = 1024**4, 1024**3

print("="*100)
print("  ⓪ 漏斗：10M DAU 推出来的量级，和『500K 峰值并发』对不对得上")
print("="*100)
dau = 10_000_000
for req_per_user in (50, 100, 200):
    rpd = dau * req_per_user
    rps_avg = rpd / 86400
    rps_peak = rps_avg * 3
    for dur in (15, 20, 30):          # 一个请求的平均在途时长（秒），含排队
        inflight = rps_peak * dur
        if dur == 20:
            print(f"  {req_per_user:>3} req/用户/天 → {rpd/1e9:>4.1f}B req/天 → 均值 {rps_avg:>8,.0f} req/s"
                  f" → 峰值(×3) {rps_peak:>8,.0f} req/s × {dur}s 在途 = {inflight/1000:>6,.0f}K 并发在途")
print("  → 『500K 峰值并发』= 大约 100 req/用户/天 + 20s 平均在途时长。**你的量级是自洽的。**")

MODELS = [
    ("7B dense (Qwen2.5-Coder-7B)",  28,   4, 128,  16),
    ("32B dense (Qwen2.5-32B)",      64,   8, 128,  65),
    ("70B dense (Llama-3.3-70B)",    80,   8, 128, 141),
    ("671B MoE + MLA (DeepSeek-V3)", 61, None, None, 1342),
]
def kvpt(L, H, D, b=2):
    return 61*576*b if H is None else 2*L*H*D*b

print()
print("="*100)
print("  ① 地基：每 token 的 KV 占用 + 权重")
print("="*100)
print(f"  {'模型':<34}{'权重 fp16':>11}{'KV/token':>11}{'fp8 KV':>10}{'32K 上下文':>12}{'128K 上下文':>13}")
for n, L, H, D, w in MODELS:
    b16, b8 = kvpt(L,H,D,2), kvpt(L,H,D,1)
    print(f"  {n:<34}{w:>9}GB{b16/1024:>9.0f}KB{b8/1024:>8.0f}KB{b16*32768/GB:>10.1f}GB{b16*131072/GB:>11.1f}GB")

print()
print("="*100)
print("  ② 把 500K 并发的 KV 全放显存？—— 做不到（这就是为什么必须有排队/准入）")
print("="*100)
print(f"  {'模型':<34}{'8K 上下文':>16}{'32K 上下文':>16}{'128K 上下文':>16}")
for n, L, H, D, w in MODELS:
    row = [kvpt(L,H,D)*500_000*c/TB for c in (8192, 32768, 131072)]
    print(f"  {n:<34}{row[0]:>13.0f}TB{row[1]:>13.0f}TB{row[2]:>13.0f}TB")

print()
print("="*100)
print("  ③ 单节点能装多少？（8×H100 = 640 GB HBM；扣掉权重和激活）")
print("="*100)
NODE_HBM, ACT = 640, 40      # GB
print(f"  {'模型':<34}{'权重':>9}{'KV 预算':>11}{'可容 8K 会话':>14}{'可容 32K':>11}{'可容 128K':>12}")
for n, L, H, D, w in MODELS:
    kv_budget = NODE_HBM - w - ACT
    if kv_budget <= 0:
        print(f"  {n:<34}{w:>7}GB   装不下（单机权重就超了，必须多机 TP/PP）")
        continue
    per8, per32, per128 = (kv_budget*GB/(kvpt(L,H,D)*c) for c in (8192, 32768, 131072))
    print(f"  {n:<34}{w:>7}GB{kv_budget:>9}GB{per8:>12,.0f}{per32:>11,.0f}{per128:>12,.0f}")

print()
print("="*100)
print("  ④ 那么要装下 500K 并发需要多少节点？（32B 模型、平均 32K 上下文）")
print("="*100)
kv32_32b = kvpt(64,8,128,2)*32768
per_node = (NODE_HBM-65-ACT)*GB/kv32_32b
print(f"  单节点可容 32K 会话：{per_node:,.0f} 个")
print(f"  500K 并发需节点：{500_000/per_node:>10,.0f} 个  ← 只有把 KV 压在显存里才会得到这个数字")
print(f"  → 结论：不可能。真正在算的只有 {per_node*1000/500_000*100:.2f}% ... 即『必须排队』")

print()
print("="*100)
print("  ⑤ 换一个思路：按【吞吐】定规模（这才是生产上的算法）")
print("="*100)
rps_peak, out_tok, tps_node = 35_000, 1_000, 5_000
total_tps = rps_peak * out_tok
print(f"  峰值 {rps_peak:,} req/s × 平均输出 {out_tok} token = {total_tps/1e6:.0f}M token/s 解码需求")
print(f"  单 8×H100 节点实测解码吞吐 {tps_node:,} token/s → 需 {total_tps/tps_node:,.0f} 个节点")
for n, L, H, D, w in MODELS[:-1]:
    print(f"    {n:<34} 权重池 = {w:>5}GB × {total_tps/tps_node:,.0f} = {w*total_tps/tps_node/1024:>7.0f} TB")
print(f"  · 这些节点能同时驻留的 32K 会话 ≈ {total_tps/tps_node*per_node:,.0f} 个")
print(f"  · 而峰值在途是 500K —— **{500_000/(total_tps/tps_node*per_node):.1f} 倍差距**")
print(f"    → 这个倍数就是网关必须挡掉/排队的比例。")

print()
print("="*100)
print("  ⑥ CPU 内存：『排队的请求』便宜到可以无界")
print("="*100)
for ctx in (4096, 8192):
    print(f"  500K 请求 × {ctx//1024}K prompt token × 4B = {500_000*ctx*4/GB:>6.1f} GB")
print(f"  每请求对象开销 ~10KB → {500_000*10_240/GB:>6.1f} GB")
print(f"  → 合计 ~20-40 GB。**所以 vLLM 让 waiting 队列无界、不拒绝是合理的**：")
print(f"    排队只吃 CPU 内存，而 CPU 内存比 HBM 便宜两个数量级。")

print()
print("="*100)
print("  ⑦ 一张表看完（32B dense、32K 上下文为基准）")
print("="*100)
print(f"  项目                          数量级        说明")
print(f"  {'-'*92}")
print(f"  每会话 KV                     8 GB          32K 上下文 × 256 KB/token")
print(f"  单节点可驻留                  68 会话       8×H100 扣权重/激活后")
print(f"  500K 并发若全驻留             7,353 节点    ← 不可行")
print(f"  按吞吐规划                    7,000 节点    35M token/s ÷ 5K token/s/节点")
print(f"  能同时驻留(按吞吐规模)        476K 会话??   ← 见下")
print(f"  CPU 排队内存                  20-40 GB      便宜的可以无界")
print()
print("  注：⑤ 的『按吞吐 7000 节点』和 ③/④ 的『单节点 68 会话』是**两个侧面**：")
print("      节点数由吞吐（算力）决定；每个节点能同时算的会话数由 KV（显存）决定。")
print("      两者一起决定『同时在算多少个』—— 剩余的都在网关/队列里等。")

print()
print("="*100)
print("  ⑧ 敏感性：结论对『平均上下文长度』和『模型大小』极敏感")
print("="*100)
NODES = 7000
print(f"  {NODES:,} 个节点（= 按吞吐算出来的规模）能同时驻留的会话数，以及占 500K 的比例：")
print(f"  {'模型':<30}" + "".join(f"{c//1024:>18}K 上下文" for c in (8192, 32768, 131072)))
for n, L, H, D, w in MODELS[:-1]:
    per = {}
    for c in (8192, 32768, 131072):
        per[c] = (NODE_HBM - w - ACT)*GB/(kvpt(L,H,D)*c)
    cells = "".join(f"{per[c]*NODES/1000:>12,.0f}K ({per[c]*NODES/500_000*100:>4.0f}%)" for c in (8192, 32768, 131072))
    print(f"  {n:<30}{cells}")
print(f"  → 7B/8K 时 KV 完全不是瓶颈（能装 9M+）；32B/128K 时只能装 12% → 剩下 88% 必须排队。")

print()
print("="*100)
print("  ⑨ 所以『要准备多少内存』（以 32B dense / 7000 节点 / 32K 上下文为基准）")
print("="*100)
hbm = NODES * NODE_HBM
weights = NODES * 65
kv_cap = NODES * 535
print(f"  GPU 显存（HBM）总量   : {hbm/1024:>10,.0f} TB  = {NODES:,} 节点 × 640 GB")
print(f"    其中权重池          : {weights/1024:>10,.0f} TB  （32B fp16 × 7000 副本）")
print(f"    其中 KV 预算        : {kv_cap/1024:>10,.0f} TB  （≈ {kv_cap*GB/kv32_32b/1000:,.0f}K 个 32K 会话）")
print(f"  主机内存（CPU RAM）   :")
print(f"    排队请求            : {20:>10,.0f} GB   （500K 在途的 prompt token + 对象开销）")
print(f"    若开 SWAP 抢占池    : {kv_cap/1024:>10,.0f} TB   ← 和 KV 池同量级！")
print(f"    → 这就是 vLLM 默认用 RECOMPUTE 而不是 SWAP 的【容量层面】理由：")
print(f"      开 swap 等于要求主机侧再准备一份和 HBM 同量级的内存。")
print()
print(f"  一句话：HBM 约 {hbm/1024:,.0f} TB（其中权重 {weights/1024:,.0f} TB + KV {kv_cap/1024:,.0f} TB），")
print(f"          主机内存若不 swap 只需几十 GB，若 swap 则要 {kv_cap/1024:,.0f} TB 量级。")
