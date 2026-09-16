"""
engine/sequential.py — Core inference logic for sequential serving.

Three clearly separated concerns
---------------------------------
1. prefill()  — full-prompt forward pass; produces KV cache + first token.
2. decode()   — autoregressive loop; consumes KV cache token-by-token.
3. generate() — orchestrates prefill → decode and builds GenerationResult.

Design constraints (intentional — this is the naive baseline)
-------------------------------------------------------------
* No batching.
* No custom KV cache management (uses HuggingFace past_key_values as-is).
* No attention optimisation (Flash Attention, etc.).
* Greedy sampling only.

Memory instrumentation
----------------------
get_memory_stats() is device-aware:
  cuda  → torch.cuda.memory_allocated / reserved
  mps   → torch.mps.current_allocated_memory + psutil RSS for "reserved"
  cpu   → psutil virtual_memory used for both fields
"""

# [LEARN] 这是 Phase 1 的"幼稚基线"，也是理解 decode 的最佳起点：
#         prefill() 一次前向出 KV+首 token；decode() 只喂上一个 token 循环采样；
#         generate() 把两者拼起来并统计时延/内存。连续批处理就是把这里的
#         decode 内部循环"拆成一步"交给调度器。
#
# [LEARN] 一次 generate() 的完整阶段（对照 docs/0005-blog-prefill-decode.md）：
#
#   文本 prompt
#     │  ① tokenizer(prompt)                         [CPU]  文字 → 整数 id
#     ▼
#   input_ids  [1, prompt_len]  (int64)
#     │  .to(device)
#     ▼
#   ┌── prefill() ───────────────────────────────────────────────┐
#   │  model.forward(input_ids)               [GPU/MPS]          │
#   │    embedding 查表 → 24 层 Transformer → lm_head → logits   │
#   │  产出: past_key_values (KV cache) + logits                 │
#   │  采样: argmax(logits[:, -1, :]) → first_token_id           │
#   └────────────────────────────────────────────────────────────┘
#     │
#     ▼
#   ┌── decode() ── 循环 (max_new_tokens - 1) 次 ────────────────┐
#   │  model.forward(input_ids=[1,1], past_key_values=...)       │
#   │  采样 argmax → next_token_id → 追加 → 作为下一次输入        │
#   │  直到达到长度上限 或 命中 eos_token_id                      │
#   └────────────────────────────────────────────────────────────┘
#     │
#     ▼  ② tokenizer.decode(ids)                      [CPU]  整数 → 文字
#   文本 generated_text
#
# [LEARN] 三个要点（详见本章注释与博客）：
#   1) tokenizer 只做「文字 ↔ 整数」；embedding 是 model 内部的第一层。
#   2) prefill 与 decode 调的是同一个 model.forward，区别只是输入长度，
#      以及是否传入 past_key_values（KV cache）。
#   3) 全程用 torch.inference_mode() 关闭梯度（推理只需前向，不需反向）。

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import psutil
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ── GenerationResult ──────────────────────────────────────────────────────────


@dataclass
class GenerationResult:
    """Fully-populated result of one generate() call."""

    prompt: str
    generated_text: str
    prompt_tokens: int
    generated_tokens: int

    # Latency in milliseconds
    ttft_ms: float          # Time-To-First-Token
    total_latency_ms: float # prefill + all decode steps
    tokens_per_second: float

    # Per-token decode latencies (ms); length == generated_tokens - 1
    # (first token comes from prefill, so no separate decode step for it)
    per_token_latencies_ms: List[float]

    # Memory at the END of inference (post-generate snapshot)
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float

    # ISO-8601 UTC timestamp when generate() was called
    timestamp: str


# ── Memory helpers ────────────────────────────────────────────────────────────


def _bytes_to_mb(n: int) -> float:
    return n / (1024 ** 2)


def get_memory_stats(device: str) -> Tuple[float, float]:
    """
    Return (allocated_mb, reserved_mb) for the given device.

    The semantics differ by device:

    CUDA
        allocated = torch.cuda.memory_allocated()  (tensors currently in use)
        reserved  = torch.cuda.memory_reserved()   (total pool held by PyTorch)

    MPS (Apple Silicon)
        allocated = torch.mps.current_allocated_memory()
                    (driver-level allocation for MPS tensors)
        reserved  = process RSS from psutil
                    (MPS has no concept of a reserved pool)
        Falls back to psutil only if the MPS API is unavailable (PyTorch < 2.1).

    CPU
        Both fields are set to psutil virtual_memory().used because there is
        no meaningful distinction between "allocated by tensors" and "reserved"
        in a CPU-only setting.
    """
    # [GOTCHA] MPS 的 "reserved" 用进程 RSS 代替，而 RSS 包含模型权重本身；
    #         所以字段名叫 gpu_memory_reserved_mb，实际语义和 CUDA 不同。
    if device == "cuda":
        allocated = _bytes_to_mb(torch.cuda.memory_allocated())
        reserved = _bytes_to_mb(torch.cuda.memory_reserved())
        return allocated, reserved

    if device == "mps":
        # torch.mps.current_allocated_memory() — available since PyTorch 2.1
        try:
            allocated = _bytes_to_mb(torch.mps.current_allocated_memory())
        except AttributeError:
            # PyTorch < 2.1 — fall back to process RSS
            allocated = _bytes_to_mb(
                psutil.Process().memory_info().rss
            )
        # MPS has no "reserved pool" concept; use process RSS as a proxy.
        reserved = _bytes_to_mb(psutil.Process().memory_info().rss)
        return allocated, reserved

    # CPU fallback
    vm = psutil.virtual_memory()
    used_mb = _bytes_to_mb(vm.used)
    return used_mb, used_mb


# ── Prefill ───────────────────────────────────────────────────────────────────


def prefill(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> Tuple[object, int, float]:
    """
    Tokenize *prompt*, run one full forward pass, return KV cache + first token.

    Returns
    -------
    past_key_values
        HuggingFace KV cache (tuple of layer tensors).
    first_token_id : int
        Greedily sampled token from the last position's logits.
    ttft_ms : float
        Time in milliseconds from start of forward pass to logits available.
        This is the canonical TTFT measurement for the sequential baseline.
    """
    # [LEARN] next(...) 是 Python 内置函数，不是任何框架的东西。
    #         model.parameters() 返回一个“迭代器”（逐个吐出参数张量的对象），
    #         next(迭代器) = 取出第一个元素，即“第一个参数张量”；
    #         再接 .device 就是“模型参数所在的设备”。
    #         这是 PyTorch 社区的惯用简写，等价于“模型现在在哪个设备上”。
    device = next(model.parameters()).device

    # Tokenise — stay on CPU, then move to model device
    # [LEARN] 这一步是 CPU 上的“查字典”，不是 embedding！逐步拆解：
    #   tokenizer(prompt, return_tensors="pt")
    #     → 返回一个 BatchEncoding（类 dict），内部流程：
    #         文本规范化 → 预分词 → BPE 切分 → 每个子词查词表 → token id
    #         → 加特殊 token → 生成 attention_mask。
    #   return_tensors="pt"
    #     → 结果打包成 PyTorch 张量（“pt”=PyTorch；不写则返回普通 Python list）。
    #   enc["input_ids"]
    #     → token id 张量，形状 [batch=1, seq_len]，dtype=int64（长整型）。
    #   enc.get("attention_mask")
    #     → 同形状的 0/1 张量：1=真实 token，0=padding。单条 prompt 全是 1；
    #       批处理/补齐时才需要它告诉模型忽略 pad。用 .get 是因为某些 tokenizer
    #       可能不返回它，所以下面做了 if 判空。
    #   .to(device)
    #     → 把这些整数张量从 CPU 拷到模型所在的 GPU/MPS。
    # [GOTCHA] 这里全程没有向量、没有神经网络；把 id 变成向量（embedding）是在
    #          下面 model(...) 内部的第一层做的。
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # ── TTFT measurement starts here ──────────────────────────────────────────
    t0 = time.perf_counter()

    # [LEARN] torch.inference_mode() 是一个“上下文管理器”（就是 with 语法）：
    #         进入 with 后 PyTorch 关闭梯度记录（不再搭建反向传播用的计算图），
    #         并开启推理优化。生成 token 只需前向、不需反向，所以更快更省内存。
    #         它比 torch.no_grad() 更严格（连版本计数都不记），纯推理首选。
    # [LEARN] 作用域只在 with 块内，退出即恢复；嵌套是允许的（内部有计数）。
    #         所以 prefill 和 decode 这两个独立函数各自包一层；也可以把整个
    #         generate() 包一层，效果一样。
    # [TRACE] model(...) 就是真正干活的地方（下面 decode 里调的是同一个 model）：
    #         它是 Qwen2ForCausalLM 实例，调用它会执行 forward()：
    #           embedding 查表 → 24 层 Transformer（attention + MLP）→ lm_head
    #         参数含义：
    #           input_ids      [1, prompt_len] 的整数 id
    #           attention_mask 告诉模型哪些位置是真 token（单条时全是 1）
    #           use_cache=True 让模型顺便算出并返回 KV cache（past_key_values）
    #           return_dict=True 返回带属性的对象（outputs.logits / .past_key_values）
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,         # ask HF to return past_key_values
            return_dict=True,
        )

    # Logits available → TTFT
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    logits = outputs.logits          # shape: (1, seq_len, vocab_size)
    past_key_values = outputs.past_key_values

    # Greedy sample from last position
    # [LEARN] logits 形状 [1, seq_len, vocab_size]；[:, -1, :] 取“最后一个位置”的
    #         词表分数（因果语言模型：第 t 个位置的输出用来预测第 t+1 个 token）。
    #         argmax 取分数最大的下标 = 最可能的 token id（贪心采样）。
    first_token_id: int = int(logits[:, -1, :].argmax(dim=-1).item())

    logger.debug(
        "prefill: %d prompt tokens, first_token_id=%d, ttft=%.1f ms",
        input_ids.shape[1],
        first_token_id,
        ttft_ms,
    )

    return past_key_values, first_token_id, ttft_ms


# ── Decode ────────────────────────────────────────────────────────────────────


def decode(
    model: PreTrainedModel,
    past_key_values: object,
    first_token_id: int,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> Tuple[List[int], List[float]]:
    """
    Autoregressive decode loop starting from *first_token_id*.

    Each step feeds only the last token and the accumulated KV cache —
    no recomputation of the prompt.

    Returns
    -------
    generated_ids : list[int]
        Token ids produced, INCLUDING first_token_id.
    per_token_latencies_ms : list[float]
        Wall-clock latency (ms) for each decode step AFTER the first token.
        Length is len(generated_ids) - 1.
    """
    # [LEARN] 自回归核心：每次都只输入"上一个 token"（形状 [1,1]），
    #         靠 past_key_values 记住历史，所以 prompt 不会重复计算。
    # [TRACE] 调度器的 _decode_step_single 就是把这个 for 循环体执行一次。
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    # 同一个 next() 惯用法：拿模型所在设备（含义见 prefill() 里的注释）
    device = next(model.parameters()).device

    generated_ids: List[int] = [first_token_id]
    per_token_latencies_ms: List[float] = []

    current_token = torch.tensor([[first_token_id]], dtype=torch.long, device=device)

    if eos_token_id is not None and first_token_id == eos_token_id:
        return generated_ids, per_token_latencies_ms

    # [LEARN] 这就是自回归（autoregressive）decode 的主循环：
    #         每轮只把“上一个 token”喂给模型，模型靠 past_key_values 记得历史，
    #         所以 prompt 不会被重复计算。range(max_new_tokens - 1) 是因为
    #         第一个 token 已经在 prefill 里产生了，所以少循环一次。
    for step in range(max_new_tokens - 1):  # -1 because first token already counted
        t_step = time.perf_counter()

        with torch.inference_mode():
            # [LEARN] 和 prefill 里是同一个 model.forward，区别只在参数：
            #   - input_ids 形状是 [1, 1]（只喂上一个 token，而不是整段 prompt）
            #   - past_key_values 传入上一轮的 KV cache，所以历史不用重算
            # 这就是 prefill 与 decode 的唯一本质区别。
            outputs = model(
                input_ids=current_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        step_ms = (time.perf_counter() - t_step) * 1000.0
        per_token_latencies_ms.append(step_ms)

        logits = outputs.logits            # shape: (1, 1, vocab_size)
        past_key_values = outputs.past_key_values

        # [LEARN] 这一行是“从 logits 选出下一个 token”，逐段拆开看：
        #   logits                     形状 [1, 1, vocab_size]（batch、当前 1 个位置、词表）
        #   logits[:, -1, :]           取最后一个位置的所有词表分数 → [1, vocab_size]
        #   .argmax(dim=-1)            在最后一维（词表）里找最大值的下标 → [1]
        #   .item()                    把这个单元素张量变成 Python 数字
        #   int(...)                   确保是 Python int（后面当索引/传给 tokenizer）
        # [LEARN] logits 是未归一化的原始分数；对它们取 argmax 和对 softmax 后取
        #         argmax 结果一样，所以贪心解码不需要先做 softmax。
        #         若要采样（temperature/top-k/top-p）才需要 softmax。
        next_token_id = int(logits[:, -1, :].argmax(dim=-1).item())
        generated_ids.append(next_token_id)
        # 把刚生成的 token 变成下一次 forward 的输入，形状 [1, 1]
        current_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

        # EOS check
        if eos_token_id is not None and next_token_id == eos_token_id:
            logger.debug("decode: EOS hit at step %d", step + 1)
            break

    logger.debug(
        "decode: generated %d tokens, avg step %.1f ms",
        len(generated_ids),
        sum(per_token_latencies_ms) / max(len(per_token_latencies_ms), 1),
    )

    return generated_ids, per_token_latencies_ms


# ── Generate (orchestrator) ───────────────────────────────────────────────────


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> GenerationResult:
    """
    Full prefill → decode pipeline.  Returns a fully-populated GenerationResult.

    Memory snapshots are taken AFTER inference completes (post-generate state),
    which is the most informative point for a sequential baseline: it shows the
    peak footprint a single request leaves behind.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    # [LEARN] 完整流程 = prefill（只跑 1 次，一次吃掉整段 prompt）+ decode
    #         （跑 N-1 次，每次推进 1 个 token）。两者用的是同一个模型对象。
    # [GOTCHA] 这里直接把 prompt 交给 tokenizer 后生成，没有套 chat template，
    #         也没有 system/user 角色包装。所以用 base 模型（如 Qwen2-0.5B）时
    #         它会“续写文本”而不是“回答问题”；想要问答效果要换 -Instruct 模型。
    timestamp = datetime.now(timezone.utc).isoformat()

    # Count prompt tokens (don't move to device yet; prefill() handles that)
    prompt_token_count = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])

    # ── Wall-clock start ──────────────────────────────────────────────────────
    t_total_start = time.perf_counter()

    past_key_values, first_token_id, ttft_ms = prefill(model, tokenizer, prompt)

    generated_ids, per_token_latencies_ms = decode(
        model=model,
        past_key_values=past_key_values,
        first_token_id=first_token_id,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )

    total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0

    # ── Memory snapshot ───────────────────────────────────────────────────────
    allocated_mb, reserved_mb = get_memory_stats(device)

    # ── Decode text ───────────────────────────────────────────────────────────
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    generated_token_count = len(generated_ids)
    tps = (generated_token_count / total_latency_ms * 1000.0) if total_latency_ms > 0 else 0.0

    return GenerationResult(
        prompt=prompt,
        generated_text=generated_text,
        prompt_tokens=prompt_token_count,
        generated_tokens=generated_token_count,
        ttft_ms=ttft_ms,
        total_latency_ms=total_latency_ms,
        tokens_per_second=tps,
        per_token_latencies_ms=per_token_latencies_ms,
        gpu_memory_allocated_mb=allocated_mb,
        gpu_memory_reserved_mb=reserved_mb,
        timestamp=timestamp,
    )
