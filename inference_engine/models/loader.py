"""
models/loader.py — Model and tokenizer loading with device-aware strategy.

Key design decisions
--------------------
* MPS (Apple Silicon): device_map=None, manual .to("mps") after load.
  device_map="auto" is a CUDA multi-GPU feature; on MPS it can silently
  fall back some ops to float32 and produces inconsistent memory readings.
* CUDA: device_map="auto" for standard multi-GPU handling.
* CPU: torch_dtype=torch.float32 — float16 matmuls on CPU are emulated and
  very slow; float32 is the right default.
"""

# [LEARN] 三种设备各有一套加载策略，这是为了避开各自平台的坑。
# [GOTCHA] transformers 5.x 用 `dtype=`，4.x 用 `torch_dtype=`；
#          升级/降级版本时这里最容易报错或静默变成 float32。

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from inference_engine.config import Config

logger = logging.getLogger(__name__)


class LoadedModel(NamedTuple):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str


def load_model_and_tokenizer(config: Config) -> LoadedModel:
    """
    Load a causal LM and its tokenizer according to config.device.

    Returns a LoadedModel named tuple so callers can unpack as:
        model, tokenizer, device = load_model_and_tokenizer(config)
    """
    device = config.device
    model_name = config.model_name

    logger.info("Loading model '%s' targeting device '%s'", model_name, device)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    # Ensure a pad token exists (some models omit it).
    # [GOTCHA] Qwen 等模型没有 pad_token；不补的话 batch padding 会报错。
    #         本引擎目前单序列前向，但下游用 pad 做 batching 时会依赖这一行。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model — device-conditional loading ────────────────────────────────────
    if device == "cuda":
        # Multi-GPU-aware: let HuggingFace distribute layers.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        logger.info("Model loaded with device_map='auto' on CUDA")

    elif device == "mps":
        # Load to CPU first, then move to MPS in one shot.
        # device_map="auto" on MPS can silently dispatch ops to CPU (float32
        # fallback), which corrupts memory measurements.
        # [WHY] 先 CPU 后整体 .to("mps")，不用 device_map——否则部分算子会落到 CPU。
        # [GOTCHA] 注释里写 torch_dtype 是因为这是通用叫法；实际参数名是 dtype。
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=None,          # <-- intentional, not a mistake
            low_cpu_mem_usage=True,   # reduce peak RAM during load
        )
        model = model.to("mps")
        logger.info("Model loaded to MPS via explicit .to('mps')")

    else:  # cpu
        # float16 matmuls on CPU are emulated → very slow. Use float32.
        # [LEARN] CPU 上 fp16 矩阵乘是软件模拟的，反而更慢，所以用 fp32。
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        logger.info("Model loaded on CPU with float32")

    model.eval()  # disable dropout, set BN to eval mode

    logger.info(
        "Model ready. Parameters: %s M | dtype: %s",
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}",
        next(model.parameters()).dtype,  # 第一个参数的 dtype（详见 sequential.prefill 的 next() 注释）
    )

    return LoadedModel(model=model, tokenizer=tokenizer, device=device)
