"""
scripts/serve.py — single, debug-friendly entry point for the PageServe API servers.

Why this file exists
--------------------
Neither server module (`inference_engine/server/app.py` and `app_v2.py`) has an
``if __name__ == "__main__"`` block.  Upstream they are started from a shell
with the uvicorn CLI::

    uvicorn inference_engine.server.app_v2:app --host 0.0.0.0 --port 8001

That is fine for production, but awkward to debug: you cannot press F5 on a
file that never runs.  This wrapper turns the launch into a *normal Python
script* and runs uvicorn **in-process** (``reload=False``) so that breakpoints,
the call stack, and variable inspection all work.

⚠️  Do NOT enable ``reload=True`` while debugging: the reloader spawns a child
process and debugpy will attach to the wrong one.

Usage
-----
    uv run python scripts/serve.py --phase 2          # continuous batching, :8001
    uv run python scripts/serve.py --phase 1          # sequential baseline,  :8000
    MAX_BATCH_SIZE=8 uv run python scripts/serve.py --phase 2

Environment variables (read by ``inference_engine.config.Config``):
    MODEL_NAME, DEVICE, MAX_BATCH_SIZE, PREFILL_BUDGET_TOKENS,
    DECODE_BATCH_LIMIT, KV_BLOCK_SIZE, KV_NUM_BLOCKS, ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

# When run as ``python scripts/serve.py`` the script's own folder (scripts/) is
# on sys.path, not the repo root.  Put the repo root first so that
# ``import inference_engine`` resolves the same way it does under pytest/uvicorn.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_engine.config import Config  # noqa: E402  (after sys.path fix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PageServe debug server entry point")
    parser.add_argument(
        "--phase",
        type=int,
        choices=(1, 2),
        default=2,
        help="1 = sequential baseline (:8000), 2 = continuous batching (:8001)",
    )
    parser.add_argument("--host", default=None, help="bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="bind port")
    parser.add_argument("--log-level", default=None, help="uvicorn log level")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Config() resolves MODEL_NAME / DEVICE / tunables from the environment and
    # auto-detects the accelerator (cuda > mps > cpu).
    config = Config()

    if args.phase == 1:
        app_path = "inference_engine.server.app:app"
        default_port = 8000
    else:
        app_path = "inference_engine.server.app_v2:app"
        default_port = 8001

    host = args.host or os.environ.get("HOST", "127.0.0.1")
    port = args.port or int(os.environ.get("PORT", default_port))
    log_level = args.log_level or config.log_level

    # [GOTCHA] uvicorn 的顺序是 lifespan.startup()（会加载模型，阻塞几秒~几十秒）
    #          然后才 create_server() 绑定端口。所以**加载期间端口是不监听的**，
    #          此时 curl 会得到 Connection refused (errno 61) 而不是超时。
    #          先把这个说清楚，免得被当成"服务起不来"。
    print(
        f"\n[serve] 正在启动 Phase {args.phase}：端口 {args.port or (8001 if args.phase == 2 else 8000)} "
        f"要等模型加载完成后才会开始监听。\n"
        f"        在此之前 curl 会得到 Connection refused —— 这是正常的，"
        f"等服务端打印 'Uvicorn running on ...' 再发请求。\n",
        flush=True,
    )

    # uvicorn imports the app by string in *this* process; make sure the repo
    # root is importable there too.
    os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))

    print(
        f"[serve] phase={args.phase} app={app_path} "
        f"host={host} port={port} device={config.device} "
        f"model={config.model_name}",
        flush=True,
    )

    uvicorn.run(
        app_path,
        host=host,
        port=port,
        log_level=log_level,
        reload=False,   # keep the app in-process so the debugger stays attached
        workers=1,
    )


if __name__ == "__main__":
    main()
