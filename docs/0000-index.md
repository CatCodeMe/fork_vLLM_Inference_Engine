# 文档索引（docs/）

本目录文档统一用 **4 位数字前缀** 编号。数字只表示**管理顺序**（谁先写的），
**不代表阅读顺序** —— 阅读入口见下面的主题分组与路线图。

---

## 1. 主题分组（推荐从这里进）

### 🧭 导航与工具

| 编号 | 文档 | 一句话 |
| --- | --- | --- |
| 0001 | [0001-learning-roadmap.md](0001-learning-roadmap.md) | **起点**：阅读路线、Phase 表、术语表（含 GIL / await / event loop） |
| 0002 | [0002-debugging.md](0002-debugging.md) | 把调试环境跑起来：uv、VSCode launch 配置、断点地图 |
| 0010 | [0010-phase-index.md](0010-phase-index.md) | **Phase 1~11 ↔ 文件 ↔ 代码位置**反查表；被 `Phase N` 注释绕晕时查它 |
| 0004 | [0004-figures.md](0004-figures.md) | 图表约定：matplotlib 数据图 + drawio 结构图（含怎么改图） |

### 📘 主线 A · 读代码（按 Phase 演进）

Phase 1 → 11 的代码主线。建议顺序：**先原理，再调度器，再动手，最后扣细节**。

| 编号 | 文档 | 一句话 |
| --- | --- | --- |
| 0005 | [0005-blog-prefill-decode.md](0005-blog-prefill-decode.md) | 博客草稿：一次 `generate()` 讲透 Transformer / KV cache / 采样 / Roofline |
| 0006 | [0006-scheduler-lifecycle.md](0006-scheduler-lifecycle.md) | 调度器全貌：请求时序图 + `_schedule()` 阶段图 |
| 0007 | [0007-phase2-guide.md](0007-phase2-guide.md) | Phase 2 上手指南：六个研究阶段 + 实验清单 E1~E7 |
| 0014 | [0014-prefill-and-chunking.md](0014-prefill-and-chunking.md) | **分块 prefill 详解**：为什么能切、顺序靠什么保证（三个 offset）、`chunk` vs `block`、三个预算参数的取舍、两套容器与抢占 |

### 🔧 主线 B · 深度机制（引擎之外的原理）

这几篇不按 Phase 走，回答的是"**为什么这么设计**"。它们由读代码时的具体疑问引出，
但结论是通用的基础设施知识。

| 编号 | 文档 | 一句话 |
| --- | --- | --- |
| 0011 | [0011-stopping-and-cancellation.md](0011-stopping-and-cancellation.md) | 生成什么时候停：EOS / `max_new_tokens` / `stop` / 上下文窗口；以及**用户取消为什么掐不掉 GPU** |
| 0012 | [0012-async-and-blocking.md](0012-async-and-blocking.md) | 异步、阻塞与 GIL：`await` 为什么不等于「不等待」、`to_thread` 换来了什么（含实测）、`_blocking` 契约 |
| 0013 | [0013-polling-and-os-scheduling.md](0013-polling-and-os-scheduling.md) | **1ms 之谜**：轮询间隔从哪来（Linux/Windows/macOS）、`sleep`/`yield` 到底在干什么、OS 调度 vs 协程调度、Java `wait` vs `sleep` |
| 0014 | [0014-prefill-and-chunking.md](0014-prefill-and-chunking.md) | **分块 prefill 详解**：为什么能切、顺序靠什么保证（三个 offset）、`chunk` vs `block`、三个预算参数的取舍、两套容器与抢占 |
| 0015 | [0015-kv-cache-lifecycle.md](0015-kv-cache-lifecycle.md) | **KV cache 的一生**：谁计算它（模型里的 `W_k`/`W_v`）、为什么要 pool、为什么要一次性分块、prefill/decode 怎么交接 |

### 🎯 出口 · 走向真实系统

| 编号 | 文档 | 一句话 |
| --- | --- | --- |
| 0016 | [0016-simplified-vs-real-vllm.md](0016-simplified-vs-real-vllm.md) | **简版 ↔ 真实版对照手册**：本仓库概念 → vLLM/llm-d 对应物、容量估算（10M DAU / 500K 并发）、读码顺序建议 |
| 0017 | [0017-capacity-planning.md](0017-capacity-planning.md) | **容量规划**：10M DAU / 500K 峰值并发要多少 HBM 和主机内存（含假设清单与敏感性分析） |
| 0018 | [0018-constrained-decoding.md](0018-constrained-decoding.md) | **约束解码**：正则怎么作用在神经网络上（状态机 + logits 掩码，含可运行的最小实现） |

### 🧠 主线 C · 模型与算法

| 编号 | 文档 | 一句话 |
| --- | --- | --- |
| 0008 | [0008-attention-variants.md](0008-attention-variants.md) | 注意力演进：MHA → MQA → GQA → MLA → 稀疏注意力 |
| 0009 | [0009-attention-landscape.md](0009-attention-landscape.md) | 注意力优化全景：六个正交维度（Flash / Paged / 线性 / 量化 / MLA…） |

### 🔍 代码体检

| 编号 | 文档 | 一句话 |
| --- | --- | --- |
| 0003 | [0003-code-review-findings.md](0003-code-review-findings.md) | 代码梳理：设计局限、可疑点、缺失功能（F1~F30） |

---

## 2. 三条阅读路线

```
路线 A（Phase 主线，边跑边读代码）
  0001 → 0002（把环境跑起来）→ 0005（原理）→ 0006（调度器全貌）
       → 0007（动手做实验）→ 0014（扣 prefill 细节）→ 0010（随时反查 Phase）

路线 B（机制支线，遇到疑问就查）
  0011（停止/取消）· 0012（异步/GIL）· 0013（1ms/OS 调度）
  —— 三篇都从"代码里一个看不懂的地方"出发，但讲的是通用原理

路线 C（模型与算法）
  0008（注意力演进）→ 0009（优化全景）
  —— 与代码主线弱耦合，可独立读；0003 建议对照代码一起看
```

> **被某行注释卡住时**：先在 **0010** 查这个函数属于哪个 Phase、对应哪个文档，
> 再去 0011~0014 里找对应的机制解释。

---

## 3. 全部文档（编号序）

| 编号 | 文档 | 一句话说明 |
| --- | --- | --- |
| 0001 | [0001-learning-roadmap.md](0001-learning-roadmap.md) | 学习路线图：核心链路阅读顺序、Phase 表、术语表 |
| 0002 | [0002-debugging.md](0002-debugging.md) | VSCode + uv 环境、launch 配置、断点地图 |
| 0003 | [0003-code-review-findings.md](0003-code-review-findings.md) | 代码梳理：设计局限、可疑点、缺失功能（F1~F30） |
| 0004 | [0004-figures.md](0004-figures.md) | 图表约定：数据图 `plot_*.py` → `figures/*.svg`；结构图用 drawio |
| 0005 | [0005-blog-prefill-decode.md](0005-blog-prefill-decode.md) | 博客草稿：一次 `generate()` 讲透 Transformer / KV cache / 采样 / 梯度 |
| 0006 | [0006-scheduler-lifecycle.md](0006-scheduler-lifecycle.md) | Phase 2 调度器：请求时序图 + `_schedule` 阶段图 |
| 0007 | [0007-phase2-guide.md](0007-phase2-guide.md) | Phase 2 上手指南：六个研究阶段 + 实验清单（E1~E7） |
| 0008 | [0008-attention-variants.md](0008-attention-variants.md) | 注意力演进：MHA → MQA → GQA → MLA → 稀疏注意力 |
| 0009 | [0009-attention-landscape.md](0009-attention-landscape.md) | 注意力优化全景：六个正交维度（Flash/Paged/线性/量化/MLA…） |
| 0010 | [0010-phase-index.md](0010-phase-index.md) | Phase 1~11 编号 ↔ 文件 ↔ 代码位置对照 |
| 0011 | [0011-stopping-and-cancellation.md](0011-stopping-and-cancellation.md) | 生成什么时候停 + 取消为什么掐不掉 GPU |
| 0012 | [0012-async-and-blocking.md](0012-async-and-blocking.md) | 异步 / 阻塞 / GIL 与本仓库的 async 边界 |
| 0013 | [0013-polling-and-os-scheduling.md](0013-polling-and-os-scheduling.md) | 1ms 轮询间隔、`sleep`/`yield`、OS 调度 vs 协程调度 |
| 0014 | [0014-prefill-and-chunking.md](0014-prefill-and-chunking.md) | 分块 prefill 详解：切片、三个 offset、预算取舍、两套容器与抢占 |
| 0015 | [0015-kv-cache-lifecycle.md](0015-kv-cache-lifecycle.md) | KV cache 的一生：谁计算、为什么有 pool、为什么一次性分块 |
| 0016 | [0016-simplified-vs-real-vllm.md](0016-simplified-vs-real-vllm.md) | 简版 ↔ 真实版对照手册 + 读 vLLM 的顺序建议 |

---

## 4. 预览

- **Markdown + 公式**：VSCode 装 *Markdown Preview Enhanced* 后 `Cmd+Shift+V`。
- **浏览器**：本目录有通用查看器，可渲染任意带公式的 md：

  ```bash
  cd docs && python3 -m http.server 8080
  # 默认渲染博客：
  #   http://localhost:8080/0005-blog-prefill-decode.html
  # 渲染任意文档（用 ?doc= 指定）：
  #   http://localhost:8080/0005-blog-prefill-decode.html?doc=0014-prefill-and-chunking.md
  ```

- **改 drawio 结构图**：双击 `figures/*.drawio.svg` → 进画布 → `Cmd+S`，
  markdown 引用不用动（约定见 [0004](0004-figures.md)）。
- **重生成 matplotlib 图**：`uv run python scripts/plot_all.py`
  （drawio 图手工编辑，不参与重生成）。
