# 辅助图表约定（Figures）

工程里的解释性图表分两类，**按「图是算出来的还是画出来的」来选工具**：

| 类型 | 工具 | 产出 | 命名 |
| --- | --- | --- | --- |
| **数据图**（Roofline、分位数曲线、头分组…） | Python 脚本 | `docs/figures/<name>.svg` / `.png` | `scripts/plot_<name>.py` |
| **结构图**（架构、调用链、时序、状态机） | draw.io 手绘 | `docs/figures/<name>.drawio.svg` | 直接用 VS Code 画 |

两者的共同点是：**图是版本管理的源文件，不是导出物**，markdown 里都用普通图片语法引用。

> 为什么结构图不再手写/生成 SVG：SVG 的文本定位、自动换行、连线路由很难调，
> 改一次布局就要重调坐标。draw.io 的布局是交互式的，改动成本低得多。

---

## 一、draw.io 结构图（推荐用于架构/时序图）

### 1.1 关键结论：正式图表用 `.drawio.svg` + 图片引用

两种放置方式的**覆盖面差很多**（已在本仓库实测）：

| 方式 | VS Code 内置预览 | docs 的 HTML 查看器 | GitHub | draw.io 扩展可编辑 |
| --- | --- | --- | --- | --- |
| **A** `.drawio.svg` + 图片引用 | ✅ | ✅ | ✅ | ✅ 双击即画布 |
| **B** ` ```drawio ` 围栏 | ❌ 显示成 XML | ❌ | ❌ | ❌ |

所以**仓库里的正式图一律走 A**。

为什么不走 B：围栏渲染需要查看器**专门支持**这个语言（静态检查显示
`xicilion.markdown-viewer-extension` / docu.md 声明了
`codeBlockLanguages:["drawio"], isDiagram:true`，但**实际预览未渲染成功**，已弃用），
而 VS Code 内置预览、GitHub、本仓库的 HTML 查看器都不认。

`hediet.vscode-drawio` **没有** markdown 集成——它只注册了独立文件的编辑器：

```
*.drawio  *.dio  *.drawio.svg  *.dio.svg  *.drawio.png  *.dio.png
```

没有 markdown-it 插件、没有 CodeLens，所以围栏内嵌在预览里就是一段代码块。
可自行验证（扩展升级后请重跑）：

```bash
# 应该只剩 markdownDescription，说明没有 markdown 特性
grep -o 'markdown[^"]*' \
  ~/.vscode/extensions/hediet.vscode-drawio-*/dist/extension/index.js | sort -u
```

**正式做法**：把图画成 `.drawio.svg`，markdown 里照旧写图片：

```markdown
![调度器三层结构](figures/scheduler-layers.drawio.svg)
```

```markdown
![调度器三层结构](figures/scheduler-layers.drawio.svg)
```

### 1.2 为什么 `.drawio.svg` 就是「编辑原文件 + markdown 实时同步」

`.drawio.svg` 是**双格式单文件**：

- 外壳是**合法 SVG** → VS Code 预览、docs 的 HTML 查看器、GitHub、任何
  markdown 渲染器都能当普通图片显示；
- 内部根 `<svg>` 元素的 `content` 属性（URL-encode 的）里嵌了 **mxfile XML**
  → draw.io 扩展能把它当可编辑的图打开。

保存时扩展会**自己重写 SVG 部分**。所以没有「导出」这一步：
**你编辑的那个文件本身就是 markdown 引用的那个文件。**

`.drawio.png` 同理（位图版本），但优先 `.svg`，缩放下清楚得多。

### 1.3 「实时」到什么程度

| 查看方式 | 改完 svg 后会不会自动刷新 |
| --- | --- |
| VS Code 内置 markdown 预览 | **不保证**。预览的重渲染由 *md 文档本身变更* 触发；只改 svg 时可能还是旧图。保存后在 md 里随便敲一下（或 `Cmd+Z`）即可 |
| `docs/*.html` 查看器（`python3 -m http.server`） | 浏览器 `Ctrl+R` 一次就新（本地无缓存） |
| Markdown Preview Enhanced（`@import`） | 监听被导入的文件，**能自动刷新** |

`.vscode/extensions.json` 已推荐 `shd101wyy.markdown-preview-enhanced`；它
**不认** `drawio` 围栏，但把 `.svg` 当图片导入，所以这样写能出图且自动刷新：

```markdown
@import "figures/scheduler-layers.drawio.svg"
```

### 1.4 画图流程

1. `docs/figures/` 下新建文件 `scheduler-layers.drawio.svg`（**空文件即可**，
   扩展会初始化它）。
2. 在 VS Code 里打开它 → 直接进 draw.io 编辑器。
3. 分屏（View → Editor Layout → Split Right）：左边写 markdown / 开预览，
   右边画图。
4. `Cmd+S` 保存 → 回到 md 预览确认。
5. markdown 里引用：`![说明](figures/<name>.drawio.svg)`。

补充技巧：

- `Draw.io: Convert To...` 命令可以在 `.drawio` ↔ `.drawio.svg` ↔ `.drawio.png`
  之间互转。
- **在意 git diff 就另存一份 `.drawio` 源文件**：`.drawio`（纯 XML）diff 可读，
  `.drawio.svg` 的 diff 噪音较大。二者选一即可，别都提交又不同步。
- 节点文字以 `#Symbol` 开头，双击可跳转到同名代码符号（扩展的 *Code Link* 功能，
  状态栏开关）。

---

## 二、matplotlib 数据图

**脚本 → SVG/PNG → markdown 引用**，可随时重跑、可版本管理、不依赖手绘。

### 目录与命名

```
scripts/
├── _plot_common.py          # 共享工具：统一风格 + 保存 svg/png
├── plot_all.py              # 重生成所有图
├── plot_rope.py             # RoPE 频率与长程衰减
├── plot_gqa_heads.py        # 注意力头分组 MHA / GQA / MQA
└── plot_roofline.py         # prefill vs decode 的 Roofline

docs/figures/
├── rope_freq_decay.svg/.png
├── gqa_heads.svg/.png
└── roofline_prefill_decode.svg/.png
```

规则：

- 脚本命名 **`scripts/plot_<name>.py`** → 输出 **`docs/figures/<name>.svg` 和 `.png`**。
- 图内文字统一**英文 + DejaVu Sans**（字形完整，不依赖系统字体）；中文解释写在
  markdown 的图注里。
- markdown（位于 `docs/`）里用**相对路径**引用：

  ```markdown
  ![说明](figures/<name>.svg)
  ```

  HTML 查看器会随 md 一起解析出 `<img>`，同名目录直接可加载。
- drawio 图同样放 `docs/figures/`，引用方式不变。

**命名区分来源**（看文件名就知道该改脚本还是该开画布）：

| 文件名 | 来源 | 怎么改 |
| --- | --- | --- |
| `rope_freq_decay.svg`（下划线） | 脚本生成 | 改 `plot_*.py` 后重跑 |
| `scheduler-sequence.drawio.svg`（连字符 + `.drawio.svg`） | drawio 手绘 | 双击进画布 |

---

## 三、命令

```bash
# 生成单张
uv run python scripts/plot_rope.py

# 重新生成全部（独立进程，逐个跑 plot_*.py）
uv run python scripts/plot_all.py
```

---

## 四、现有图表

| 图 | 来源 | 用在 |
| --- | --- | --- |
| scheduler.py 三层结构（接线/执行/收尾） | **drawio** `scheduler-layers.drawio.svg` | `0010-phase-index.md` 5 节 |
| Phase 2 请求生命周期时序图 | **drawio** `scheduler-sequence.drawio.svg` | `0006-scheduler-lifecycle.md` 1 节 |
| `_schedule()` 一次迭代 | **drawio** `scheduler-stages.drawio.svg` | `0006-scheduler-lifecycle.md` 2 节 |
| 生成停止判定流程（含 finish_reason） | **drawio** `stop-conditions.drawio.svg` | `0011` §2 |
| 取消为什么延迟 ≤ 1 个 step | **drawio** `cancel-granularity.drawio.svg` | `0011` §5 |
| RoPE 各维频率 + 长程衰减 | `plot_rope.py` | `0005-blog-prefill-decode.md` 4.2 节 |
| 注意力头分组 MHA/GQA/MQA | `plot_gqa_heads.py` | `0005-blog-prefill-decode.md` 4.6 节 |
| Roofline：prefill vs decode | `plot_roofline.py` | `0005-blog-prefill-decode.md` 8.3 节 |
| KV cache 体积对比 MHA/GQA/MQA/MLA | `plot_kv_cache_variants.py` | `0005-...` 4.6 节、`0008-attention-variants.md` 4 节 |
| 注意力演进时间线 | `plot_attention_timeline.py` | `0008-attention-variants.md` 0 节 |
| 注意力优化全景（六维） | `plot_attention_landscape.py` | `0009-attention-landscape.md` 0 节 |
| MLA 矩阵吸收维度变化 | `plot_mla_absorption.py` | `0009-attention-landscape.md` 1.1 节 |
| 调度器请求时序图 | ~~`plot_scheduler_sequence.py`~~ → **drawio** | `0006` 1 节 |
| 调度器 `_schedule` 阶段图 | ~~`plot_scheduler_stages.py`~~ → **drawio** | `0006` 2 节 |

> 上面两行划掉的脚本已改名 `scripts/legacy_plot_scheduler_*.py`（不再被 `plot_all.py` 扫到）。
> `docs/figures/scheduler_sequence.svg|png`、`scheduler_stages.svg|png` 是它们的旧产物，已废弃，
> 待删（新图见 `scheduler-sequence.drawio.svg` / `scheduler-stages.drawio.svg`）。

> 时序图 / 阶段图这类「结构图」将来如果改版，优先考虑改用 drawio（见 §一）：
> 加一条泳道、改一段时序，不用重算坐标。

---

## 五、新增一张 matplotlib 图的模板

```python
"""scripts/plot_<name>.py — 一句话说明。"""
from __future__ import annotations

import matplotlib.pyplot as plt
from _plot_common import save, setup_style  # scripts/ 目录在 sys.path[0]

def main() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([0, 1, 2], [0, 1, 4])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("My figure")
    save(fig, "<name>")     # → docs/figures/<name>.svg / .png

if __name__ == "__main__":
    main()
```

然后在 markdown 里写 `![说明](figures/<name>.svg)`。运行 `plot_all.py` 会自动带上它。
