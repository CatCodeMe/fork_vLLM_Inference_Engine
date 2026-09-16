"""
scripts/check_readme_i18n.py — 校验中英文 README 的语言切换条是否一致。

GitHub **没有**原生的多语言 README / 切换 tab 功能，所以只能靠两份文件 + 手写切换条。
手写的东西最容易出现"加了第三种语言忘了改另一边"这类漂移，所以写个校验：

    uv run python scripts/check_readme_i18n.py

检查项：
  1. `README.md` 和 `README.zh-CN.md` 都存在，且都含 `<!-- README-I18N:START ... END -->` 块
  2. 两边的语言条目**完全一致**（同一组语言、同一个顺序）
  3. 每种语言只有一条，且"当前语言"（没有 <a> 包裹的那条）与本文件匹配
  4. 未激活的条目链接指向同一个仓库里的对应文件

退出码 0 = 通过；1 = 有问题（可以直接放进 CI）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 期望的语言集合（要加语言就往这里加一项，并在两个 README 里各补一条）
LANGS = ["English", "简体中文"]

# 语言 → 承载它的文件名
FILES = {"English": "README.md", "简体中文": "README.zh-CN.md"}

BLOCK_RE = re.compile(r"<!-- README-I18N:START.*?-->(.*?)<!-- README-I18N:END -->", re.S)
# 一条语言项：可能是激活态 <img ...>，也可能是可点态 <a href="..."><img ...></a>
ITEM_RE = re.compile(
    r'(?P<link><a\s+href="(?P<href>[^"]+)"\s*>\s*)?'
    r'<img\s+alt="(?P<alt>[^"]+)"\s+src="(?P<src>[^"]+)"\s*/?>\s*'
    r'(?(link)</a>)',
    re.S,
)


def parse(path: Path) -> tuple[str, list[dict]]:
    """返回 (文件里的激活语言, 全部语言条目)。"""
    text = path.read_text(encoding="utf-8")
    m = BLOCK_RE.search(text)
    if not m:
        return "", []
    items = [g.groupdict() for g in ITEM_RE.finditer(m.group(1))]
    active = [i["alt"] for i in items if not i["link"]]
    return (active[0] if len(active) == 1 else ""), items


def main() -> int:
    problems: list[str] = []
    parsed: dict[str, list[dict]] = {}

    for lang, fname in FILES.items():
        path = ROOT / fname
        if not path.exists():
            problems.append(f"{fname} 不存在")
            continue
        active, items = parse(path)
        parsed[fname] = items
        if not items:
            problems.append(f"{fname} 里找不到 <!-- README-I18N:START/END --> 块")
            continue

        alts = [i["alt"] for i in items]
        if alts != LANGS:
            problems.append(f"{fname} 的语言条目是 {alts}，期望 {LANGS}（顺序也要一致）")
        if active != lang:
            problems.append(f"{fname} 里未链接（=当前语言）的是 {active!r}，期望 {lang!r}")
        for i in items:
            if i["link"] and i["href"] != FILES.get(i["alt"]):
                problems.append(
                    f"{fname} 里 {i['alt']!r} 的链接是 {i['href']!r}，"
                    f"期望 {FILES.get(i['alt'])!r}"
                )

    # 两份文件的条目要一致（除激活态外）
    files = list(parsed)
    if len(files) == 2:
        a, b = (parsed[f] for f in files)
        if [i["alt"] for i in a] != [i["alt"] for i in b]:
            problems.append(f"{files[0]} 与 {files[1]} 的语言条目顺序不一致")

    print(f"  README-I18N 校验：检查了 {len(parsed)} 个文件、{len(LANGS)} 种语言")
    for fname, items in parsed.items():
        desc = ", ".join(
            f"{i['alt']}{'（当前）' if not i['link'] else '→' + i['href']}" for i in items
        )
        print(f"    {fname:<20} {desc}")
    if problems:
        print("\n  ❌ 发现问题：")
        for p in problems:
            print(f"    · {p}")
        return 1
    print("\n  ✅ 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
