"""
scripts/demo_constrained_decoding.py — 约束解码的最小可运行实现。

用最少的代码回答一个问题：**正则约束怎么可能作用在一个神经网络上？**

答案分三步，本脚本把三步都跑出来给你看：
  ① 正则 → 状态机（经典自动机，**没有任何神经网络的数学**）
  ② 状态机 → 每个状态允许哪些 token（预计算 + 缓存，这是"引擎层"的活）
  ③ 生成时 logits[不合法] = -inf，然后照常 argmax（**模型的数学一点没变**）

用法：
    HF_HUB_OFFLINE=1 uv run python scripts/demo_constrained_decoding.py

⚠️ 本脚本为了讲清原理做了两处简化（真实库要处理，见 docs/0018）：
    · 状态机建在**字符**上，真实实现建在**字节**上（要处理多字节 UTF-8）
    · 用 `torch.full_like(-inf)` 逐 token 赋值，真实实现用**位掩码** + 缓存
"""
from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2-0.5B"
GRAMMAR_RE = r"\d{4}-\d{2}-\d{2}"          # 目标：输出一个日期
PROMPT = "The Eiffel Tower was completed in the year"


# ── ① 正则 → 状态机 ──────────────────────────────────────────────────────────
def build_fsm() -> tuple[dict[int, dict[str, int]], set[int], int]:
    """把 \\d{4}-\\d{2}-\\d{2} 编译成状态机。

    状态 = "已经匹配到哪了"；转移表 = "下一个字符允许是什么"。
    这就是编译原理里的自动机，和 Transformer 毫无关系。
    """
    trans: dict[int, dict[str, int]] = {}
    for s in range(3):                              # 前 3 个数字 → 下一个状态
        trans[s] = {str(d): s + 1 for d in range(10)}
    trans[3] = {str(d): 4 for d in range(10)}       # 第 4 个数字 → 该 '-'
    trans[4] = {"-": 5}
    trans[5] = {str(d): 6 for d in range(10)}
    trans[6] = {str(d): 7 for d in range(10)}       # 该 '-'
    trans[7] = {"-": 8}
    trans[8] = {str(d): 9 for d in range(10)}
    trans[9] = {str(d): 10 for d in range(10)}      # 10 = 接受态
    trans[10] = {}                                  # 接受态：只能停
    return trans, {10}, 0


TRANS, ACCEPT, START = build_fsm()


def fsm_step(state: int, ch: str) -> int | None:
    """从 state 读入一个字符；None = 这个字符不合法。"""
    return TRANS.get(state, {}).get(ch)


def can_consume(state: int, text: str) -> int | None:
    """这个 token 的每个字符都能合法走完吗？（返回落点状态）"""
    for ch in text:
        state = fsm_step(state, ch)
        if state is None or state not in TRANS:
            return None
    return state


def build_token_masks(tok) -> tuple[dict[int, set[int]], int]:
    """② 把「允许的字符」翻译成「允许的 token」。

    这是引擎层真正要做的事：对每个 FSM 状态，算出一个词表级集合。
    真实实现会把它压成 bitmask 并缓存（每个状态一份）。
    """
    ids = sorted(tok.get_vocab().values())
    text_of: dict[int, str] = {}
    for i in ids:
        try:
            s = tok.decode([i])
        except Exception:  # noqa: BLE001
            continue
        if s and "\ufffd" not in s:                # 跳过解不干净的（半个多字节字符）
            text_of[i] = s
    masks = {
        st: {i for i, t in text_of.items() if can_consume(st, t) is not None}
        for st in TRANS
    }
    return masks, len(ids)


def generate(model, tok, masks, constrained: bool, max_steps: int = 14):
    """③ 生成：mask 掉不合法 token 的 logits，然后照常 argmax。"""
    ids = tok(PROMPT, return_tensors="pt")["input_ids"]
    state = START if constrained else None
    out: list[int] = []
    trace: list[tuple[int, int, str]] = []
    with torch.no_grad():
        for _ in range(max_steps):
            if constrained and state in ACCEPT:
                break
            logits = model(input_ids=ids).logits[0, -1, :].float()
            if constrained:
                masked = torch.full_like(logits, float("-inf"))
                allowed = masks[state]
                idx = torch.tensor(sorted(allowed))
                masked[idx] = logits[idx]          # ★ 唯一"约束"发生的地方
                logits = masked
            nxt = int(logits.argmax())
            ch = tok.decode([nxt])
            out.append(nxt)
            if constrained:
                trace.append((state, len(masks[state]), ch))
                state = fsm_step(state, ch)
                if state is None:
                    break
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
            if not constrained and nxt == tok.eos_token_id:
                break
    return tok.decode(out, skip_special_tokens=True), trace


# ── 级别 1：手写 if/else —— 约束的【全部本质】，没有任何数学 ─────────────────
def level1_handwritten() -> None:
    """目标：只允许模型输出 "YES" 或 "NO"。

    【关键认识】这份 if/else 就是约束的"知识"本身。
    它和神经网络无关 —— 它就是普通业务代码。
    mask 只是把这份规则"应用到 logits 上"的那一步。
    """
    table = {"": {"Y", "N"}, "Y": {"E"}, "YE": {"S"}, "YES": set(),
             "N": {"O"}, "NO": set()}

    def allowed(so_far: str) -> set[str] | None:
        return table.get(so_far)          # None = 非法前缀（约束已经在拒绝它了）

    print("=" * 92)
    print("级别 1：手写 if/else —— 这就是约束的【全部本质】，没有任何数学")
    print("=" * 92)
    print("""
  目标：只允许输出 "YES" / "NO"。规则长这样（普通代码，和神经网络无关）：

      def allowed_chars(so_far):
          if   so_far == "":    return {"Y", "N"}
          elif so_far == "Y":   return {"E"}
          elif so_far == "YE":  return {"S"}
          elif so_far == "YES": return set()      # 结束
          elif so_far == "N":   return {"O"}
          elif so_far == "NO":  return set()      # 结束
          else:                 return None       # 非法

  生成时：allowed = allowed_chars(已生成文本)
          logits[所有不在 allowed 里的 token] = -inf     ← mask 就是这么来的""")

    print("\n  走一遍 'YES'：")
    so_far = ""
    while True:
        a = allowed(so_far)
        print(f"    已生成 {so_far!r:7} → 允许下一个字符 {sorted(a) if a else '（结束）'}")
        if not a:
            break
        so_far += sorted(a)[0]

    print("\n  走一遍 'YEP'（故意非法）：")
    so_far = ""
    for ch in "YEP":
        a = allowed(so_far) or set()
        ok = ch in a
        print(f"    已生成 {so_far!r:7} → 允许 {sorted(a) if a else '（结束）'}"
              f"；输入 {ch!r} → {'✅ 合法' if ok else '❌ 被拒绝（这个字符的 logit 会是 -inf）'}")
        so_far += ch


# ── 级别 2：正则 → 自动生成级别 1 的表 ────────────────────────────────────────
def _month_loose() -> dict:
    """宽松：月份就两位数字。"""
    t = {s: {str(d): s + 1 for d in range(10)} for s in range(3)}
    t[3] = {str(d): 4 for d in range(10)}
    t[4] = {"-": 5}
    t[5] = {str(d): 6 for d in range(10)}
    t[6] = {str(d): 7 for d in range(10)}
    t[7] = {"-": 8}
    t[8] = {str(d): 9 for d in range(10)}
    t[9] = {str(d): 10 for d in range(10)}
    t[10] = {}
    return t


def _month_strict() -> dict:
    """严格：月份必须是 01~12。

    [LEARN] 关键：状态必须"记得足够多"。
    "0 之后"允许的第二位是 1~9，"1 之后"允许的是 0~2 —— 两者不同，
    所以它们【必须是两个状态】（6a / 6b），不能合并成一个。
    """
    t = {s: {str(d): s + 1 for d in range(10)} for s in range(3)}
    t[3] = {str(d): 4 for d in range(10)}
    t[4] = {"-": 5}
    t[5] = {"0": "6a", "1": "6b"}
    t["6a"] = {str(d): 7 for d in range(1, 10)}     # 0 之后 → 1~9
    t["6b"] = {str(d): 7 for d in range(0, 3)}      # 1 之后 → 0~2
    t[7] = {"-": 8}
    t[8] = {str(d): 9 for d in range(10)}
    t[9] = {str(d): 10 for d in range(10)}
    t[10] = {}
    return t


def level2_regex(text: str = "1889-18-19") -> None:
    """正则不是"另一种数学"，它只是【写规则的语法】，会被编译成级别 1 那种表。"""
    print()
    print("=" * 92)
    print("级别 2：手写级别 1 太累 → 用正则自动生成（1968 年的 Thompson 构造）")
    print("=" * 92)
    print("""
  正则 → 状态机 是【编译器算法】（grep / sed / awk 用的就是它），
  和神经网络的数学毫无关系。它最终产出的是级别 1 那种"状态 → 允许字符"的表。

      \\d{4}-\\d{2}-\\d{2}                宽松：月份两位数字
      \\d{4}-(0[1-9]|1[0-2])-\\d{2}         严格：月份 01~12
""")
    loose, strict = _month_loose(), _month_strict()
    print(f"  宽松版编译出 {len(loose)} 个状态；严格版 {len(strict)} 个"
          f"（多 {len(strict) - len(loose)} 个 —— 因为要区分『0 之后』和『1 之后』）")
    print(f"\n  用两个 grammar 分别验证同一个输出 {text!r}：")
    for name, trans in (("宽松 \\d{2}", loose), ("严格 (0[1-9]|1[0-2])", strict)):
        st, verdict = 0, "接受"
        for i, ch in enumerate(text):
            nxt = trans.get(st, {}).get(ch)
            if nxt is None:
                verdict = f"❌ 拒绝（第 {i} 个字符 {ch!r}，状态 {st}）"
                break
            st = nxt
        else:
            verdict = "✅ 接受" if st == 10 else f"❌ 拒绝（停在状态 {st}）"
        print(f"    {name:<24} → {verdict}")
    print("""
  ⚠️ 这就是那个 gap 的答案：
     约束解码【不检查『日期是否正确』】—— 它只检查『是否符合你给的 grammar』。
     『18 月』是否合法，完全取决于【你写的 grammar 能不能表达 1~12】：
       宽松 grammar → 通过；严格 grammar → 在第 6 个字符被拒。
     所以：约束的能力上限 = 【形式语言能表达的东西】
       · 正则：能表达『月份 01~12』『邮箱格式』
       · JSON Schema：能表达『字段名/类型/枚举』
       · 都不能表达：『这个数字是真实的温度』『这段代码能跑』""")


def main() -> None:
    level1_handwritten()
    level2_regex()

    tok = AutoTokenizer.from_pretrained(MODEL)
    print()
    print("=" * 92)
    print("级别 3：把『允许的字符』接到模型的『允许的 token』上")
    print("=" * 92)
    for s in sorted(TRANS):
        t = TRANS[s]
        if len(t) == 10:
            desc = "只允许 0~9"
        elif t:
            desc = f"只允许 {list(t)}"
        else:
            desc = "接受态（必须停）"
        print(f"    状态 {s:>2}: {desc}")

    masks, vocab_size = build_token_masks(tok)
    print()
    print("=" * 88)
    print("② 「允许的字符」→「允许的 token」：每个状态只有极小一部分词表合法")
    print("=" * 88)
    print(f"  词表 {vocab_size:,} 个 token")
    print(f"  {'状态':>4}  {'合法 token 数':>14}  {'占词表':>9}   例子")
    for s in sorted(TRANS):
        n = len(masks[s])
        ex = [tok.decode([i]) for i in sorted(masks[s])[:6]]
        print(f"  {s:>4}  {n:>12,}  {n / vocab_size * 100:>8.3f}%   {ex}")

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).eval()
    date_re = re.compile(GRAMMAR_RE)
    print()
    print("=" * 88)
    print("③ 同一个模型、同一个 prompt，只差一个 mask")
    print("=" * 88)
    unc, _ = generate(model, tok, masks, constrained=False)
    con, trace = generate(model, tok, masks, constrained=True)
    print(f"  未约束（普通 greedy）      : {unc!r}")
    print(f"      匹配 {GRAMMAR_RE}?  {bool(date_re.fullmatch(unc.strip()))}")
    print(f"\n  已约束（mask 非法 token）  : {con!r}")
    print(f"      匹配 {GRAMMAR_RE}?  {bool(date_re.fullmatch(con))}")
    print(f"\n  逐步过程（状态 / 该步合法 token 数 / 选出）：")
    for st, n_ok, ch in trace:
        print(f"    状态 {st:>2}   合法 {n_ok:>7,}/{vocab_size:,}   选了 {ch!r}")
    print()
    print("  ⚠️ 注意『已约束』的输出 1889-18-19 —— 格式完全合法，但 18 月 19 日并不存在。")
    print("     约束只保证【语法】，不保证【语义】。这是它最常见的误用。")


if __name__ == "__main__":
    main()
