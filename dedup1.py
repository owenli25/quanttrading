import ast
import pathlib

# ============ 签名去重: search.py ============
p = pathlib.Path('/mnt/d/quant/scripts/factor_mining/search.py')
s = p.read_text(encoding='utf-8')

# 1) 新增退化分析与签名函数 (插在 bucket_of 之后)
anchor = '''DEFAULT_BUCKET = "other"
'''
sig_fn = '''DEFAULT_BUCKET = "other"


# ---------- 退化结构分析 + 结构签名 (2026-08-22) ----------
def prune_dead_subtrees(e: tuple):
    """剪掉数学上退化为常数的子树.

    规则 (截面排序不变性): 常数子树在 cs_rank/cs_zscore/cs_demean 下
    输出全零向量; ts_min/ts_max 对常数输入仍为常数. 递归剪枝:
      - ("feat", x)          -> 保留
      - op 为纯截面算子且任一参数为常数叶子 -> 该参数替换为 None
      - 参数全为常数/None 的子树 -> 整体视为常数
    返回 (剪枝后的树, is_constant).
    """
    if not isinstance(e, tuple):
        return e, True
    if e[0] == "feat":
        return e, False
    if e[0] == "const":
        return None, True
    op = e[0]
    args, consts = [], []
    for a in e[1:]:
        pa, is_c = prune_dead_subtrees(a)
        args.append(pa)
        consts.append(is_c)
    if all(consts):
        return None, True
    # 截面算子的常数参数是死代码 (排序不变); 时序算子的常数参数有效
    if op in ("cs_rank", "cs_zscore", "cs_demean", "add", "sub"):
        kept = [(op, [a]) for a in [args]] if False else None
        new_args = []
        for i, (pa, is_c) in enumerate(zip(args, consts)):
            if is_c and op in ("cs_rank", "cs_zscore", "cs_demean"):
                continue            # 死参数
            if is_c and op in ("add", "sub") and i == 1:
                continue            # 加减常数不改截面排序
            if pa is not None:
                new_args.append(pa)
        if not new_args:
            return None, True
        if len(new_args) == 1:
            # 单参数化简: add(x, c) -> x; 一元算子包一层
            if op in ("add", "sub"):
                return new_args[0], False
            return (op, *new_args), False
        return (op, *new_args), False
    if any(pa is None for pa in args):
        return None, True           # 时序算子吃常数 -> 常数
    return (op, *args), False


def structural_signature(e: tuple) -> str:
    """剥离全部常量为占位符后的规范化字符串 — 同构异参同签名."""
    def canon(t):
        if not isinstance(t, tuple):
            return str(t)
        if t[0] == "const":
            return "<C>"
        if t[0] == "feat":
            return f"F:{t[1]}"
        args = []
        for a in t[1:]:
            if isinstance(a, tuple):
                args.append(canon(a))
            elif isinstance(a, int):
                args.append("<W>")
            else:
                args.append("<C>")
        return f"{t[0]}({','.join(args)})"
    pruned, is_const = prune_dead_subtrees(e)
    if is_const or pruned is None:
        return "DEGENERATE"
    return canon(pruned)


def degenerate_by_variance(S_row: torch.Tensor, tol: float = 1e-9) -> bool:
    """截面方差≈0 的因子无排序信息."""
    return bool(torch.nan_to_num(S_row).std() < tol)
'''
assert anchor in s
s = s.replace(anchor, sig_fn, 1)

p.write_text(s, encoding='utf-8')
ast.parse(s)
print('part1 OK')
