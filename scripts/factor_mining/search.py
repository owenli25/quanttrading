"""
search.py — 因子搜索: 随机生成 + 遗传迭代 (GPU 批量评估) v2

v2 升级 (2026-08-21):
  1. 温和变异: 子树替换概率 40%→12%, 新增微变异算子
     (窗口重抽 / 常量重抽 / 叶子特征替换)
  2. 锦标赛选择 (k=5) 替代纯截断选择, 保留选择压力梯度
  3. 复合适应度: fitness = ICIR − λ·turnover − μ·max_corr
     λ/μ 预注册默认 (0.5, 0.3)
  4. 机制分桶精英池 (MAP-Elites 简化版): 按表达式主特征族分桶,
     每桶保 top-k, 防止精英被同结构近亲塞满
  5. 两阶段搜索: GP 粗搜 → top50 窗口/常量网格精修 (批量 GPU 评估)

流程: gen0 随机 N 个表达式 -> GPU 批量 IC 评估 -> 复合适应度保留精英
      -> 变异/交叉/新鲜随机生成下一代 -> 迭代 G 代 -> 精英合并终评+精修。
"""
from __future__ import annotations

import gc
import itertools
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

import factor_lib as fl
from factor_mining.evaluate import ic_matrix, ic_summary, quantile_spread, turnover
from factor_mining.expressions import (
    CONSTS,
    DELTAS,
    Evaluator,
    WINDOWS,
    expr_str,
    random_expr,
)


def _sanitize(f: torch.Tensor) -> torch.Tensor:
    """NaN/Inf 清零 (warmup 期与除零保护)."""
    return torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


# ---------- 机制分桶: 表达式主特征族 ----------
FEATURE_BUCKETS = {
    "vol": ("vol_", "bb_width"),
    "volume": ("vol_z", "turnover", "amihud", "dollar_vol", "adv"),
    "momentum": ("mom", "ret_", "pos_in_range", "vwap_dev", "gap"),
    "meanrev": ("rsi",),
}
DEFAULT_BUCKET = "other"


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
    if op in ("add", "sub"):
        # 加减常数不改截面排序 → 死参数记为 DEAD 占位符 (不静默丢弃)
        live = [(i, pa) for i, (pa, is_c) in enumerate(zip(args, consts))
                if not is_c]
        dead = [i for i, (_, is_c) in enumerate(zip(args, consts)) if is_c]
        if not live:
            return None, True
        base_i, base_pa = live[0]
        e_out = base_pa
        for j in dead:
            e_out = ("add", e_out, "<DEAD>") if op == "add" else ("sub", e_out, "<DEAD>")
        for i, pa in live[1:]:
            e_out = (op, e_out, pa)
        return e_out, False
    if op in ("cs_rank", "cs_zscore", "cs_demean"):
        # 截面算子的常数参数排序不变 → 死参数, 但保留占位符以区分结构位置
        if args[0] is None and consts[0]:
            return ("<RANK-DEAD>",), True   # cs_rank(纯常数) 无信息
        return (op, args[0]), False
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
        if t[0] == "<RANK-DEAD>":
            return "<RD>"
        # 截面排序外壳透明化: cs_rank/cs_zscore/cs_demean 不改变横截面排序,
        # 包裹与否同效 → 签名相同
        if t[0] in ("cs_rank", "cs_zscore", "cs_demean") and len(t) == 2 \
                and isinstance(t[1], tuple):
            return canon(t[1])
        args = []
        for a in t[1:]:
            if isinstance(a, tuple):
                args.append(canon(a))
            elif isinstance(a, int):
                args.append("<W>")
            else:
                args.append("<C>")
        return f"{t[0]}({','.join(args)})"
    # 注意: 不做整树 DEGENERATE 判定 — 含 -inf/NaN 传播路径的表达式
    # 行为依赖算子实现细节 (实测 log1p(-1.0) 链输出非均匀面板),
    # 静态剪枝不可靠. 死参数统一替换为 <DEAD> 占位符参与签名:
    # 死参数选择不同的变体会得到相同签名从而被合并.
    pruned, _ = prune_dead_subtrees(e)
    if pruned is None:
        return "DEAD"          # 仅当整树字面常数时
    sig = canon(pruned)
    if "<DEAD>" in sig or "None" in sig:
        pass                   # 已由 prune 替换, canon 会输出占位符
    return sig


def degenerate_by_variance(S_row: torch.Tensor, tol: float = 1e-9) -> bool:
    """截面方差≈0 的因子无排序信息."""
    return bool(torch.nan_to_num(S_row).std() < tol)


def bucket_of(e: tuple) -> str:
    """按表达式引用最多的特征族分桶; 无特征命中则 other."""
    s = expr_str(e)
    counts = {}
    for key, needles in FEATURE_BUCKETS.items():
        counts[key] = sum(s.count(nd) for nd in needles)
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else DEFAULT_BUCKET


# ---------- 张量辅助 ----------
def _batch_turnover_gpu(S: torch.Tensor) -> torch.Tensor:
    """[N,A,D] -> [N] 日均排名位移 (GPU 向量化换手代理)."""
    fr = S.argsort(dim=1).argsort(dim=1).float()
    fr = fr - fr.mean(dim=1, keepdim=True)
    return (fr[:, :, 1:] - fr[:, :, :-1]).abs().mean(dim=(1, 2))


def _batch_corr_abs(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """[N,F] 行两两 Pearson |相关| 矩阵 [N,N]."""
    An = A - A.mean(dim=1, keepdim=True)
    Bn = B - B.mean(dim=1, keepdim=True)
    num = An @ Bn.T
    den = An.norm(dim=1, keepdim=True) @ Bn.norm(dim=1, keepdim=True).T + 1e-12
    return (num / den).clamp(-1, 1)


def _collect_positions(t: tuple, kind: str, path=()) -> list[tuple]:
    """收集窗口 int 参数或 const 叶子的位置路径."""
    out = []
    if not isinstance(t, tuple):
        return out
    if t[0] == "const" and kind == "const":
        out.append(path)
        return out
    for i, a in enumerate(t[1:]):
        if isinstance(a, tuple):
            out += _collect_positions(a, kind, path + (i,))
        elif kind == "window" and isinstance(a, int):
            out.append(path + (i,))
    return out


def _set_at(e: tuple, path: tuple, val) -> tuple:
    """在 path 处写入 val (int 参数或叶子替换)."""
    if not path:
        if e[0] in ("feat", "const"):
            return ("const", val)
        args = list(e[1:])
        # 无 rest 且目标是 int 参数: 由调用方保证
        return (e[0], *args) if len(args) != 1 else (e[0], val)
    args = list(e[1:])
    i, rest = path[0], path[1:]
    if isinstance(args[i], tuple):
        args[i] = _set_at(args[i], rest, val)
    else:
        args[i] = val
    return (e[0], *args)


def _enumerate_remaps(t: tuple, cap: int = 64) -> list[tuple]:
    """枚举一棵树的窗口×常量重映射变体; 超过 cap 随机采样."""
    wpos = _collect_positions(t, "window")
    cpos = _collect_positions(t, "const")
    n_w = len(wpos)
    n_c = len(cpos)
    total = (len(WINDOWS) ** n_w if wpos else 1) * (len(CONSTS) ** n_c if cpos else 1)
    # 指数防护: 组合超 cap 用随机采样索引, 不物化笛卡尔积
    if total <= cap:
        idx_pairs = list(itertools.product(
            itertools.product(range(len(WINDOWS)), repeat=n_w),
            itertools.product(range(len(CONSTS)), repeat=n_c)))
        random.Random(7).shuffle(idx_pairs)
        idx_pairs = idx_pairs[:cap]
    else:
        rr = random.Random(7)
        seen_pair = set()
        idx_pairs = []
        while len(idx_pairs) < min(cap, total):
            wi = tuple(rr.randrange(len(WINDOWS)) for _ in range(n_w))
            ci = tuple(rr.randrange(len(CONSTS)) for _ in range(n_c))
            key = (wi, ci)
            if key not in seen_pair:
                seen_pair.add(key)
                idx_pairs.append(key)
    base_str = expr_str(t)
    outs = []
    for wi, ci in idx_pairs:
        wvals = [WINDOWS[k] for k in wi] if wpos else None
        cvals = [CONSTS[k] for k in ci] if cpos else None
        e = t
        if wvals:
            for p, w in zip(wpos, wvals):
                e = _set_at(e, p, w)
        if cvals:
            for p, cv in zip(cpos, cvals):
                e = _set_at(e, p, cv)
        if expr_str(e) != base_str:
            outs.append(e)
    return outs


class FactorMiner:
    # 复合适应度惩罚系数 (预注册, 不随结果调整)
    LAMBDA_TURNOVER = 0.25
    MU_CORR = 0.3

    def __init__(self, features: dict[str, torch.Tensor],
                 fwd_rank: torch.Tensor, dates: pd.DatetimeIndex,
                 feats_order: list[str], device=fl.DEVICE,
                 col_slice: tuple[int, int] | None = None):
        """col_slice=(lo,hi): 因子张量评估后切到 [lo,hi) 列, 与 fwd_rank 窗口对齐 (OOS验证用)."""
        self.features = features
        self.fwd_rank = fwd_rank          # [A,D] 前瞻收益截面排名
        self.dates = dates
        self.feats_order = feats_order
        self.device = device
        self.col_slice = col_slice
        self.evaluator = Evaluator(features)

    # ---------- 批量评估 ----------
    def evaluate(self, exprs: list[tuple]) -> tuple[pd.DataFrame, torch.Tensor]:
        """评估表达式列表 -> (IC汇总表, 因子堆叠 [N,A,D])."""
        self.evaluator.clear()
        names = [expr_str(e) for e in exprs]
        stack = [_sanitize(self.evaluator.eval(e)) for e in exprs]
        S = torch.stack(stack, dim=0)
        if self.col_slice is not None:
            S = S[:, :, self.col_slice[0]:self.col_slice[1]]
        ic = ic_matrix(S, self.fwd_rank)
        df = ic_summary(ic, names, self.dates)
        self.evaluator.clear()          # 批后释放子树 memo, 防变体批积累
        return df, S

    def evaluate_with_fitness(self, exprs: list[tuple]) -> tuple[pd.DataFrame, torch.Tensor]:
        """评估 + 复合适应度: fitness = ICIR − λ·turnover − μ·mean_offdiag|corr|.

        换手与相关性均在批量张量层面计算. 相关性惩罚用行内 off-diagonal 均值
        近似"与种群其余成员的重复度"——比逐个对精英算 max 相关便宜一个数量级,
        且随种群去重自动收紧.
        """
        # 评估 + 取回居中排名 zf: 换手直接复用, 免第二次双重 argsort (省 ~3x 面板瞬态)
        self.evaluator.clear()
        names = [expr_str(e) for e in exprs]
        stack = [_sanitize(self.evaluator.eval(e)) for e in exprs]
        S = torch.stack(stack, dim=0)
        del stack
        if self.col_slice is not None:
            S = S[:, :, self.col_slice[0]:self.col_slice[1]]
        n = S.shape[0]

        ic, zf = ic_matrix(S, self.fwd_rank, want_ranks=True)
        df = ic_summary(ic, names, self.dates)
        tr = _batch_turnover_gpu(zf)      # 居中不影响相邻位移差
        df["turnover"] = tr.cpu().numpy()
        del zf, ic
        gc.collect()
        torch.cuda.empty_cache()

        if n > 1:
            flat = S.reshape(n, -1)
            C = _batch_corr_abs(flat, flat)
            eye = torch.eye(n, device=S.device, dtype=torch.bool)
            Cm = C.masked_fill(eye, float("nan"))
            corr_pen = torch.nan_to_num(Cm.nanmean(dim=1), nan=0.0)
        else:
            corr_pen = torch.zeros(n, device=S.device)
        df["corr_penalty"] = corr_pen.cpu().numpy()

        df["fitness"] = (df["ICIR"]
                         - self.LAMBDA_TURNOVER * df["turnover"]
                         - self.MU_CORR * df["corr_penalty"])
        return df, S

    # ---------- 微变异算子 ----------
    @staticmethod
    def _map_leaves(e: tuple, fn) -> tuple:
        if not isinstance(e, tuple):
            return e
        if e[0] in ("feat", "const"):
            return fn(e)
        return (e[0], *(FactorMiner._map_leaves(a, fn) if isinstance(a, tuple) else a
                        for a in e[1:]))

    def _mutate_windows(self, e: tuple, rng: random.Random) -> tuple:
        """所有窗口参数均匀重抽."""
        out = []
        for a in e[1:]:
            if isinstance(a, tuple):
                out.append(self._mutate_windows(a, rng))
            elif isinstance(a, int):
                pool = DELTAS if e[0] == "ts_delta" else WINDOWS
                out.append(rng.choice(pool))
            else:
                out.append(a)
        return (e[0], *out)

    def mutate(self, e: tuple, rng: random.Random,
               subtree_reset_p: float = 0.12) -> tuple:
        """温和变异: 20% 窗口微调 / 15% 特征替换 / 12% 子树重掷 / 其余单点递归.

        v1 是 40% 根节点整体重掷 — 精英好结构难以局部保留.
        """
        r = rng.random()
        if r < 0.20:
            return self._mutate_windows(e, rng)
        if r < 0.35:
            return self._map_leaves(
                e, lambda lf: ("feat", rng.choice(self.feats_order))
                if lf[0] == "feat" else lf)
        if not isinstance(e, tuple) or e[0] in ("feat", "const") or rng.random() < subtree_reset_p:
            return random_expr(rng, 2, self.feats_order)
        args = list(e[1:])
        idxs = [i for i, a in enumerate(args) if isinstance(a, tuple)]
        if not idxs:
            return random_expr(rng, 2, self.feats_order)
        i = rng.choice(idxs)
        args[i] = self.mutate(args[i], rng, subtree_reset_p)
        return (e[0], *args)

    def crossover(self, a: tuple, b: tuple, rng: random.Random,
                  swap_p: float = 0.15) -> tuple:
        """单点子树交换; 根交换概率降到 15% (v1 为 40%)."""
        if a[0] in ("feat", "const") or rng.random() < swap_p:
            return b if rng.random() < 0.5 else a
        args = list(a[1:])
        idxs = [i for i, x in enumerate(args) if isinstance(x, tuple)]
        if not idxs:
            return b if rng.random() < 0.5 else a
        i = rng.choice(idxs)
        args[i] = b
        return (a[0], *args)

    # ---------- 锦标赛选择 ----------
    @staticmethod
    def tournament(pop: list[tuple], fits: np.ndarray,
                   rng: random.Random, k: int = 5) -> tuple:
        idxs = rng.sample(range(len(pop)), min(k, len(pop)))
        return pop[max(idxs, key=lambda i: fits[i])]

    # ---------- 分桶精英合并 ----------
    @staticmethod
    def merge_elite_bucketed(elite_by_name: dict[str, tuple],
                             cand_trees: list[tuple],
                             cand_fit: dict[str, float],
                             per_bucket_k: int = 8,
                             total_cap: int = 60) -> dict[str, tuple]:
        pool = {}
        for e in cand_trees + list(elite_by_name.values()):
            pool[expr_str(e)] = e
        # 结构签名去重: 同签名只留 fitness 最高的变体 (DEAD 整树除外)
        sig_fit: dict[str, float] = {}
        sig_tree: dict[str, tuple] = {}
        for s_, e_ in pool.items():
            sig = structural_signature(e_)
            f_ = cand_fit.get(s_, -9e9)
            if sig == "DEAD":
                continue
            if sig not in sig_fit or f_ > sig_fit[sig]:
                sig_fit[sig] = f_
                sig_tree[sig] = e_
        pool = {expr_str(e): e for e in sig_tree.values()}
        buckets: dict[str, list] = defaultdict(list)
        for s, e in pool.items():
            buckets[bucket_of(e)].append((cand_fit.get(s, -9e9), s, e))
        merged: dict[str, tuple] = {}
        for items in buckets.values():
            items.sort(key=lambda x: -x[0])
            for _, s, e in items[:per_bucket_k]:
                merged[s] = e
        if len(merged) > total_cap:
            ranked = sorted(merged.items(),
                            key=lambda kv: cand_fit.get(kv[0], -9e9),
                            reverse=True)
            merged = dict(ranked[:total_cap])
        return merged

    # ---------- 搜索主循环 ----------
    def genetic_search(self, n_pop: int = 400, generations: int = 4,
                       seed: int = 42, refine_top: int = 50,
                       ) -> tuple[pd.DataFrame, torch.Tensor]:
        rng = random.Random(seed)

        def fresh_pop(n: int) -> list[tuple]:
            seen, out = set(), []
            while len(out) < n:
                e = random_expr(rng, 3, self.feats_order)
                s = expr_str(e)
                if s not in seen:
                    seen.add(s)
                    out.append(e)
            return out

        pop = fresh_pop(n_pop)
        elite_by_name: dict[str, tuple] = {}

        for gen in range(generations):
            t0 = time.perf_counter()
            df, S = self.evaluate_with_fitness(pop)
            dt = time.perf_counter() - t0
            df = df.sort_values("fitness", ascending=False).reset_index(drop=True)
            print(f"  [gen {gen}] 种群 {len(pop)} | GPU {dt:.1f}s | "
                  f"top fit {df['fitness'].iloc[0]:+.3f} "
                  f"(ICIR {df['ICIR'].iloc[0]:+.3f}, TO {df['turnover'].iloc[0]:.3f})")

            fits = dict(zip(df["factor"], df["fitness"]))
            names2tree = {expr_str(e): e for e in pop}
            elite_by_name = self.merge_elite_bucketed(
                elite_by_name, list(names2tree.values()), fits)

            # 锦标赛选父代 (按复合适应度), 温和变异/交叉/新鲜随机
            ordered_names = df["factor"].tolist()
            fit_vec = df["fitness"].to_numpy()
            tree_of = {expr_str(e): e for e in pop}
            ordered_trees = [tree_of[n] for n in ordered_names]

            new_pop, seen = [], set(ordered_trees and [expr_str(e) for e in ordered_trees])
            while len(new_pop) < n_pop:
                r = rng.random()
                if r < 0.35 and ordered_trees:
                    child = self.mutate(self.tournament(ordered_trees, fit_vec, rng), rng)
                elif r < 0.55 and len(elite_by_name) >= 2:
                    ea, eb = rng.sample(list(elite_by_name.values()), 2)
                    child = self.crossover(ea, eb, rng)
                else:
                    child = random_expr(rng, 3, self.feats_order)
                s = expr_str(child)
                if s not in seen:
                    seen.add(s)
                    new_pop.append(child)
            pop = new_pop

        # 终评: 精英并集
        final_exprs = list(elite_by_name.values())
        df, S = self.evaluate_with_fitness(final_exprs)
        df["expr"] = final_exprs
        df = df.sort_values("fitness", ascending=False).reset_index(drop=True)

        # ---- 第二阶段: 窗口网格精修 ----
        if refine_top > 0 and len(df) > 0:
            top_trees = df["expr"].tolist()[:refine_top]
            print(f"  [refine] top{len(top_trees)} 窗口×常量网格精修 "
                  f"(每树≤64变体, 批量GPU)")
            ref_df, ref_S = self.refine(top_trees)
            if ref_S is not None:
                keep = ~ref_df["factor"].isin(set(df["factor"]))
                # 全局签名去重: 精修变体间 + 变体与原精英间
                df = pd.concat([df, ref_df[keep]], ignore_index=True)
                S = torch.cat([S, ref_S[keep.to_numpy()]], dim=0)
                df = df.sort_values("ICIR", ascending=False).reset_index(drop=True)
                seen_sig: set[str] = set()
                keep_rows = []
                for i, row in df.iterrows():
                    # expr 列只在终评 df 有; 变体行用 factor 字符串反查树不可行,
                    # 直接对 factor 字符串做常量归一化签名 (轻量近似)
                    import re as _re
                    sig = _re.sub(r"-?\d+\.\d+", "<C>", row["factor"])
                    # 剥掉截面排序外壳 (不改横截面排序): 正确配对右括号
                    for _ in range(4):
                        m = _re.search(r"(cs_rank|cs_zscore|cs_demean)\(", sig)
                        if not m:
                            break
                        start, depth, j = m.end(), 1, m.end()
                        while j < len(sig) and depth:
                            if sig[j] == "(":
                                depth += 1
                            elif sig[j] == ")":
                                depth -= 1
                                if depth == 0:
                                    break
                            j += 1
                        inner = sig[start:j]
                        sig = sig[:m.start()] + inner + sig[j + 1:]
                    # 窗口/常量归一
                    sig = _re.sub(r"(?<![\w])\b(5|10|20|60|120|1)\b(?!\d)", "<N>", sig)
                    if sig in seen_sig:
                        continue
                    seen_sig.add(sig)
                    keep_rows.append(i)
                df = df.loc[keep_rows].reset_index(drop=True)
                S = S[torch.tensor(keep_rows, device=S.device)]
                df = df.sort_values("fitness", ascending=False).reset_index(drop=True)
        # 返回前裁剪: 调用方只用 top60, S 保留对应行即可 (省 ~97% 面板内存)
        df = df.reset_index(drop=True)
        if "degenerate" not in df.columns:
            df["degenerate"] = False
        S = S[df.head(60).index.to_numpy()]
        df = df.head(60).reset_index(drop=True)
        return df, S

    # ---------- 网格精修 ----------
    def refine(self, trees: list[tuple]) -> tuple[pd.DataFrame, torch.Tensor | None]:
        variants: list[tuple] = []
        owner: list[int] = []
        for ti, t in enumerate(trees):
            for v in _enumerate_remaps(t, cap=64):
                variants.append(v)
                owner.append(ti)
        if not variants:
            return pd.DataFrame(), None
        # 分批评估防 OOM: 每批评估后只保留每 parent 的最优行,
        # 不累积全部 chunk (峰值从 全部变体+cat结果 降为 1批+每parent 1行)
        BATCH = 256
        best_rows = []          # 每 parent 当前最优行 (df)
        best_pos_in_batch = []  # 对应批内行号
        owner_of_best = []
        for pi in range(0, len(variants), BATCH):
            db, Sb = self.evaluate_with_fitness(variants[pi:pi + BATCH])
            db["parent_id"] = owner[pi:pi + len(db)]
            # 每批内: 先剔退化变体, 每 parent 取 fitness 最优
            if "degenerate" not in db.columns:
                db["degenerate"] = False
            db = db[~db["degenerate"].astype(bool)]
            if db.empty:
                del db
                continue
            bi = db.groupby("parent_id")["fitness"].idxmax().to_numpy()
            best_rows.append(db.loc[bi])
            best_pos_in_batch.append(bi - pi)   # 批内行号
            owner_of_best.append(Sb[bi - pi])   # 对应 S 行 (引用, 未拷贝)
            del db, Sb
            gc.collect()
            torch.cuda.empty_cache()
        dfv = pd.concat(best_rows, ignore_index=True)
        Sv = torch.cat(owner_of_best, dim=0)
        del best_rows, owner_of_best, best_pos_in_batch
        gc.collect()
        return dfv, Sv

    # ---------- 附加评估 ----------
    def extras(self, S: torch.Tensor, names: list[str],
               fwd: torch.Tensor) -> tuple[pd.DataFrame, pd.DataFrame]:
        q = quantile_spread(S, fwd, names)
        t = turnover(S, names)
        return q, t
