"""
expressions.py — 因子表达式: 语法树, 随机生成, GPU 评估

表达式 = 嵌套元组 ("op", args...), 叶子为 ("feat", name) 或 ("const", value)。
评估在 GPU 上进行, 共享子树用 memo 缓存 (种群内大量重复子结构)。

算子集:
  单目  cs_rank / cs_zscore / cs_demean / sign / abs / neg / log1p
        ts_zscore(x,w) / ts_rank(x,w) / ts_mean(x,w) / ts_std(x,w)
        ts_delta(x,d) / ts_max(x,w) / ts_min(x,w)
  双目  add / sub / mul / div / max / min / ts_corr(x,y,w)
"""
from __future__ import annotations

import random

import torch

import factor_lib as fl

EPS = 1e-8

WINDOWS = (5, 10, 20, 60, 120)
DELTAS = (1, 5, 10, 20)
CONSTS = (0.5, 1.0, 2.0, 3.0, -1.0)

UNARY_OPS = ["cs_rank", "cs_zscore", "cs_demean", "sign", "abs", "neg", "log1p",
             "ts_zscore", "ts_rank", "ts_mean", "ts_std", "ts_delta", "ts_max", "ts_min"]
BINARY_OPS = ["add", "sub", "mul", "div", "max", "min", "ts_corr"]


# ---------- 随机生成 ----------

def random_expr(rng: random.Random, depth: int, feats: list[str]) -> tuple:
    if depth <= 0 or rng.random() < 0.35:
        if rng.random() < 0.8:
            return ("feat", rng.choice(feats))
        return ("const", rng.choice(CONSTS))
    if rng.random() < 0.5:
        op = rng.choice(UNARY_OPS)
        x = random_expr(rng, depth - 1, feats)
        if op.startswith("ts_") and op not in ("ts_delta",):
            return (op, x, rng.choice(WINDOWS))
        if op == "ts_delta":
            return (op, x, rng.choice(DELTAS))
        return (op, x)
    op = rng.choice(BINARY_OPS)
    x = random_expr(rng, depth - 1, feats)
    y = random_expr(rng, depth - 1, feats)
    if op == "ts_corr":
        return (op, x, y, rng.choice(WINDOWS))
    return (op, x, y)


def expr_str(e: tuple) -> str:
    op = e[0]
    if op == "feat":
        return e[1]
    if op == "const":
        return str(e[1])
    args = " ".join(expr_str(a) if isinstance(a, tuple) else str(a) for a in e[1:])
    return f"{op}({args})"


# ---------- GPU 评估 ----------

class Evaluator:
    def __init__(self, features: dict[str, torch.Tensor], device=fl.DEVICE):
        self.features = features
        self.device = device
        self._memo: dict[tuple, torch.Tensor] = {}
        # 面板形状 (取自第一个特征) — const 需要 [A,D] 全尺寸张量
        self.shape = next(iter(features.values())).shape

    def clear(self):
        self._memo.clear()

    def eval(self, e: tuple) -> torch.Tensor:
        cached = self._memo.get(e)
        if cached is not None:
            return cached
        x = self._eval(e)
        self._memo[e] = x
        return x

    def _eval(self, e: tuple) -> torch.Tensor:
        op = e[0]
        if op == "feat":
            return self.features[e[1]]
        if op == "const":
            return torch.full(self.shape, float(e[1]), device=self.device)

        if op in ("cs_rank", "cs_zscore", "cs_demean", "sign", "abs", "neg", "log1p"):
            a = self.eval(e[1])
            if op == "cs_rank":
                return fl.cs_rank(a)
            if op == "cs_zscore":
                return fl.cs_zscore(a)
            if op == "cs_demean":
                return a - a.mean(dim=0, keepdim=True)
            if op == "sign":
                return torch.sign(a)
            if op == "abs":
                return a.abs()
            if op == "neg":
                return -a
            return torch.log1p(torch.clamp(a, min=-0.9, max=1e6))

        if op in ("ts_zscore", "ts_rank", "ts_mean", "ts_std", "ts_max", "ts_min", "ts_delta"):
            a = self.eval(e[1])
            w = e[2]
            if op == "ts_zscore":
                return fl.rolling_zscore(a, w)
            if op == "ts_rank":
                return fl.ts_rank(a, w)
            if op == "ts_mean":
                return fl.rolling_mean(a, w)
            if op == "ts_std":
                return fl.rolling_std(a, w)
            if op == "ts_max":
                return _rolling_max(a, w)
            if op == "ts_min":
                return -_rolling_max(-a, w)
            return _ts_delta(a, w)

        if op in ("add", "sub", "mul", "div", "max", "min", "ts_corr"):
            a = self.eval(e[1])
            b = self.eval(e[2])
            if op == "add":
                return a + b
            if op == "sub":
                return a - b
            if op == "mul":
                return a * b
            if op == "div":
                return a / (b.abs() + EPS)
            if op == "max":
                return torch.maximum(a, b)
            if op == "min":
                return torch.minimum(a, b)
            return fl.rolling_corr(a, b, e[3])
        raise ValueError(f"未知算子: {op}")


def _rolling_max(x: torch.Tensor, w: int) -> torch.Tensor:
    """滚动最大值 (用 unfold, w 小的时候高效)."""
    n = x.shape[1]
    if w >= n:
        return x.max(dim=1, keepdim=True).values.expand_as(x)
    pad = w - 1
    xp = torch.nn.functional.pad(x, (pad, 0), value=-float("inf"))
    windows = xp.unfold(1, w, 1)  # [A, n, w]
    return windows.max(dim=2).values


def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    if d < x.shape[1]:
        out[:, d:] = x[:, d:] - x[:, :-d]
    return out
