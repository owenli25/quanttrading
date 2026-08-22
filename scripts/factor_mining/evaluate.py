# -*- coding: utf-8 -*-
"""
evaluate.py — GPU 批量因子评估

核心: 把 N 个因子堆成 [N, A, D] 张量, 一次性截面排名 + Spearman IC 计算,
      完全避免逐因子小内核启动 (适配 DXG 高启动开销环境)。

产出: 每个因子的 mean_IC / ICIR / t_stat / IC_positive_pct / n_days,
      可选分层价差 (Q5-Q1) 与换手率。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import factor_lib as fl


def stack_factors(factors: list[torch.Tensor], device=fl.DEVICE) -> torch.Tensor:
    """[N,A,D] 堆叠 (每个因子已含 NaN 处理)."""
    return torch.stack(factors, dim=0)


def _rank_along_assets(x: torch.Tensor) -> torch.Tensor:
    """沿资产维排名归一化到 [0,1]; 支持 [A,D] 或 [N,A,D]."""
    return fl.cs_rank(x) if x.dim() == 2 else _batch_cs_rank(x)


def _batch_cs_rank(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[1]
    idx = torch.argsort(torch.argsort(x, dim=1), dim=1).float()
    return (idx + 0.5) / n


def _batch_cs_rank_chunked(x: torch.Tensor, rows_per_chunk: int = 128) -> torch.Tensor:
    """分块截面排名: 峰值显存从 N 面板排名的 ~5x 瞬态降到 chunk 级."""
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for i in range(0, x.shape[0], rows_per_chunk):
        out[i:i + rows_per_chunk] = _batch_cs_rank(x[i:i + rows_per_chunk])
    return out


def ic_matrix(factor_stack: torch.Tensor, fwd_rank: torch.Tensor,
              want_ranks: bool = False):
    """逐日截面 Spearman IC: [N,D]. fwd_rank 为 [A,D] 排名 [0,1].

    want_ranks=True 时同时返回居中后的 zf [N,A,D], 供换手复用,
    免去第二次双重 argsort (省 ~3x 面板的瞬态显存).
    """
    zf = _batch_cs_rank_chunked(factor_stack)
    zf = zf - zf.mean(dim=1, keepdim=True)
    zr = fwd_rank - fwd_rank.mean(dim=0, keepdim=True)
    num = torch.einsum("nad,ad->nd", zf, zr)
    den = torch.sqrt(torch.einsum("nad->nd", zf * zf) *
                     torch.einsum("ad->d", zr * zr).unsqueeze(0)) + 1e-12
    ic = num / den
    # fwd 尾部 NaN 日期置 NaN (整列非有限 或 排名无方差 的日期剔除)
    valid = torch.isfinite(fwd_rank).all(dim=0) & (fwd_rank.std(dim=0) > 1e-9)
    ic[:, ~valid] = float("nan")
    if want_ranks:
        return ic.cpu().numpy(), zf
    return ic.cpu().numpy()


def ic_summary(ic: np.ndarray, names: list[str], dates: pd.DatetimeIndex,
               min_days: int = 60) -> pd.DataFrame:
    """IC 矩阵 [N,D] -> 汇总表."""
    rows = []
    for i, name in enumerate(names):
        s = pd.Series(ic[i], index=dates).dropna()
        if len(s) < min_days:
            rows.append({"factor": name, "mean_IC": np.nan, "ICIR": np.nan,
                         "t_stat": np.nan, "IC_positive_pct": np.nan, "n_days": len(s)})
            continue
        m, sd = s.mean(), s.std(ddof=1)
        rows.append({"factor": name, "mean_IC": m,
                     "ICIR": m / sd if sd > 0 else np.nan,
                     "t_stat": m / (sd / np.sqrt(len(s))) if sd > 0 else np.nan,
                     "IC_positive_pct": (s > 0).mean() * 100,
                     "n_days": len(s)})
    return pd.DataFrame(rows)


def quantile_spread(factor_stack: torch.Tensor, fwd: torch.Tensor,
                    names: list[str], n_q: int = 5) -> pd.DataFrame:
    """分层收益 Q5-Q1 — 全 GPU 向量化: [N,A,D] 一次算完, 无逐因子循环."""
    device = factor_stack.device
    f_rank = _batch_cs_rank(factor_stack)                      # [N,A,D]
    r = torch.nan_to_num(fwd, nan=0.0)                         # [A,D]
    valid = torch.isfinite(fwd).float()                        # [A,D]
    n_valid = valid.sum(dim=0)                                 # [D]

    lo_mask = (f_rank < 1.0 / n_q).float() * valid             # [N,A,D]
    hi_mask = (f_rank >= 1.0 - 1.0 / n_q).float() * valid

    # 每因子每日: Q1/Q5 平均收益 (分子 einsum / 分母 mask 计数)
    cnt_lo = lo_mask.sum(dim=1)                                # [N,D]
    cnt_hi = hi_mask.sum(dim=1)
    sum_lo = torch.einsum("nad,ad->nd", lo_mask, r)
    sum_hi = torch.einsum("nad,ad->nd", hi_mask, r)

    ok = (cnt_lo >= 10) & (cnt_hi >= 10) & (n_valid > 0).unsqueeze(0)
    q1 = torch.where(ok, sum_lo / cnt_lo.clamp(min=1), torch.tensor(float("nan"), device=device))
    q5 = torch.where(ok, sum_hi / cnt_hi.clamp(min=1), torch.tensor(float("nan"), device=device))

    spread = (q5 - q1) * 100                                   # [N,D] 百分比
    res = []
    for i, name in enumerate(names):
        s_i = spread[i]
        s_i = s_i[torch.isfinite(s_i)]
        if len(s_i) == 0:
            continue
        res.append({"factor": name,
                    "Q1_ret": float(torch.nanmean(q1[i])) * 100,
                    "Q5_ret": float(torch.nanmean(q5[i])) * 100,
                    "Q5_minus_Q1": float(s_i.mean())})
    return pd.DataFrame(res)


def turnover(factor_stack: torch.Tensor, names: list[str],
             ranks: torch.Tensor | None = None) -> pd.DataFrame:
    """Daily avg rank shift (GPU-vectorized). Pass ranks to skip re-sorting."""
    f_rank = ranks if ranks is not None else _batch_cs_rank(factor_stack)
    d = (f_rank[:, :, 1:] - f_rank[:, :, :-1]).abs().mean(dim=(1, 2))
    return pd.DataFrame({"factor": names,
                         "turnover": d.detach().cpu().numpy()})
