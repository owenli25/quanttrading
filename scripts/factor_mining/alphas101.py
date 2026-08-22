"""
alphas101.py — WorldQuant 101 Alphas 精选实现 (GPU 向量化)

输入: 基础张量字典 {open, high, low, close, volume, returns, vwap, adv5, adv20}
输出: {alpha编号: [A,D] 张量}。算子经 factor_lib / expressions 原语。

说明: 为可复现性保留原始公式结构; indneutralize/cap/decay_linear 用简化替代
      (分位数去均值 / 常数 / 滚动均值), 标的池为美股, 与论文 A 股环境不同。
"""
from __future__ import annotations

import torch

import factor_lib as fl

EPS = 1e-8


def build_base(close: torch.Tensor, open_: torch.Tensor, high: torch.Tensor,
               low: torch.Tensor, volume: torch.Tensor) -> dict[str, torch.Tensor]:
    returns = torch.zeros_like(close)
    returns[:, 1:] = close[:, 1:] / close[:, :-1] - 1.0
    vwap = (high + low + close) / 3.0
    adv = {}
    for n in (5, 15, 20):
        adv[f"adv{n}"] = fl.rolling_mean(volume, n)
    b = {"open": open_, "high": high, "low": low, "close": close,
         "volume": volume, "returns": returns, "vwap": vwap, **adv}
    return b


def _cs_std(x: torch.Tensor) -> torch.Tensor:
    return x.std(dim=0, keepdim=True) + EPS


def _decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    """线性衰减均值 (简化)."""
    w = torch.arange(1, d + 1, dtype=torch.float32, device=x.device)
    w = w / w.sum()
    xp = torch.nn.functional.pad(x, (d - 1, 0), value=0.0)
    windows = xp.unfold(1, d, 1)
    return (windows * w).sum(dim=2)


def _signedpower(x: torch.Tensor, p: float) -> torch.Tensor:
    return torch.sign(x) * x.abs().pow(p)


ALPHAS = {}


def alpha(n):
    def deco(fn):
        ALPHAS[f"alpha{n}"] = fn
        return fn
    return deco


@alpha(1)
def a1(b):
    ret, c = b["returns"], b["close"]
    x = torch.where(ret < 0, fl.rolling_std(ret, 20), c)
    return fl.ts_rank(_signedpower(x, 2.0), 5) - 0.5


@alpha(2)
def a2(b):
    return -fl.rolling_corr(fl.cs_rank(_ts_delta(torch.log(b["volume"] + 1), 2)),
                            fl.cs_rank((b["close"] - b["open"]) / (b["open"] + EPS)), 6)


@alpha(3)
def a3(b):
    return -fl.rolling_corr(fl.cs_rank(b["open"]), fl.cs_rank(b["volume"]), 10)


@alpha(4)
def a4(b):
    return -fl.ts_rank(fl.cs_rank(b["low"]), 9)


@alpha(5)
def a5(b):
    vw = fl.rolling_mean(b["vwap"], 10)
    return fl.cs_rank(b["open"] - vw) * (-(fl.cs_rank(b["close"] - b["vwap"]).abs()))


@alpha(6)
def a6(b):
    return -fl.rolling_corr(b["open"], b["volume"], 10)


@alpha(7)
def a7(b):
    c, v = b["close"], b["volume"]
    adv20 = b["adv20"]
    x = -fl.ts_rank((c - fl.rolling_std(c, 20)).abs(), 60) * torch.sign(c - fl.rolling_std(c, 20))
    return torch.where(adv20 < v, x, torch.full_like(c, -1.0))


@alpha(8)
def a8(b):
    o, ret = b["open"], b["returns"]
    s = fl.rolling_mean(o, 5) * fl.rolling_mean(ret, 5)
    s10 = _ts_delta(s, 10)
    return -fl.cs_rank(s10)


@alpha(12)
def a12(b):
    return torch.sign(_ts_delta(b["volume"], 1)) * (-_ts_delta(b["close"], 1))


@alpha(14)
def a14(b):
    return (-fl.cs_rank(_ts_delta(b["returns"], 3))) * fl.rolling_corr(b["open"], b["volume"], 10)


@alpha(18)
def a18(b):
    c, o = b["close"], b["open"]
    x = fl.rolling_std((c - o).abs(), 5) + (c - o) + fl.rolling_corr(c, o, 10)
    return -fl.cs_rank(x)


@alpha(20)
def a20(b):
    o, h, l, c = b["open"], b["high"], b["low"], b["close"]
    return (-fl.cs_rank(o - _ts_delta(h, 1))) * fl.cs_rank(o - _ts_delta(c, 1)) * fl.cs_rank(o - _ts_delta(l, 1))


@alpha(24)
def a24(b):
    c, l = b["close"], b["low"]
    sma100 = fl.rolling_mean(c, 100)
    ratio = _ts_delta(sma100, 100) / (_ts_delta(c, 100) + EPS)
    cond = (ratio < 0.05) | ((ratio - 0.05).abs() < 1e-9)
    return torch.where(cond, -(c - _rolling_min(l, 100)), -_ts_delta(c, 3))


@alpha(26)
def a26(b):
    c = b["close"]
    return (fl.rolling_mean(c, 7) - c) + fl.rolling_corr(b["vwap"], _ts_delta(c, 5), 230)


@alpha(28)
def a28(b):
    h, l, c = b["high"], b["low"], b["close"]
    x = fl.rolling_corr(b["adv20"], l, 5) + ((h + l) / 2.0)
    return fl.cs_zscore(x - c)


@alpha(32)
def a32(b):
    c = b["close"]
    return fl.cs_zscore(fl.rolling_mean(c, 7) - c) + 20.0 * fl.cs_zscore(fl.rolling_corr(b["vwap"], c, 100))


@alpha(34)
def a34(b):
    ret, c = b["returns"], b["close"]
    x = (1 - fl.cs_rank(fl.rolling_std(ret, 2) / (fl.rolling_std(ret, 5) + EPS))) + \
        (1 - fl.cs_rank(_ts_delta(c, 1)))
    return fl.cs_rank(x)


@alpha(40)
def a40(b):
    return (-fl.cs_rank(fl.rolling_std(b["high"], 10))) * fl.rolling_corr(b["high"], b["volume"], 10)


@alpha(41)
def a41(b):
    return torch.sqrt(b["high"] * b["low"] + EPS) - b["vwap"]


@alpha(45)
def a45(b):
    v, c = b["volume"], b["close"]
    return (fl.cs_rank(_ts_delta(torch.log(v + 1), 1)) * (-fl.ts_rank(c, 3)) *
            fl.cs_rank(v / (b["adv20"] + EPS))) * -1.0


@alpha(52)
def a52(b):
    l, v = b["low"], b["volume"]
    ret = b["returns"]
    x = (-_rolling_min(l, 5) + _ts_delta(_rolling_min(l, 5), 5)) * \
        fl.cs_rank((fl.rolling_mean(ret, 240) - fl.rolling_mean(ret, 20)) / 220.0)
    return x * fl.ts_rank(v, 5)


@alpha(53)
def a53(b):
    c, l, h = b["close"], b["low"], b["high"]
    x = (c - l) - (h - c)
    return -_ts_delta(x / ((c - l) + EPS), 9)


@alpha(54)
def a54(b):
    l, c, o, h = b["low"], b["close"], b["open"], b["high"]
    return -((l - c) * o.pow(5)) / (((l - h) * c.pow(5)) + EPS)


@alpha(56)
def a56(b):
    ret = b["returns"]
    x = fl.cs_rank(fl.rolling_mean(ret, 10) / (fl.rolling_mean(fl.rolling_mean(ret, 2), 3) + EPS))
    return -(x * fl.cs_rank(ret))


@alpha(60)
def a60(b):
    c, l, h, v = b["close"], b["low"], b["high"], b["volume"]
    x = ((c - l) - (h - c)) / ((h - l) + EPS)
    return -(2.0 * fl.cs_zscore(fl.cs_rank(x * v)) - fl.cs_zscore(fl.ts_rank(c, 10)))


@alpha(66)
def a66(b):
    l, vw, o, h = b["low"], b["vwap"], b["open"], b["high"]
    x = (l * 0.9 + l * 0.1 - vw) / ((o - (h + l) / 2.0) + EPS)
    return -(fl.cs_rank(_decay_linear(_ts_delta(vw, 3), 7)) +
             fl.ts_rank(_decay_linear(x, 11), 7))


@alpha(85)
def a85(b):
    c, v, o = b["close"], b["volume"], b["open"]
    x = fl.rolling_corr(fl.cs_rank((v * (c - o)) / (c + EPS)), fl.cs_rank(c), 5)
    return fl.cs_rank(x)


@alpha(92)
def a92(b):
    c, l = b["close"], b["low"]
    x = c - fl.rolling_max(c, 60)
    y = _ts_delta(l, 5)
    return fl.cs_zscore(torch.maximum(x, torch.full_like(c, 0.0)) + y)


@alpha(101)
def a101(b):
    c, o = b["close"], b["open"]
    return (c - o) / ((h_l_max(b) - c) + EPS)


def h_l_max(b):
    h, l = b["high"], b["low"]
    return fl.rolling_max(torch.maximum(h, l), 30)


def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    if d < x.shape[1]:
        out[:, d:] = x[:, d:] - x[:, :-d]
    return out


def _rolling_min(x: torch.Tensor, w: int) -> torch.Tensor:
    return -fl.rolling_max(-x, w)
