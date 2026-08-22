"""
factor_lib.py — GPU 加速因子原语 (PyTorch / ROCm)

设计约定:
- 张量形状: [n_assets, n_days] (资产为行, 日期为列) — 截面操作沿 dim=0, 时序操作沿 dim=1
- 所有输入先经 NaN 处理 (ffill + warmup + 截面中位数填充), 保证 GPU 上无 NaN
- 设备: 自动选择 cuda (ROCm) 否则 CPU
"""
from __future__ import annotations

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8


def to_tensor(df, device=DEVICE):
    """pandas DataFrame [date x asset] -> torch [asset x date]."""
    return torch.tensor(df.values.T, dtype=torch.float32, device=device)


def to_df(t: torch.Tensor, df):
    """torch [asset x date] -> pandas DataFrame [date x asset]."""
    return df.T  # placeholder; 调用方直接用 DataFrame(t.cpu().numpy().T, index=..., columns=...)


# ---------- 时序滚动原语 (沿 dim=1, 资产并行) ----------

def _cumsum(x: torch.Tensor, window: int):
    """返回 S[t] = sum(x[t-w+1 .. t]) 的滚动和, 边界处为累积和."""
    S = torch.cumsum(x, dim=1)
    S_shift = torch.zeros_like(S)
    if window > 1:
        S_shift[:, window:] = S[:, :-window]
    else:
        S_shift = S
    return S - S_shift


def rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    return _cumsum(x, window) / min(window, x.shape[1])


def rolling_std(x: torch.Tensor, window: int, ddof: int = 0) -> torch.Tensor:
    n = min(window, x.shape[1])
    m = rolling_mean(x, window)
    m2 = _cumsum(x * x, window) / n
    v = m2 - m * m
    if ddof == 1:
        v = v * n / (n - 1)
    return torch.sqrt(torch.clamp(v, min=0)) + EPS


def rolling_zscore(x: torch.Tensor, window: int) -> torch.Tensor:
    """滚动 z-score: (x - rolling_mean) / rolling_std, 无量纲."""
    m = rolling_mean(x, window)
    s = rolling_std(x, window)
    return (x - m) / s


def ema(x: torch.Tensor, span: int) -> torch.Tensor:
    """指数移动平均 (有限窗口指数卷积, 资产并行).

    alpha = 2/(span+1), 窗口取 5*span (截断误差 < (1-alpha)^W, 可忽略).
    左侧边界用零填充近似, warmup 之后收敛, 无实际影响.
    """
    import torch.nn.functional as F
    W = min(5 * span, x.shape[1])
    alpha = 2.0 / (span + 1.0)
    w = torch.pow(1.0 - alpha, torch.arange(W, device=x.device, dtype=x.dtype))
    w = w / w.sum()
    x_pad = F.pad(x, (W - 1, 0))
    kernel = w.view(1, 1, W)
    return F.conv1d(x_pad.unsqueeze(1), kernel, padding=0).squeeze(1)


def macd(x: torch.Tensor, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD: (ema_fast - ema_slow, 信号线, 柱). 返回 (macd, signal, hist)."""
    line = ema(x, fast) - ema(x, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def rolling_corr(a: torch.Tensor, b: torch.Tensor, window: int) -> torch.Tensor:
    """两序列滚动 Pearson 相关 (沿时间)."""
    n = min(window, a.shape[1])
    ma, mb = rolling_mean(a, window), rolling_mean(b, window)
    eab = _cumsum(a * b, window) / n
    ea, eb = rolling_mean(a, window), rolling_mean(b, window)
    cov = eab - ea * eb
    va = rolling_std(a, window) ** 2
    vb = rolling_std(b, window) ** 2
    return cov / (torch.sqrt(va * vb) + EPS)


def rolling_max(x: torch.Tensor, window: int) -> torch.Tensor:
    """滚动最大值 (unfold 实现, window 较小时高效)."""
    import torch.nn.functional as F
    n = x.shape[1]
    if window >= n:
        return x.max(dim=1, keepdim=True).values.expand_as(x)
    xp = F.pad(x, (window - 1, 0), value=-float("inf"))
    return xp.unfold(1, window, 1).max(dim=2).values


def ts_rank(x: torch.Tensor, window: int) -> torch.Tensor:
    """时序排名 (percentile): x[t] 在过去 window 天窗口内的分位 [0,1].
    向量化: 对每个滞后 k, 统计 x[t] > x[t-k] 的次数."""
    count = torch.zeros_like(x)
    for k in range(1, min(window, x.shape[1])):
        count[:, k:] += (x[:, k:] > x[:, :-k]).float()
    n = min(window, x.shape[1])
    return count / max(n - 1, 1)


# ---------- 截面原语 (沿 dim=0) ----------

def cs_rank(x: torch.Tensor) -> torch.Tensor:
    """截面排名, 归一化到 [0,1]. 每个日期内对资产排名."""
    n = x.shape[0]
    idx = torch.argsort(torch.argsort(x, dim=0), dim=0).float()
    return (idx + 0.5) / n


def cs_zscore(x: torch.Tensor) -> torch.Tensor:
    """截面 z-score (按日去均值除标准差)."""
    m = x.mean(dim=0, keepdim=True)
    s = x.std(dim=0, keepdim=True) + EPS
    return (x - m) / s


def cs_standardize(x: torch.Tensor) -> torch.Tensor:
    """截面中性化: z-score 后再按截面 rank 变换 (去极值)."""
    r = cs_rank(x)
    return cs_zscore(r)


# ---------- 经典因子 (输入: close/volume 面板) ----------

def factor_momentum(close: torch.Tensor, lookback: int = 252, skip: int = 21) -> torch.Tensor:
    """12-1 动量: close[t-skip] / close[t-lookback-skip] - 1, 前 lookback+skip 天为 0."""
    import torch.nn.functional as F
    past = lookback + skip
    n = close.shape[1]
    out = torch.zeros_like(close)
    if n > past:
        # num[t] = close[t-skip] (t>=skip), den[t] = close[t-past] (t>=past)
        num = F.pad(close[:, : n - skip], (skip, 0))
        den = F.pad(close[:, : n - past], (past, 0))
        out = num / (den + EPS) - 1.0
        out[:, :past] = 0.0
    return out


def factor_reversal_5d(close: torch.Tensor) -> torch.Tensor:
    """5 日短期反转: 过去 5 天收益取负 (反转因子 = -ret)."""
    ret = close[:, 1:] / close[:, :-1] - 1
    r5 = torch.cat([torch.zeros_like(ret[:, :1]), ret], dim=1)
    r5 = _cumsum(r5, 5)
    return -r5


def factor_volatility(close: torch.Tensor, window: int = 20) -> torch.Tensor:
    """20 日已实现波动率 (低波因子取负)."""
    ret = torch.zeros_like(close)
    ret[:, 1:] = close[:, 1:] / close[:, :-1] - 1
    return rolling_std(ret, window)


def factor_volume_trend(volume: torch.Tensor, fast: int = 20, slow: int = 60) -> torch.Tensor:
    """量能趋势: 快线均量 / 慢线均量."""
    return rolling_mean(volume, fast) / (rolling_mean(volume, slow) + EPS)


def factor_amihud(close: torch.Tensor, volume: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Amihud 非流动性: mean(|ret| / volume), 取负后为流动性因子."""
    ret = torch.zeros_like(close)
    ret[:, 1:] = (close[:, 1:] / close[:, :-1] - 1).abs()
    illiq = rolling_mean(ret / (volume + EPS), window)
    return -illiq


def factor_rsi(close: torch.Tensor, window: int = 14) -> torch.Tensor:
    """RSI 型动量 (简化版, 用收益符号的滚动占比)."""
    ret = torch.zeros_like(close)
    ret[:, 1:] = close[:, 1:] / close[:, :-1] - 1
    up = torch.clamp(ret, min=0)
    dn = torch.clamp(-ret, min=0)
    rsi = rolling_mean(up, window) / (rolling_mean(up, window) + rolling_mean(dn, window) + EPS)
    return rsi


FACTOR_FUNCS = {
    "momentum_12_1": lambda c, v: factor_momentum(c),
    "reversal_5d": lambda c, v: factor_reversal_5d(c),
    "volatility_20d": lambda c, v: -factor_volatility(c),  # 低波为正
    "volume_trend_20_60": lambda c, v: factor_volume_trend(v),
    "amihud_illiq_neg": lambda c, v: factor_amihud(c, v),
    "rsi_14": lambda c, v: factor_rsi(c),
}


def compute_factors(close: torch.Tensor, volume: torch.Tensor,
                    names: list[str] | None = None) -> dict[str, torch.Tensor]:
    """批量计算因子. close/volume 为 [asset, day] 张量."""
    names = names or list(FACTOR_FUNCS.keys())
    out = {}
    for name in names:
        f = FACTOR_FUNCS[name](close, volume)
        # 因子统一做截面标准化 + 时序 NaN 安全 (前 warmup 期为 NaN 已在外部填充)
        out[name] = cs_standardize(f)
    return out


def gpu_timing_benchmark(n_assets: int = 500, n_days: int = 2500, window: int = 60,
                         trials: int = 5) -> dict:
    """GPU vs CPU 滚动 z-score 基准. 返回每批耗时 (ms)."""
    x = torch.randn(n_assets, n_days, device=DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(trials):
            rolling_zscore(x, window)
        t1.record(); torch.cuda.synchronize()
        gpu_ms = t0.elapsed_time(t1) / trials
    else:
        gpu_ms = None
    xc = x.cpu()
    import time
    t0 = time.perf_counter()
    for _ in range(trials):
        rolling_zscore(xc, window)
    cpu_ms = (time.perf_counter() - t0) * 1000 / trials
    return {"device": str(DEVICE), "gpu_ms": gpu_ms, "cpu_ms": cpu_ms,
            "speedup": (cpu_ms / gpu_ms) if gpu_ms else None}


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    b = gpu_timing_benchmark()
    print(f"Benchmark [500 assets x 2500 days, window=60]:")
    print(f"  GPU: {b['gpu_ms']:.2f} ms/batch" if b['gpu_ms'] else "  GPU: N/A")
    print(f"  CPU: {b['cpu_ms']:.2f} ms/batch")
    if b['speedup']:
        print(f"  Speedup: {b['speedup']:.1f}x")
