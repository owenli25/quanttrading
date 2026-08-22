"""
features.py — 数据获取 + 预处理 + GPU 特征面板

产出: 一组 [n_assets, n_days] 的 GPU 特征张量 (float32),
      以及对应的 pandas 索引 (日期/代码) 用于回映射。

缓存: data/ 下 parquet, 避免重复下载 (yfinance 逐标的下载慢)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent  # D:\quant
sys.path.insert(0, str(ROOT / "scripts"))

import factor_lib as fl  # noqa: E402


# ---------- 标的池: AI 资本开支链条 + 跨行业大盘 (美股) ----------
UNIVERSE = [
    # === AI 芯片 / 半导体 ===
    "NVDA", "AMD", "MU", "AVGO", "TSM", "ASML", "QCOM", "INTC", "ARM", "AMAT",
    "LRCX", "KLAC", "TXN", "MRVL", "ADI", "NXPI", "ON", "MCHP", "MPWR", "SWKS",
    "ALAB", "CRDO", "WDC", "STX", "ENTG", "GFS", "UMC",
    # === AI 网络 / 光互连 (server 侧) ===
    "ANET", "CIEN", "COHR", "LITE",
    # === AI Server / ODM / 基础设施 ===
    "SMCI", "DELL", "HPE", "VRT",
    # === Neocloud / AI 数据中心 (含矿场转型) ===
    "CRWV", "NBIS", "CRCL", "APLD", "IREN", "CORZ", "WULF", "CIFR", "HUT", "BTDR",
    # === Clean Energy (Talon 8/17 主题) ===
    "BE", "RUN", "FSLR", "ENPH",
    # === Fintech / Crypto (Talon 8/17) ===
    "COIN", "SOFI", "SQ",
    # === BTC 矿工 (Talon 8/17) ===
    "CLSK", "RIOT", "MARA",
    # === Materials (Talon 8/17) ===
    "AA", "X",
    # === Quantum Computing (Talon 8/17) ===
    "IONQ", "RGTI", "QBTS",
    # === China ADR (Talon 8/17) ===
    "PDD", "XPEV", "TCOM", "LI", "BIDU",
    # === EV (Talon 8/17) ===
    "RIVN", "LCID", "GM",
    # === Nuclear Energy (Talon 8/17) ===
    "CEG", "SMR", "VST",
    # === Gold Miners / Cybersecurity (Talon 8/17) ===
    "KGC", "OKTA",
    # === 大科技 ===
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "ORCL",
    "CRM", "ADBE", "IBM", "CSCO", "NOW", "PLTR", "UBER", "SHOP", "SNOW",
    # === 金融 ===
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "V", "MA", "PYPL",
    # === 能源/材料 ===
    "XOM", "CVX", "COP", "SLB", "EOG", "FCX", "NEM", "LIN",
    # === 医疗 ===
    "LLY", "UNH", "JNJ", "MRK", "PFE", "ABBV", "TMO", "AMGN", "GILD", "ISRG", "SYK",
    # === 消费 ===
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "MELI", "LULU", "HD", "LOW",
    # === 工业/航空/国防 ===
    "BA", "CAT", "DE", "GE", "HON", "LMT", "RTX", "UNP", "UPS", "FDX", "ETN",
    # === 通信/媒体 ===
    "T", "VZ", "CMCSA", "DIS", "WBD", "TMUS",
    # === ETF 基准 ===
    "SPY", "QQQ", "IWM", "TLT", "DIA", "SOXX", "XLF", "XLE", "XLV",
    # === 杠杆 ETF: 市场杠杆率代理 ===
    "SOXL",
]


def load_data(tickers: list[str], start: str = "2019-01-01", cache: bool = True) -> dict[str, pd.DataFrame]:
    """下载/读取 OHLCV 面板, 返回 {field: DataFrame[date x asset]}."""
    cache_dir = ROOT / "data"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"ohlcv_{len(tickers)}_{start[:4]}.parquet"

    if cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        import yfinance as yf
        print(f"  下载 {len(tickers)} 标的 (起始 {start}) ...")
        data = yf.download(tickers, start=start, auto_adjust=False, progress=False)
        # 展平 MultiIndex 列: (field, ticker) -> ticker
        frames = {}
        for field in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]:
            if field in data:
                sub = data[field]
                if isinstance(sub.columns, pd.MultiIndex):
                    sub.columns = sub.columns.get_level_values(0)
                frames[field] = sub[tickers]
        df = pd.concat(frames, axis=1, keys=frames.keys())
        if cache:
            df.to_parquet(cache_path)
            print(f"  已缓存 -> {cache_path}")
    return {f: df[f] for f in df.columns.get_level_values(0).unique()}


def preprocess(close: pd.DataFrame, volume: pd.DataFrame,
               other: dict[str, pd.DataFrame], warmup: int = 300,
               max_nan_frac: float = 0.5) -> dict[str, pd.DataFrame]:
    """ffill + warmup 截断 + 截面中位数填充.

    新上市标的 (CRWV/NBIS 等): 首个有效日之前的区域保留 NaN,
    仅对"当日缺失率 < max_nan_frac"的行做中位数填充,
    避免用假中位数数据污染因子 (因子层会将其清零, 截面中性).
    """
    out = {}
    for name, df in [("Close", close), ("Volume", volume), *other.items()]:
        d = df.ffill().iloc[warmup:]
        # 每列首个有效日之前的区域 -> NaN
        for col in d.columns:
            first = d[col].first_valid_index()
            if first is not None:
                d.loc[d.index < first, col] = np.nan
        # 按行缺失率决定是否中位数填充
        nan_frac = d.isna().mean(axis=1)
        d = d.apply(lambda row: row.fillna(row.median())
                    if nan_frac.loc[row.name] < max_nan_frac else row, axis=1)
        out[name] = d
    return out


class FeaturePanel:
    """GPU 特征面板: 持有 [A,D] 张量 + 元数据. 所有算子经 factor_lib (GPU)."""

    def __init__(self, frames: dict[str, pd.DataFrame], device=fl.DEVICE):
        self.frames = frames
        self.device = device
        self.dates = frames["Close"].index
        self.tickers = list(frames["Close"].columns)
        self.n_assets, self.n_days = len(self.tickers), len(self.dates)

    def tensor(self, name: str) -> torch.Tensor:
        return torch.tensor(self.frames[name].values.T, dtype=torch.float32, device=self.device)

    def build_features(self) -> dict[str, torch.Tensor]:
        """计算全部基础特征 [A,D] 张量."""
        c = self.tensor("Close")
        v = self.tensor("Volume")
        o = self.tensor("Open")
        h = self.tensor("High")
        lo = self.tensor("Low")

        def ret(n: int) -> torch.Tensor:
            r = torch.zeros_like(c)
            if self.n_days > n:
                r[:, n:] = c[:, n:] / c[:, :-n] - 1.0
            return r

        f = {}
        for n in (1, 5, 10, 20, 60):
            f[f"ret_{n}"] = ret(n)
        # 收益波动率
        r1 = f["ret_1"]
        f["vol_20"] = fl.rolling_std(r1, 20)
        f["vol_60"] = fl.rolling_std(r1, 60)
        # 量能
        f["vol_z"] = fl.rolling_zscore(v, 20)
        f["vol_ratio_5_20"] = fl.rolling_mean(v, 5) / (fl.rolling_mean(v, 20) + 1e-8)
        # 日内位置: (close-low)/(high-low), 滚动均值
        rng = (h - lo) + 1e-8
        f["pos_in_range"] = fl.rolling_mean((c - lo) / rng, 10)
        # 跳空: open/prev_close - 1
        gap = torch.zeros_like(c)
        gap[:, 1:] = o[:, 1:] / c[:, :-1] - 1.0
        f["gap"] = gap
        # Amihud 非流动性 (负)
        f["amihud"] = fl.rolling_mean(r1.abs() / (v + 1e-8), 20)
        # 成交额 zscore
        f["dollar_vol"] = fl.rolling_zscore(c * v, 20)
        # 动量 (12-1): mom[t] = close[t-21]/close[t-253] - 1
        f["mom_12_1"] = torch.zeros_like(c)
        if self.n_days > 253:
            import torch.nn.functional as F
            num = F.pad(c[:, : self.n_days - 21], (21, 0))
            den = F.pad(c[:, : self.n_days - 253], (253, 0))
            f["mom_12_1"] = num / (den + 1e-8) - 1.0
        # RSI 型
        up = torch.clamp(r1, min=0)
        dn = torch.clamp(-r1, min=0)
        f["rsi_14"] = fl.rolling_mean(up, 14) / (fl.rolling_mean(up, 14) + fl.rolling_mean(dn, 14) + 1e-8)
        # MACD (12/26/9): 线 + 柱 (信号线本身信息冗余, 用柱即可)
        macd_line, _, macd_hist = fl.macd(c)
        f["macd"] = macd_line
        f["macd_hist"] = macd_hist
        # VWAP (日线用典型价近似): 20日滚动 VWAP 偏离度
        tp = (h + lo + c) / 3.0
        vwap20 = fl.rolling_mean(tp * v, 20) / (fl.rolling_mean(v, 20) + 1e-8)
        f["vwap_dev"] = c / vwap20 - 1.0
        # OBV: 累积量能指标 -> 20日 zscore (量能动量)
        obv = torch.cumsum(torch.sign(r1) * v, dim=1)
        f["obv_z"] = fl.rolling_zscore(obv, 20)
        # 动量增强 (区别于 ret_n 原始收益):
        #   rs_20 相对强度 = 20日收益 - 截面中位数 (相对市场动量)
        f["rs_20"] = f["ret_20"] - f["ret_20"].median(dim=0, keepdim=True).values
        #   mom_accel 动量加速度 = 20日动量 - 5日动量 (短长差)
        f["mom_accel"] = f["ret_20"] - f["ret_5"]
        # 布林带 (20日, 2倍标准差):
        ma20 = fl.rolling_mean(c, 20)
        sd20 = fl.rolling_std(c, 20)
        #   bb_pos_20 带内位置: 0=中轨, +1=上轨, -1=下轨 (均值回归信号)
        f["bb_pos_20"] = (c - ma20) / (2.0 * sd20 + 1e-8)
        #   bb_width_20 带宽: 波动率压缩/扩张状态
        f["bb_width_20"] = sd20 / (ma20 + 1e-8)
        return f


FEATURES_ORDER = [
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    "vol_20", "vol_60", "vol_z", "vol_ratio_5_20",
    "pos_in_range", "gap", "amihud", "dollar_vol", "mom_12_1", "rsi_14",
    "macd", "macd_hist", "vwap_dev", "obv_z",
    "rs_20", "mom_accel", "bb_pos_20", "bb_width_20",
]
